from util_tf import tf, placeholder


def mlp(x, dim, act=tf.nn.relu, name='mlp'):
    with tf.variable_scope(name):
        x = tf.layers.dense(x, dim*4, activation=act)
        x = tf.layers.dense(x, dim)
    return x


def vAe(tgt, dim_tgt, dim_emb, dim_rep, rnn_layers=2, dropout=0.2, warmup=5e4, accelerate=5e-5, eos=1):
    # tgt : int32 (b, t)  | batchsize, timestep
    # dim_tgt : vocab size
    # dim_emb : model dimension
    # dim_rep : representation dimension

    tgt = placeholder(tf.int32, (None, None), tgt, 'tgt')
    batch_size = tf.shape(tgt)[0]
    dropout = placeholder(tf.float32, (), dropout, 'dropout')

    with tf.variable_scope('length'):
        length = tf.reduce_sum(tf.to_int32(tf.not_equal(tgt, eos)), -1)

    with tf.variable_scope('enc_embed'):
        # (b, t) -> (b, t, dim_emb)
        # same embedding for encoder and decoder
        embed_mtrx = tf.get_variable(name="embed_mtrx", shape=[dim_tgt, dim_emb])
        embed = tf.gather(embed_mtrx, tgt)

    with tf.variable_scope('encode'):
        # (b, t, dim_emb) -> (b, dim_emb)
        x = tf.layers.dropout(embed, dropout)
        stacked_cells = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(dim_emb),
                output_keep_prob=1.0 - dropout,
                variational_recurrent=True,
                dtype=tf.float32)
             for _ in range(rnn_layers)])
        initial_state = stacked_cells.zero_state(batch_size, dtype=tf.float32)
        _, h = tf.nn.dynamic_rnn(stacked_cells, x, sequence_length=length, initial_state=initial_state, swap_memory=True)
        h = h[0]

    with tf.variable_scope('latent'):
        # (b, dim_emb) -> (b, dim_rep)
        mu = mlp(h, dim_rep, name='mu')
        lv = mlp(h, dim_rep, name='lv')
        with tf.name_scope('z'):
            h = z = mu + tf.exp(0.5 * lv) * tf.random_normal(shape=tf.shape(lv))

    with tf.variable_scope('decode'):
        # (b, dim_rep) -> (b, dim_emb)
        x = tf.layers.dropout(embed, dropout) # todo switch to word dropout
        h = mlp(h, dim_emb)
        stacked_cells_ = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(dim_emb),
                output_keep_prob=1.0 - dropout,
                variational_recurrent=True,
                dtype=tf.float32)
             for _ in range(rnn_layers)])
        initial_state_ = tuple(h for _ in range(rnn_layers))
        h, _ = tf.nn.dynamic_rnn(stacked_cells_, x, initial_state=initial_state_, sequence_length=length, swap_memory=True)

    with tf.variable_scope('mask'):
        # (b, t, dim_tgt) -> (b * ?, dim_tgt)
        mask = tf.sequence_mask(length)

    with tf.variable_scope('logit'):
        h = tf.boolean_mask(h, mask)
        h = mlp(h, dim_emb)
        h = tf.layers.dropout(h, dropout)
        logits = tf.layers.dense(h, dim_tgt, name="logit")
        labels = tf.boolean_mask(tgt[:, 1:], mask)

    with tf.variable_scope('prob'):
        prob = tf.nn.softmax(logits)

    with tf.variable_scope('pred'):
        pred = tf.argmax(logits, -1, output_type=tf.int32)

    with tf.variable_scope('acc'):
        acc = tf.reduce_mean(tf.to_float(tf.equal(labels, tf.argmax(logits, -1, output_type=tf.int32))))

    step = tf.train.get_or_create_global_step()
    with tf.variable_scope('loss'):
        with tf.name_scope('loss_gen'):
            loss_gen = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        with tf.name_scope('loss_kld'):
            loss_kld = 0.5 * tf.reduce_mean(
                tf.reduce_sum(tf.square(mu) + tf.exp(lv) - lv - 1.0, axis=1))
        with tf.name_scope('balance'):
            balance = tf.nn.sigmoid(accelerate * (tf.to_float(step) - warmup))
        loss = balance * loss_kld + loss_gen
    train_step = tf.train.AdamOptimizer().minimize(loss, step)

    return dict(
        tgt=tgt, dropout=dropout, balance=balance,
        mu=mu, lv=lv,
        logits=logits, prob=prob, pred=pred, acc=acc,
        step=step,
        loss=loss,
        loss_gen=loss_gen,
        loss_kld=loss_kld,
        train_step=train_step)
