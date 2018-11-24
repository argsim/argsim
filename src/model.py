from util_tf import tf, placeholder


def vAe(tgt, dim_tgt, dim_emb, dim_rep, rnn_layers=1, dropout=0.2, warmup=5e3, accelerate=5e-4, eos=1):
    # tgt : int32 (b, t)  | batchsize, timestep
    # dim_tgt : vocab size
    # dim_emb : model dimension
    # dim_rep : representation dimension

    tgt = placeholder(tf.int32, (None, None), tgt, 'tgt')

    with tf.variable_scope('dropout'):
        dropout = placeholder(tf.float32, (), dropout, 'dropout')
        keep_prob = 1.0 - dropout

    # disable dropout for now
    keep_prob = 1.0
    # hard code word dropout for now
    dropout_word = 0.5

    with tf.variable_scope('length'):
        length = tf.reduce_sum(tf.to_int32(tf.not_equal(tgt, eos)), -1)

    with tf.variable_scope('mask'):
        mask = tf.sequence_mask(length, maxlen=tf.shape(tgt)[1]-1)

    with tf.variable_scope('gold'):
        labels = tf.boolean_mask(tgt[:,1:], mask)

    with tf.variable_scope('embed'):
        # (b, t) -> (b, t, dim_emb)
        # same embedding for encoder and decoder
        embed_mtrx = tf.get_variable(name="embed_mtrx", shape=[dim_tgt, dim_emb])
        embed_enc = tf.gather(embed_mtrx, tgt[:,1:])
        embed_dec = tf.gather(embed_mtrx, tgt * tf.to_int32(dropout_word < tf.random_uniform(tf.shape(tgt))))

    with tf.variable_scope('encode'):
        # (b, t, dim_emb) -> (b, dim_emb)
        stacked_cells_fw = [
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(dim_emb),
                input_keep_prob=keep_prob,
                output_keep_prob=keep_prob,
                state_keep_prob=keep_prob,
                variational_recurrent=True,
                input_size=dim_emb,
                dtype=tf.float32)
            for _ in range(rnn_layers)]
        stacked_cells_bw = [
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(dim_emb),
                input_keep_prob=keep_prob,
                output_keep_prob=keep_prob,
                state_keep_prob=keep_prob,
                variational_recurrent=True,
                input_size=dim_emb,
                dtype=tf.float32)
            for _ in range(rnn_layers)]
        batch_size = tf.shape(embed_enc)[0]
        initial_state_fw = [cell.zero_state(batch_size, dtype=tf.float32) for cell in stacked_cells_fw]
        initial_state_bw = [cell.zero_state(batch_size, dtype=tf.float32) for cell in stacked_cells_bw]
        _, h_fw, h_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=stacked_cells_fw,
            cells_bw=stacked_cells_bw,
            inputs=embed_enc,
            initial_states_fw=initial_state_fw,
            initial_states_bw=initial_state_bw,
            sequence_length=length)
        h = tf.concat((h_fw[-1], h_bw[-1]), -1)

    with tf.variable_scope('latent'):
        # (b, dim_emb) -> (b, dim_rep)
        # h = tf.layers.dense(h, dim_emb, activation=tf.tanh, name='in')
        mu = tf.layers.dense(h, dim_rep, name='mu')
        lv = tf.layers.dense(h, dim_rep, name='lv')
        with tf.name_scope('z'):
            h = z = mu + tf.exp(0.5 * lv) * tf.random_normal(shape=tf.shape(lv))
        # h = tf.layers.dense(h, dim_emb, activation=tf.tanh, name='ex')

    with tf.variable_scope('decode'):
        # (b, dim_rep) -> (b, t, dim_emb)
        stacked_cells = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(dim_emb),
                input_keep_prob=keep_prob,
                output_keep_prob=keep_prob,
                state_keep_prob=keep_prob,
                variational_recurrent=True,
                input_size=dim_emb,
                dtype=tf.float32)
             for _ in range(rnn_layers)])
        initial_state = tuple(h for _ in range(rnn_layers))
        h, _ = tf.nn.dynamic_rnn(stacked_cells, embed_dec, initial_state=initial_state, sequence_length=length)

    with tf.variable_scope('logit'):
        # (b, t, dim_emb) -> (?, dim_tgt)
        h = tf.boolean_mask(h, mask)
        # h = tf.layers.dense(h, dim_emb, activation=tf.tanh)
        # h = tf.nn.dropout(h, keep_prob) # maybe remove this
        # logits = tf.layers.dense(h, dim_tgt)
        logits = tf.matmul(h, embed_mtrx, transpose_b=True) # embedding sharing

    with tf.variable_scope('prob'):
        prob = tf.nn.softmax(logits)

    with tf.variable_scope('pred'):
        pred = tf.argmax(logits, -1, output_type=tf.int32)

    with tf.variable_scope('acc'):
        acc = tf.reduce_mean(tf.to_float(tf.equal(labels, pred)))

    with tf.variable_scope('loss'):
        step = tf.train.get_or_create_global_step()
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
