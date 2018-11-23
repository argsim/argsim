from util_tf import tf, placeholder


def vae(tgt, dim_tgt, dim_emb, dim_rep, warmup=1e5, accelerate=1e-5, eos=1):
    # tgt : int32 (b, t)  | batchsize, timestep
    # dim_tgt : vocab size
    # dim_emb : model dimension
    # dim_rep : representation dimension

    tgt = placeholder(tf.int32, (None, None), tgt, 'tgt')
    batch_size = tgt.get_shape().as_list()[0]
    batch_size=2

    with tf.variable_scope('length'):
        length = tf.reduce_sum(tf.to_int32(tf.not_equal(tgt, eos)), -1)

    with tf.variable_scope('enc_embed'):
        # (b, t) -> (b, t, dim_emb)
        embed_mtrx = tf.get_variable(name="embed_mtrx", shape=[dim_tgt, dim_emb])
        embed = tf.gather(embed_mtrx, tgt)

    with tf.variable_scope('encode'):
        # (b, t, dim_emb) -> (b, dim_emb)
        cell = tf.contrib.rnn.GRUCell(dim_emb)
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        _, h = tf.nn.dynamic_rnn(cell, embed, sequence_length=length, initial_state=initial_state)
        # note more rnn layer here
        h = h[0]

    with tf.variable_scope('latent'):
        # (b, dim_emb) -> (b, dim_rep)
        # note maybe add more dense layers with activation for mu and lv
        mu = tf.layers.dense(h, dim_rep, name='mu')
        lv = tf.layers.dense(h, dim_rep, name='lv')
        with tf.name_scope('z'):
            z = mu + tf.exp(0.5 * lv) * tf.random_normal(shape=tf.shape(lv))
        # (b, dim_rep) -> (b, dim_emb)
        h = tf.layers.dense(z, dim_emb, name='proj')  # consider adding activation here

    with tf.variable_scope('dec_embed'):
        # (b, t) -> (b, t, dim_emb)
        # note we can just use `embed_mtrx` here, since it's the same language, same space
        dec_embed_mtrx = tf.get_variable(name="dec_embed_mtrx", shape=[dim_tgt, dim_emb])
        dec_embed = tf.gather(dec_embed_mtrx, tgt[:, :-1])

    with tf.variable_scope('decode'):
        cell_ = tf.nn.rnn_cell.GRUCell(dim_emb)
        h, _ = tf.nn.dynamic_rnn(cell_, dec_embed, initial_state=h, sequence_length=length)
        # note more rnn layer here

    with tf.variable_scope('mask'):
        # (b, t, dim_tgt) -> (b * ?, dim_tgt)
        mask = tf.sequence_mask(length)
        h = tf.boolean_mask(h, mask)

    with tf.variable_scope('logit'):
        # note more dense layers with activation here
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
