from util_tf import tf, placeholder


def vae(tgt, dim_tgt, dim_emb, dim_rep, warmup= 1e5, accelerate= 1e-5, eos= 1):
    # tgt : int32 (b, t)
    # dim_tgt : vocab size
    # dim_emb : model dimension
    # dim_rep : representation dimension

    tgt = placeholder(tf.int32, (None, None), tgt, 'tgt')

    with tf.variable_scope('length'):
        length = tf.reduce_sum(tf.to_int32(tf.not_equal(tgt, eos)), -1)

    with tf.variable_scope('embed'):
        # (b, t) -> (b, t, dim_emb)
        # create a variable for embedding (dim_tgt, dim_emb)
        # look up using tf.gather
        todo

    with tf.variable_scope('encode'):
        # (b, t, dim_emb) -> (b, dim_emb)
        h = todo

    with tf.variable_scope('latent'):
        # (b, dim_emb) -> (b, dim_rep)
        mu = tf.layers.dense(h, dim_rep, name= 'mu')
        lv = tf.layers.dense(h, dim_rep, name= 'lv')
        with tf.name_scope('z'):
            z = mu + tf.exp(0.5 * lv) * tf.random_normal(shape= tf.shape(lv))
        # (b, dim_rep) -> (b, dim_emb)
        h = tf.layers.dense(z, dim_emb, name= 'proj') # consider adding activation here

    with tf.variable_scope('decode'):
        # (b, dim_emb) -> (b, t, dim_emb)
        h = todo

    with tf.variable_scope('logit'):
        # (b, t, dim_emb) -> (b, t, dim_tgt)
        y = todo

    with tf.variable_scope('mask'):
        # (b, t, dim_tgt) -> (b * ?, dim_tgt)
        # use tf.sequence_mask and tf.boolean_mask to extract positions within length
        logits = todo # (b * ?, dim_tgt)
        labels = todo # (b * ?,), this should come from tgt[:,1:]

    with tf.variable_scope('prob'): prob = tf.nn.softmax(y)
    with tf.variable_scope('pred'): pred = tf.argmax(y, -1, output_type= tf.int32)
    with tf.variable_scope('acc'):
        acc = tf.reduce_mean(tf.to_float(tf.equal(labels, tf.argmax(logits, -1, output_type= tf.int32))))

    step = tf.train.get_or_create_global_step()
    with tf.variable_scope('loss'):
        with tf.name_scope('loss_gen'):
            loss_gen = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels= labels, logits= logits))
        with tf.name_scope('loss_kld'):
            loss_kld = 0.5 * tf.reduce_mean(
                tf.reduce_sum(tf.square(mu) + tf.exp(lv) - lv - 1.0, axis= 1))
        with tf.name_scope('balance'):
            balance = tf.nn.sigmoid(accelerate * (tf.to_float(step) - warmup))
        loss = balance * loss_kld + loss_gen
    up = tf.train.AdamOptimizer().minimize(loss, step)
