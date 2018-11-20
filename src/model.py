from util_tf import tf, placeholder


def vae(dim_src, dim_tgt, dim_emb, dim_rep, warmup= 1e5, accelerate= 1e-5):

    # dim_src : src vocab size
    # dim_tgt : tgt vocab size
    # dim_emb : model dimension
    # dim_rep : representation dimension

    src_ = placeholder(tf.int32, (None, None), src, 'src_')
    tgt_ = placeholder(tf.int32, (None, None), tgt, 'tgt_')

    gold = todo

    with tf.variable_scope('emb_src'):
        todo

    with tf.variable_scope('emb_tgt'):
        todo

    with tf.variable_scope('encode'):
        h = todo

    with tf.variable_scope('latent'):
        mu = tf.layers.dense(h, dim_rep, name= 'mu')
        lv = tf.layers.dense(h, dim_rep, name= 'lv')
        with tf.name_scope('z'):
            h = z = mu + tf.exp(0.5 * lv) * tf.random_normal(shape= tf.shape(lv))

    with tf.variable_scope('decode'):
        h = todo

    with tf.variable_scope('logit'):
        y = todo

    with tf.variable_scope('prob'):
        prob = tf.nn.softmax(y)

    with tf.variable_scope('pred'):
        pred = tf.argmax(y, -1, output_type= tf.int32)

    with tf.variable_scope('acc'):
        acc = tf.reduce_mean(tf.to_float(tf.equal(gold, pred)))

    step = tf.train.get_or_create_global_step()
    with tf.variable_scope('loss'):
        with tf.name_scope('loss_gen'):
            loss_gen = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels= gold, logits= y))
        with tf.name_scope('loss_kld'):
            loss_kld = 0.5 * tf.reduce_mean(
                tf.reduce_sum(tf.square(mu) + tf.exp(lv) - lv - 1.0, axis= 1))
        with tf.name_scope('balance'):
            balance = tf.nn.sigmoid(accelerate * (tf.to_float(step) - warmup))
        loss = balance * loss_kld + loss_gen

    up = tf.train.AdamOptimizer().minimize(loss, step)
