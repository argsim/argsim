from util_tf import tf, placeholder


def vae(src, tgt, len_src, len_tgt, dim_src, dim_tgt, dim_emb, dim_rep, warmup= 1e5, accelerate= 1e-5):

    # src, tgt : int32 (b, t)
    # len_src, len_tgt : int32 (b,)

    # dim_src : src vocab size
    # dim_tgt : tgt vocab size
    # dim_emb : model dimension
    # dim_rep : representation dimension

    src_ = placeholder(tf.int32, (None, None), src, 'src_')
    tgt_ = placeholder(tf.int32, (None, None), tgt, 'tgt_')

    gold = todo # tgt_ one step into future

    with tf.variable_scope('emb_src'):
        # create a variable for source embedding (dim_src, dim_emb)
        # look up src tf.gather : (dim_src, dim_emb), (b, s) -> (b, s, dim_emb)
        # s : src length
        todo

    with tf.variable_scope('emb_tgt'):
        # t : tgt length
        # (b, t) -> (b, t, dim_emb)
        todo

    with tf.variable_scope('encode'):
        # (b, s, dim_emb) -> (b, dim_emb)
        h = todo

    with tf.variable_scope('latent'):
        # (b, dim_emb) -> (b, dim_rep)
        mu = tf.layers.dense(h, dim_rep, name= 'mu')
        lv = tf.layers.dense(h, dim_rep, name= 'lv')
        with tf.name_scope('z'):
            z = mu + tf.exp(0.5 * lv) * tf.random_normal(shape= tf.shape(lv))
        # (b, dim_rep) -> (b, dim_emb)
        h = tf.layers.dense(z, dim_emb, name= 'proj')

    with tf.variable_scope('decode'):
        # (b, dim_emb) -> (b, t, dim_emb)
        # remember to ignore the final step
        h = todo

    with tf.variable_scope('logit'):
        # (b, t, dim_emb) -> (b, t, dim_tgt)
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
