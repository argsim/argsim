from util_tf import tf, placeholder


def vae(tgt, dim_tgt, dim_emb, dim_rep, warmup= 1e5, accelerate= 1e-5, eos= 1):
    # tgt : int32 (b, t)
    # dim_tgt : vocab size
    # dim_emb : model dimension
    # dim_rep : representation dimension
    dim_tgt = 50
    dim_emb = 40
    dim_rep = 60
    eos= 1
    tgt=bat.eval()
    batch_size=64

    #batch_size = len(tgt)
    tgt = placeholder(tf.int32, (None, None), tgt, 'tgt')
    #batch_size = tgt.get_shape().as_list()[0]

    with tf.variable_scope('length'):
        length = tf.reduce_sum(tf.to_int32(tf.not_equal(tgt, eos)), -1)

    with tf.variable_scope('enc_embed'):
        # (b, t) -> (b, t, dim_emb)
        # create a variable for embedding (dim_tgt, dim_emb)
        # embed using tf.gather
        embed_mtrx = tf.get_variable(name= "embed_mtrx", shape = [dim_tgt, dim_emb])
        embed = tf.gather(embed_mtrx,tgt)

    with tf.variable_scope('encode'):
        # (b, t, dim_emb) -> (b, dim_emb)
        cell = tf.contrib.rnn.LSTMCell(dim_emb)
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        _, h = tf.nn.dynamic_rnn(cell, embed, sequence_length=length, initial_state=initial_state)
        #h = tf.placeholder(tf.float32,[None,dim_emb])

    with tf.variable_scope('latent'):
        # (b, dim_emb) -> (b, dim_rep)
        mu = tf.layers.dense(h, dim_rep, name= 'mu')
        lv = tf.layers.dense(h, dim_rep, name= 'lv')
        with tf.name_scope('z'):
            z = mu + tf.exp(0.5 * lv) * tf.random_normal(shape= tf.shape(lv))
        # (b, dim_rep) -> (b, dim_emb)
        h = tf.layers.dense(z, dim_emb, name= 'proj') # consider adding activation here

    with tf.variable_scope('dec_embed'):
        # (b, t) -> (b, t, dim_emb)
        # create a variable for embedding (dim_tgt, dim_emb)
        # embed using tf.gather
        dec_embed_mtrx = tf.get_variable(name= "dec_embed_mtrx", shape = [dim_tgt, dim_emb])
        dec_embed = tf.gather(dec_embed_mtrx, tgt)

    with tf.variable_scope('decode'):
        # (b, dim_emb) -> (b, t, dim_emb)
        cell = tf.nn.rnn_cell.LSTMCell(dim_emb)
        helper = tf.contrib.seq2seq.TrainingHelper(dec_embed, length)
        projection_layer = tf.layers.Dense(dim_tgt,use_bias=False)
        decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, initial_state=h, output_layer=projection_layer)
        outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        logits = outputs.rnn_output
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
