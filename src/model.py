from util import identity, partial
from util_tf import tf, placeholder


# init_kern = tf.variance_scaling_initializer(1.0, 'fan_avg', 'uniform')
init_kern = tf.variance_scaling_initializer(2.0, 'fan_avg', 'uniform')
init_bias = tf.zeros_initializer()

layer_aff = partial(tf.layers.dense, kernel_initializer=init_kern, bias_initializer=init_bias)
layer_act = partial(layer_aff, activation=tf.nn.relu)
rnn_cell = partial(tf.nn.rnn_cell.GRUCell, activation=tf.nn.relu, kernel_initializer=init_kern, bias_initializer=init_bias)


def vAe(tgt,
        # model spec
        dim_tgt=8192,
        dim_emb=256,
        dim_rep=256,
        rnn_layers=2,
        logit_use_embed=False,
        # training spec
        keep_word=0.5,
        keep_prob=0.9,
        warmup=1e4,
        accelerate=1e-4,
        eos=1):
    # tgt : int32 (b, t)  | batchsize, timestep
    # dim_tgt : vocab size
    # dim_emb : model dimension
    # dim_rep : representation dimension

    tgt = placeholder(tf.int32, (None, None), tgt, 'tgt')
    keep_word = placeholder(tf.float32, (), keep_word, 'keep_word')
    keep_prob = placeholder(tf.float32, (), keep_prob, 'keep_prob')

    dropout = partial(tf.nn.dropout, keep_prob=keep_prob)
    rnn_dropout = lambda cell, input_size=dim_emb: tf.nn.rnn_cell.DropoutWrapper(
        cell=cell,
        input_keep_prob=keep_prob,
        output_keep_prob=keep_prob,
        state_keep_prob=keep_prob,
        variational_recurrent=True,
        input_size=input_size,
        dtype=tf.float32)

    # disable dropout for now
    dropout = identity
    rnn_dropout = lambda cell, input_size=dim_emb: cell

    with tf.variable_scope('input'):
        with tf.variable_scope('length'):
            length = tf.reduce_sum(tf.to_int32(tf.not_equal(tgt, eos)), -1)
            maxlen = tf.reduce_max(length)
        with tf.variable_scope('mask'):
            mask = tf.sequence_mask(length, maxlen=maxlen)
        with tf.variable_scope('gold'):
            labels = tf.boolean_mask(tgt[:,1:maxlen+1], mask)
        with tf.variable_scope('tgt_dec'):
            tgt_dec = tgt[:,:maxlen]
        with tf.variable_scope('tgt_enc'):
            tgt_enc = tgt[:,1:maxlen]

    with tf.variable_scope('embed'):
        # (b, t) -> (b, t, dim_emb)
        embed_mtrx = tf.get_variable(name="embed_mtrx", shape=[dim_tgt, dim_emb], initializer=init_kern)
        embed_enc = tf.gather(embed_mtrx, tgt_enc)
        embed_dec = tf.gather(embed_mtrx, tgt_dec * tf.to_int32(tf.random_uniform(tf.shape(tgt_dec)) < keep_word))

    with tf.variable_scope('encode'):
        # (b, t, dim_emb) -> (b, dim_emb)
        _, h_fw, h_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=[rnn_dropout(rnn_cell(dim_emb), dim_emb*(1+i)) for i in range(rnn_layers)],
            cells_bw=[rnn_dropout(rnn_cell(dim_emb), dim_emb*(1+i)) for i in range(rnn_layers)],
            inputs=embed_enc,
            sequence_length=length,
            dtype=tf.float32)
        h = tf.concat((h_fw[-1], h_bw[-1]), -1)

    with tf.variable_scope('latent'):
        # (b, dim_emb) -> (b, dim_rep) -> (b, dim_emb)
        h = layer_act(h, dim_emb, name='in1')
        h = layer_act(h, dim_emb, name='in2')
        mu = layer_aff(h, dim_rep, name='mu')
        lv = layer_aff(h, dim_rep, name='lv')
        with tf.name_scope('z'):
            h = z = mu + tf.exp(0.5 * lv) * tf.random_normal(shape=tf.shape(lv))
        h = layer_act(h, dim_emb, name='ex1')
        h = layer_act(h, dim_emb, name='ex2')

    with tf.variable_scope('decode'):
        # (b, dim_emb) -> (b, t, dim_emb) -> (?, dim_emb)
        h, _ = tf.nn.dynamic_rnn(rnn_dropout(rnn_cell(dim_emb)), embed_dec, initial_state=h, sequence_length=length)
        # # keep decoder simple for now
        # stacked_cells = tf.nn.rnn_cell.MultiRNNCell([rnn_dropout(rnn_cell(dim_emb)) for _ in range(rnn_layers)])
        # initial_state = tuple(h for _ in range(rnn_layers))
        # h, _ = tf.nn.dynamic_rnn(stacked_cells, embed_dec, initial_state=initial_state, sequence_length=length)
        h = tf.boolean_mask(h, mask)
        h = dropout(layer_act(h, dim_emb, name='out1'))
        h = dropout(layer_act(h, dim_emb, name='out2'))

    # (?, dim_emb) -> (?, dim_tgt)
    if logit_use_embed:
        logits = tf.matmul(h, embed_mtrx, transpose_b=True, name='logit')
    else:
        logits = layer_aff(h, dim_tgt, name='logit')

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
            loss_kld = 0.5 * tf.reduce_mean(tf.square(mu) + tf.exp(lv) - lv - 1.0)
        with tf.name_scope('balance'):
            balance = tf.nn.relu(tf.tanh(accelerate * (tf.to_float(step) - warmup)))
        loss = balance * loss_kld + loss_gen
    train_step = tf.train.AdamOptimizer().minimize(loss, step)

    return dict(
        tgt=tgt, keep_word=keep_word, keep_prob=keep_prob,
        mu=mu, lv=lv,
        logits=logits, prob=prob, pred=pred, acc=acc,
        step=step,
        loss=loss,
        loss_gen=loss_gen,
        loss_kld=loss_kld,
        balance=balance,
        train_step=train_step)
