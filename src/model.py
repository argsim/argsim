from util import identity, partial, Record
from util_tf import tf, placeholder


# init_kern = tf.variance_scaling_initializer(1.0, 'fan_avg', 'uniform')
init_kern = tf.variance_scaling_initializer(2.0, 'fan_avg', 'uniform')
init_bias = tf.zeros_initializer()
layer_aff = partial(tf.layers.dense, kernel_initializer=init_kern, bias_initializer=init_bias)
layer_act = partial(layer_aff, activation=tf.nn.relu)
scope = partial(tf.variable_scope, reuse=tf.AUTO_REUSE)


def vAe(mode,
        tgt=None,
        # model spec
        dim_tgt=8192,
        dim_emb=256,
        dim_rep=256,
        rnn_layers=2,
        logit_use_embed=False,
        # training spec
        dropout_rate=0.2,
        dropout_word=0.5,
        accelerate=1e-4,
        warmup=1e4,
        eos=1):
    # mode
    # train: dropout, word dropout, optimizer
    # valid: without those
    # infer: autoregressive decoder

    # dim_tgt : vocab size
    # dim_emb : model dimension
    # dim_rep : representation dimension

    assert mode in ('train', 'valid', 'infer')
    self = Record()

    # int32 (b, t)  | batchsize, timestep
    tgt = self.tgt = placeholder(tf.int32, (None, None), tgt, 'tgt')

    if 'train' == mode:
        dropout = partial(tf.nn.dropout, keep_prob=1.0-dropout_rate)
    else:
        dropout = identity

    dropout_rate = 0.0
    dropout = identity

    with scope('input'):
        tgt = tf.transpose(tgt) # time major order
        with scope('length'):
            length = tf.reduce_sum(tf.to_int32(tf.not_equal(tgt, eos)), 0)
            maxlen = tf.reduce_max(length)
        with scope('mask'):
            mask = tf.transpose(tf.sequence_mask(length, maxlen=maxlen))
        with scope('gold'):
            labels = tf.boolean_mask(tgt[1:maxlen+1], mask)
        with scope('tgt_dec'):
            tgt_dec = tgt[0:maxlen]
        with scope('tgt_enc'):
            tgt_enc = tgt[1:maxlen]

    with scope('embed'):
        # (t, b) -> (t, b, dim_emb)
        embedding = tf.get_variable(
            initializer=tf.variance_scaling_initializer(1.0, 'fan_out', 'uniform'),
            shape=(dim_tgt, dim_emb),
            name="embedding")
        if 'train' == mode:
            with scope('word_dropout'):
                # pad to never drop bos
                tgt_dec *= tf.pad(
                    tf.to_int32(dropout_word <= tf.random_uniform(tf.shape(tgt_enc))),
                    paddings=((1,0),(0,0)),
                    constant_values=1)
        with scope('embed_dec'): embed_dec = dropout(tf.gather(embedding, tgt_dec))
        with scope('embed_enc'): embed_enc = dropout(tf.gather(embedding, tgt_enc))

    with scope('encode'):
        # (t, b, dim_emb) -> (b, dim_emb)
        # bidirectional won't work correctly without length mask which cudnn doesn't take;
        # stick to unidirectional for now;
        # maybe combine this with a reverse run, or add attention
        h, _ = tf.contrib.cudnn_rnn.CudnnGRU(
            num_layers=rnn_layers,
            num_units=dim_emb,
            direction='unidirectional',
            dropout=dropout_rate if 'train' == mode else 0.0
        )(embed_enc)
        # extract the states from the outputs at the steps corresponding to the lengths;
        # since the encoder does not take bos, and the indices are one less of the lengths,
        # in total it must be offset by 2
        with scope('cata'):
            h = tf.gather_nd(h, tf.stack((length-2, tf.range(tf.size(length), dtype=tf.int32)), axis=-1))

    with scope('latent'):
        # (b, dim_emb) -> (b, dim_rep) -> (b, dim_emb)
        # no dropout here since we do not want distributed representation
        with scope('in1'): h = layer_act(h, dim_emb)
        with scope('in2'): h = layer_act(h, dim_emb)
        with scope('mu'): mu = self.mu = layer_aff(h, dim_rep)
        with scope('lv'): lv = self.lv = layer_aff(h, dim_rep)
        with scope('z'): h = self.z = mu + tf.exp(0.5 * lv) * tf.random_normal(shape=tf.shape(lv))
        with scope('ex1'): h = layer_act(h, dim_emb)
        with scope('ex2'): h = layer_act(h, dim_emb)

    with scope('decode'):
        # (b, dim_emb) -> (t, b, dim_emb) -> (?, dim_emb)
        h, _ = tf.contrib.cudnn_rnn.CudnnGRU(
            num_layers=1, # keep decoder simple for now
            num_units=dim_emb,
            direction='unidirectional',
            dropout=dropout_rate if 'train' == mode else 0.0
        )(embed_dec, initial_state=(tf.expand_dims(h, 0),))
        h = tf.boolean_mask(h, mask)
        with scope('out1'): h = dropout(layer_act(h, dim_emb))
        with scope('out2'): h = dropout(layer_act(h, dim_emb))

    with scope('logit'):
        # (?, dim_emb) -> (?, dim_tgt)
        if logit_use_embed:
            logits = self.logits = tf.matmul(h, embed_mtrx, transpose_b=True)
        else:
            logits = self.logits = layer_aff(h, dim_tgt)

    with scope('prob'):
        prob = self.prob = tf.nn.softmax(logits)

    with scope('pred'):
        pred = self.pred = tf.argmax(logits, -1, output_type=tf.int32)

    with scope('acc'):
        acc = self.acc = tf.reduce_mean(tf.to_float(tf.equal(labels, pred)))

    step = self.step = tf.train.get_or_create_global_step()
    with scope('loss'):
        with scope('loss_gen'):
            loss_gen = self.loss_gen = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        with scope('loss_kld'):
            loss_kld = self.loss_kld = 0.5 * tf.reduce_mean(tf.square(mu) + tf.exp(lv) - lv - 1.0)
        with scope('balance'):
            balance = self.balance = tf.nn.relu(tf.tanh(accelerate * (tf.to_float(step) - warmup)))
        loss = self.loss = balance * loss_kld + loss_gen

    if 'train' == mode:
        with scope('train'):
            train_step = self.train_step = tf.train.AdamOptimizer().minimize(loss, step)

    return self

# - autoregressive loop for infer mode
# - functions for running model
