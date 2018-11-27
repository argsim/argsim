from util import identity, partial, Record
from util_tf import tf, placeholder


init_bias = tf.zeros_initializer()
init_embd = tf.variance_scaling_initializer(1.0, 'fan_out', 'uniform')
init_kern = tf.variance_scaling_initializer(1.0, 'fan_avg', 'uniform')
init_relu = tf.variance_scaling_initializer(2.0, 'fan_in' , 'uniform')
layer_aff = partial(tf.layers.dense, kernel_initializer=init_kern, bias_initializer=init_bias)
layer_act = partial(tf.layers.dense, kernel_initializer=init_relu, bias_initializer=init_bias)
scope = partial(tf.variable_scope, reuse=tf.AUTO_REUSE)


def vAe(mode,
        tgt=None,
        # model spec
        dim_tgt=8192,
        dim_emb=256,
        dim_rep=256,
        rnn_layers=2,
        attentive=True,
        logit_use_embed=False,
        # training spec
        drop_word=0.5,
        accelerate=1e-4,
        warmup=1e4,
        bos=2,
        eos=1):

    # mode
    # train: word dropout, optimizer
    # valid: without those
    # infer: autoregressive decoder
    #
    # dim_tgt : vocab size
    # dim_emb : model dimension
    # dim_rep : representation dimension

    assert mode in ('train', 'valid', 'infer')
    self = Record()

    with scope('tgt'):
        tgt = self.tgt = placeholder(tf.int32, (None, None), tgt, 'tgt')
        tgt = tf.transpose(tgt) # time major order
        not_eos = tf.not_equal(tgt, eos)
        len_raw = tf.reduce_sum(tf.to_int32(not_eos), axis=0)
        max_raw = tf.reduce_max(len_raw)
        # trims extra bos to make sure the lengths are right
        tgt, not_eos, len_seq = tgt[:max_raw], not_eos[:max_raw], len_raw + 1

    with scope('mask'):
        mask_dec = tf.pad(not_eos, ((1,0),(0,0)), constant_values=True)
        mask_enc = tf.pad(not_eos, ((0,1),(0,0)), constant_values=False)

    with scope('embed'): # (t, b) -> (t, b, dim_emb)
        embedding = tf.get_variable('embedding', (dim_tgt, dim_emb), initializer=init_embd)
        with scope('embed_enc'): # pads one eos for the attention query
            tgt_enc = tf.pad(tgt, paddings=((0,1),(0,0)), constant_values=eos)
            embed_enc = tf.gather(embedding, tgt_enc)
        with scope('embed_dec'): # pads one bos for the initial step
            if 'train' == mode:
                with scope('drop_word'):
                    tgt *= tf.to_int32(drop_word <= tf.random_uniform(tf.shape(tgt)))
            tgt_dec = tf.pad(tgt, paddings=((1,0),(0,0)), constant_values=bos)
            embed_dec = tf.gather(embedding, tgt_dec)

    # s : actual length
    # t : s + 1, with padding, either eos or bos
    # b : batch size
    # d : dimension aka dim_emb
    #
    # len_raw   :  b  aka s
    # len_seq   :  b  aka t
    #
    # not_eos   : sb
    # mask_enc  : tb  with eos
    # mask_dec  : tb  with bos
    #
    # tgt_enc   : tb  with eos
    # tgt_dec   : tb  with bos
    #
    # embed_enc : tbd with eos
    # embed_dec : tbd with bos

    with scope('encode'): # (t, b, dim_emb) -> (b, dim_emb)
        # bidirectional won't work correctly without length mask which cudnn doesn't take;
        # stick to unidirectional for now;
        # maybe combine this with a reverse run
        tbd, _ = tf.contrib.cudnn_rnn.CudnnGRU(
            num_layers=rnn_layers,
            num_units=dim_emb,
            kernel_initializer=init_kern,
            bias_initializer=init_bias
        )(embed_enc)
        with scope('cata'):
            # extract the final states which are the outputs from the first eos steps
            h = bd = tf.gather_nd(tbd, tf.stack((len_raw, tf.range(tf.size(len_raw), dtype=tf.int32)), axis=1))
            if attentive:
                # scaled dot-product attention;
                # the values are the outputs from all non-eos steps;
                # the queries are the final states
                v = tf.transpose(tbd, (1,0,2)) # values: btd <- tbd
                q = tf.expand_dims(bd, axis=1) # queries: b1d <- bd
                a = tf.matmul(q, v, transpose_b=True) # weights: b1t <- b1d @ (bdt <- btd)
                a *= dim_emb ** -0.5 # scale by sqrt dim_emb
                a += tf.log(tf.to_float(tf.expand_dims(tf.transpose(mask_enc), 1))) # mask eos steps
                a = tf.nn.softmax(a) # normalize
                h = tf.squeeze(a @ v, 1) # attend: bd <- b1d <- b1t @ btd
                # h += bd # residual connection

    with scope('latent'): # (b, dim_emb) -> (b, dim_rep) -> (b, dim_emb)
        h = layer_act(h, dim_emb, name='in')
        h_mu = layer_act(h, dim_emb, name='in_mu')
        h_lv = layer_act(h, dim_emb, name='in_lv')
        mu = self.mu = layer_aff(h_mu, dim_rep, name='mu')
        lv = self.lv = layer_aff(h_lv, dim_rep, name='lv')
        with scope('z'): h = self.z = mu + tf.exp(0.5 * lv) * tf.random_normal(shape=tf.shape(lv))
        h = layer_act(h, dim_emb, name='ex1')
        h = layer_act(h, dim_emb, name='ex2')

    with scope('decode'): # (b, dim_emb) -> (t, b, dim_emb) -> (?, dim_emb)
        h = tf.expand_dims(h, 0),
        h, _ = tf.contrib.cudnn_rnn.CudnnGRU(
            num_layers=1, # keep the decoder simple for now
            num_units=dim_emb,
            kernel_initializer=init_kern,
            bias_initializer=init_bias
        )(embed_dec, initial_state=h)
        h = tf.boolean_mask(h, mask_dec)
        h = layer_act(h, dim_emb, name='out1')
        h = layer_act(h, dim_emb, name='out2')

    with scope('logits'): # (?, dim_emb) -> (?, dim_tgt)
        if logit_use_embed:
            logits = self.logits = tf.matmul(h, embedding, transpose_b=True)
        else:
            logits = self.logits = layer_aff(h, dim_tgt)

    with scope('prob'):
        prob = self.prob = tf.nn.softmax(logits)

    with scope('pred'):
        pred = self.pred = tf.argmax(logits, -1, output_type=tf.int32)

    labels = tf.boolean_mask(tgt_enc, mask_dec, name='labels')

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
