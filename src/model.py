from util import partial, Record
from util_np import np
from util_tf import tf, placeholder, trim, get_shape


scope = partial(tf.variable_scope, reuse=tf.AUTO_REUSE)

init_bias = tf.zeros_initializer()
init_kern = tf.variance_scaling_initializer(1.0, 'fan_avg', 'uniform')
init_relu = tf.variance_scaling_initializer(2.0, 'fan_avg', 'uniform')

layer_nrm = tf.contrib.layers.layer_norm
layer_aff = partial(tf.layers.dense, kernel_initializer=init_kern, bias_initializer=init_bias)
layer_act = partial(tf.layers.dense, kernel_initializer=init_relu, bias_initializer=init_bias, activation=tf.nn.relu)
layer_rnn = partial(tf.contrib.cudnn_rnn.CudnnGRU, kernel_initializer=init_kern, bias_initializer=init_bias)


def attention(query, value, mask, dim, head=8):
    """computes scaled dot-product attention

    query : tensor f32 (b, d_q, t)
    value : tensor f32 (b, d_v, s)
     mask : tensor f32 (b,   t, s)
         -> tensor f32 (b, dim, t)

    `dim` must be divisible by `head`

    """
    assert not dim % head
    d,h,c = dim, head, dim // head
    b,_,t = get_shape(query)
    b,_,s = get_shape(value)
    # pretransformations
    v = tf.reshape(layer_aff(value, dim, name='v'), (b,h,c,s)) # bhcs <- bds <- bvs
    k = tf.reshape(layer_aff(value, dim, name='k'), (b,h,c,s)) # bhcs <- bds <- bvs
    q = tf.reshape(layer_aff(query, dim, name='q'), (b,h,c,s)) # bhct <- bdt <- bqt
    # weight
    a = tf.matmul(q, k, transpose_a= True) # bhts <- (bhtc <- bhct) @ bhcs
    a *= c ** -0.5
    if mask is not None: a += tf.expand_dims(mask, axis= 1)
    a = tf.nn.softmax(a, axis= -1)
    # attend
    y = tf.matmul(v, a, transpose_b= True) # bhct <- bhcs @ (bhst <- bhts)
    # posttransformation
    return layer_aff(tf.reshape(y, (b,d,t)), dim, name='p') # bdt <- bdt <- bhct


def vAe(mode,
        src=None,
        tgt=None,
        # model spec
        dim_tgt=8192,
        dim_emb=512,
        dim_rep=1024,
        rnn_layers=3,
        bidirectional=True,
        bidir_stacked=True,
        attentive=False,
        logit_use_embed=True,
        # training spec
        accelerate=1e-4,
        bos=2,
        eos=1):

    # dim_tgt : vocab size
    # dim_emb : model dimension
    # dim_rep : representation dimension
    #
    # unk=0 for word dropout

    assert mode in ('train', 'valid', 'infer')
    self = Record(bos=bos, eos=eos)

    with scope('step'):
        step = self.step = tf.train.get_or_create_global_step()
        rate = accelerate * tf.to_float(step)
        rate_keepwd = self.rate_keepwd = tf.sigmoid(rate)
        rate_anneal = self.rate_anneal = tf.tanh(rate)
        rate_update = self.rate_update = tf.rsqrt(rate + 1.0) * 1e-3

    with scope('src'):
        src = self.src = placeholder(tf.int32, (None, None), src, 'src')
        src = tf.transpose(src) # time major order
        src, msk_src, len_src = trim(src, eos)

    with scope('tgt'):
        tgt = self.tgt = placeholder(tf.int32, (None, None), tgt, 'tgt')
        tgt = tf.transpose(tgt) # time major order
        tgt, msk_tgt, len_tgt = trim(tgt, eos)
        msk_tgt = tf.pad(msk_tgt, ((1,0),(0,0)), constant_values=True)
        # pads for decoder : lead=[bos]+tgt -> gold=tgt+[eos]
        lead, gold = tgt, tf.pad(tgt, paddings=((0,1),(0,0)), constant_values=eos)
        if 'train' == mode: lead *= tf.to_int32(tf.random_uniform(tf.shape(lead)) < rate_keepwd)
        lead = self.lead = tf.pad(lead, paddings=((1,0),(0,0)), constant_values=bos)

    # s : src length
    # t : tgt length plus one padding, either eos or bos
    # b : batch size
    #
    # len_src :  b  aka s
    # msk_src : sb  without padding
    # msk_tgt : tb  with eos
    #
    #    lead : tb  with bos
    #    gold : tb  with eos

    with scope('embed'):
        b = (6 / (dim_tgt / dim_emb + 1)) ** 0.5
        embedding = tf.get_variable('embedding', (dim_tgt, dim_emb), initializer=tf.random_uniform_initializer(-b,b))
        emb_tgt = tf.gather(embedding, lead, name='emb_tgt') # (t, b) -> (t, b, dim_emb)
        emb_src = tf.gather(embedding,  src, name='emb_src') # (s, b) -> (s, b, dim_emb)

    with scope('encode'): # (s, b, dim_emb) -> (b, dim_emb)
        reverse = partial(tf.reverse_sequence, seq_lengths=len_src, seq_axis=0, batch_axis=1)

        if bidirectional and bidir_stacked:
            for i in range(rnn_layers):
                with scope("rnn{}".format(i+1)):
                    emb_fwd, _ = layer_rnn(1, dim_emb, name='fwd')(emb_src)
                    emb_bwd, _ = layer_rnn(1, dim_emb, name='bwd')(reverse(emb_src))
                    hs = emb_src = tf.concat((emb_fwd, reverse(emb_bwd)), axis=-1)

        elif bidirectional:
            with scope("rnn"):
                emb_fwd, _ = layer_rnn(rnn_layers, dim_emb, name='fwd')(emb_src)
                emb_bwd, _ = layer_rnn(rnn_layers, dim_emb, name='bwd')(reverse(emb_src))
            hs = tf.concat((emb_fwd, reverse(emb_bwd)), axis=-1)

        else:
            hs, _ = layer_rnn(rnn_layers, dim_emb, name='rnn')(emb_src)

        with scope('cata'):
            # extract the final states from the outputs: bd <- sbd, b2
            h = tf.gather_nd(hs, tf.stack((len_src-1, tf.range(tf.size(len_src), dtype=tf.int32)), axis=1))
            if attentive: # todo fixme
                # the values are the outputs from all non-padding steps;
                # the queries are the final states;
                h = layer_nrm(h + tf.squeeze( # bd <- bd1
                    attention( # bd1 <- bd1, bds, b1s
                        tf.expand_dims(h, axis=2), # query: bd1 <- bd
                        tf.transpose(hs, (1,2,0)), # value: bds <- sbd
                        tf.log(tf.to_float( # -inf,0  mask: b1s <- sb <- bs
                            tf.expand_dims(tf.transpose(msk_src), axis=1))),
                        int(h.shape[-1])), 2))

    with scope('latent'): # (b, dim_emb) -> (b, dim_rep) -> (b, dim_emb)
        # h = layer_aff(h, dim_emb, name='in')
        mu = self.mu = layer_aff(h, dim_rep, name='mu')
        lv = self.lv = layer_aff(h, dim_rep, name='lv')
        with scope('z'):
            h = mu
            if 'train' == mode:
                h += tf.exp(0.5 * lv) * tf.random_normal(shape=tf.shape(lv))
            self.z = h
        h = layer_aff(h, dim_emb, name='ex')

    with scope('decode'): # (b, dim_emb) -> (t, b, dim_emb) -> (?, dim_emb)
        h = self.state_in = tf.stack((h,)*rnn_layers)
        h, _ = _, (self.state_ex,) = layer_rnn(rnn_layers, dim_emb, name='rnn')(emb_tgt, initial_state=(h,))
        if 'infer' != mode: h = tf.boolean_mask(h, msk_tgt)
        h = layer_aff(h, dim_emb, name='out')

    with scope('logits'): # (?, dim_emb) -> (?, dim_tgt)
        if logit_use_embed:
            logits = self.logits = tf.tensordot(h, (dim_emb ** -0.5) * tf.transpose(embedding), 1)
        else:
            logits = self.logits = layer_aff(h, dim_tgt)

    with scope('prob'): prob = self.prob = tf.nn.softmax(logits)
    with scope('pred'): pred = self.pred = tf.argmax(logits, -1, output_type=tf.int32)

    if 'infer' != mode:
        labels = tf.boolean_mask(gold, msk_tgt, name='labels')
        with scope('errt'):
            errt_samp = self.errt_samp = tf.to_float(tf.not_equal(labels, pred))
            errt      = self.errt      = tf.reduce_mean(errt_samp)
        with scope('loss'):
            with scope('loss_gen'):
                loss_gen_samp = self.loss_gen_samp = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
                loss_gen      = self.loss_gen      = tf.reduce_mean(loss_gen_samp)
            with scope('loss_kld'):
                loss_kld_samp = self.loss_kld_samp = 0.5 * (tf.square(mu) + tf.exp(lv) - lv - 1.0)
                loss_kld      = self.loss_kld      = tf.reduce_mean(loss_kld_samp)
            loss = self.loss = rate_anneal * loss_kld + loss_gen

    if 'train' == mode:
        with scope('train'):
            train_step = self.train_step = tf.train.AdamOptimizer(rate_update).minimize(loss, step)

    return self


def encode(sess, vae, src):
    """returns latent states

    ->    array f32 (b, dim_rep)
    src : array i32 (b, t)

    """
    return sess.run(vae.z, {vae.src: src})


def decode(sess, vae, z, steps=256):
    """decodes latent states

    ->   array i32 (b, t)
    z  : array f32 (b, dim_rep)
    t <= steps

    """
    x = np.full((1, len(z)), vae.bos, dtype=np.int32)
    s = vae.state_in.eval({vae.z: z})
    y = []
    for _ in range(steps):
        x, s = sess.run((vae.pred, vae.state_ex), {vae.lead: x, vae.state_in: s})
        if np.all(x == vae.eos): break
        y.append(x)
    return np.concatenate(y).T
