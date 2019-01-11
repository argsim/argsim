from util import partial, Record
from util_np import np
from util_tf import tf, placeholder


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

    query : tensor f32 (b, t, d_q)
    value : tensor f32 (b, s, d_v)
     mask : tensor f32 (b, t, s)
         -> tensor f32 (b, t, dim)

    `dim` must be divisible by `head`

    """
    assert not dim % head
    k = layer_aff(value, dim, name='k') # bsd
    v = layer_aff(value, dim, name='v') # bsd
    q = layer_aff(query, dim, name='q') # btd
    if 1 < head: k, v, q = map(lambda x: tf.stack(tf.split(x, head, -1)), (k, v, q))
    a = tf.nn.softmax( # weight
        tf.matmul(q, k, transpose_b=True) # bts <- btd @ (bds <- bsd)
        * ((dim // head) ** -0.5) # scale by sqrt d
        + mask # 0 for true, -inf for false
    ) @ v # attend: btd <- bts @ bsd
    if 1 < head: a = tf.concat(tf.unstack(a), -1)
    return layer_aff(a, dim, name='p')


def vAe(mode,
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
        rate_update = self.rate_update = tf.reciprocal(1.0 + rate) * 1e-3

    with scope('tgt'):
        tgt = self.tgt = placeholder(tf.int32, (None, None), tgt, 'tgt')
        tgt = tf.transpose(tgt) # time major order
        not_eos = tf.not_equal(tgt, eos)
        len_seq = tf.reduce_sum(tf.to_int32(not_eos), axis=0)
        max_len = tf.reduce_max(len_seq)
        # trims to make sure the lengths are right
        tgt = tgt[:max_len]
        msk_enc = not_eos[:max_len]
        msk_dec = tf.pad(msk_enc, ((1,0),(0,0)), constant_values=True)
        # pads for decoder : lead=[bos]+tgt -> gold=tgt+[eos]
        lead, gold = tgt, tf.pad(tgt, paddings=((0,1),(0,0)), constant_values=eos)
        if 'train' == mode: lead *= tf.to_int32(tf.random_uniform(tf.shape(lead)) < rate_keepwd)
        lead = self.lead = tf.pad(lead, paddings=((1,0),(0,0)), constant_values=bos)

    # s : seq length
    # t : seq length plus one padding, either eos or bos
    # b : batch size
    #
    # len_seq :  b  aka s aka t-1
    # msk_enc : sb  without padding
    # msk_dec : tb  with eos
    #
    #    lead : tb  with bos
    #    gold : tb  with eos

    with scope('embed'):
        b = (6 / (dim_tgt / dim_emb + 1)) ** 0.5
        embedding = tf.get_variable('embedding', (dim_tgt, dim_emb), initializer=tf.random_uniform_initializer(-b,b))
        emb_dec = tf.gather(embedding, lead, name='emb_dec') # (t, b) -> (t, b, dim_emb)
        emb_enc = tf.gather(embedding,  tgt, name='emb_enc') # (s, b) -> (s, b, dim_emb)

    with scope('encode'): # (s, b, dim_emb) -> (b, dim_emb)
        reverse = partial(tf.reverse_sequence, seq_lengths=len_seq, seq_axis=0, batch_axis=1)

        if bidirectional and bidir_stacked:
            for i in range(rnn_layers):
                with scope("rnn{}".format(i+1)):
                    emb_tgt, _ = layer_rnn(1, dim_emb, name='fwd')(emb_enc)
                    emb_gtg, _ = layer_rnn(1, dim_emb, name='bwd')(reverse(emb_enc))
                    hs = emb_enc = tf.concat((emb_tgt, reverse(emb_gtg)), axis=-1)

        elif bidirectional:
            with scope("rnn"):
                emb_tgt, _ = layer_rnn(rnn_layers, dim_emb, name='fwd')(emb_enc)
                emb_gtg, _ = layer_rnn(rnn_layers, dim_emb, name='bwd')(reverse(emb_enc))
            hs = tf.concat((emb_tgt, reverse(emb_gtg)), axis=-1)

        else:
            hs, _ = layer_rnn(rnn_layers, dim_emb, name='rnn')(emb_enc)

        with scope('cata'):
            # extract the final states from the outputs: bd <- sbd, b2
            h = tf.gather_nd(hs, tf.stack((len_seq-1, tf.range(tf.size(len_seq), dtype=tf.int32)), axis=1))
            if attentive:
                # the values are the outputs from all non-padding steps;
                # the queries are the final states;
                h = layer_nrm(h + tf.squeeze( # bd <- b1d
                    attention( # b1d <- b1d, bsd, b1s
                        tf.expand_dims(h, axis=1), # query: b1d <- bd
                        tf.transpose(hs, (1,0,2)), # value: bsd <- sbd
                        tf.log(tf.to_float( # -inf,0  mask: b1s <- sb <- bs
                            tf.expand_dims(tf.transpose(msk_enc), axis=1))),
                        int(h.shape[-1])), 1))

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
        h, _ = _, (self.state_ex,) = layer_rnn(rnn_layers, dim_emb, name='rnn')(emb_dec, initial_state=(h,))
        if 'infer' != mode: h = tf.boolean_mask(h, msk_dec)
        h = layer_aff(h, dim_emb, name='out')

    with scope('logits'): # (?, dim_emb) -> (?, dim_tgt)
        if logit_use_embed:
            logits = self.logits = tf.tensordot(h, (dim_emb ** -0.5) * tf.transpose(embedding), 1)
        else:
            logits = self.logits = layer_aff(h, dim_tgt)

    with scope('prob'): prob = self.prob = tf.nn.softmax(logits)
    with scope('pred'): pred = self.pred = tf.argmax(logits, -1, output_type=tf.int32)

    if 'infer' != mode:
        labels = tf.boolean_mask(gold, msk_dec, name='labels')
        with scope('errt'): errt = self.errt = tf.reduce_mean(tf.to_float(tf.equal(labels, pred)))

        with scope('loss'):
            with scope('loss_gen'):
                loss_gen = self.loss_gen = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
            with scope('loss_kld'):
                loss_kld = tf.reduce_mean(tf.square(mu) + tf.exp(lv) - lv - 1.0)
                if 'train' == mode:
                    loss_kld *= 0.5 * rate_anneal
                else:
                    loss_kld *= 0.5
                self.loss_kld = loss_kld
            loss = self.loss = loss_kld + loss_gen

    if 'train' == mode:
        with scope('train'):
            train_step = self.train_step = tf.train.AdamOptimizer(rate_update).minimize(loss, step)

    return self


def encode(sess, vae, tgt):
    """returns latent states

    ->    array f32 (b, dim_rep)
    tgt : array i32 (b, t)

    """
    return sess.run(vae.z, {vae.tgt: tgt})


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
