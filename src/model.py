from util import identity, partial, Record
from util_np import np
from util_tf import tf, placeholder


init_bias = tf.zeros_initializer()
init_embd = tf.random_uniform_initializer(-0.5, 0.5)
init_kern = tf.variance_scaling_initializer(1.0, 'fan_avg', 'uniform')
init_relu = tf.variance_scaling_initializer(2.0, 'fan_avg', 'uniform')

layer_aff = partial(tf.layers.dense, kernel_initializer=init_kern, bias_initializer=init_bias)
layer_act = partial(tf.layers.dense, kernel_initializer=init_relu, bias_initializer=init_bias, activation=tf.nn.relu)
layer_rnn = partial(tf.contrib.cudnn_rnn.CudnnGRU, kernel_initializer=init_kern, bias_initializer=init_bias)

scope = partial(tf.variable_scope, reuse=tf.AUTO_REUSE)


def attention(query, value, mask, dim, head=1, transform=False):
    """computes scaled dot-product attention

    query : tensor f32 (b, s, d_q)
    value : tensor f32 (b, t, d_v)
     mask : tensor f32 (b, s, t)
         -> tensor f32 (b, s, dim)

    transform may be omitted when `dim == d_q == d_v`

    `dim` must be divisible by `head`

    """
    assert not dim % head
    k, v, q = value, value, query
    if transform:
        k = layer_aff(k, dim, name='k') # btd key
        v = layer_aff(v, dim, name='v') # btd value
        q = layer_aff(q, dim, name='q') # bsd query
    if 1 < head: k, v, q = map(lambda x: tf.stack(tf.split(x, head, -1)), (k, v, q))
    a = tf.nn.softmax(
        tf.matmul(q, k, transpose_b=True) # weight: bst <- bsd @ (bdt <- btd)
        * ((dim // head) ** -0.5) # scale by sqrt d
        + mask # mask
    ) @ v # attend: bsd <- bst @ btd
    if 1 < head: a = tf.concat(tf.unstack(a), -1)
    return a


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
        bos=2,
        eos=1):

    # dim_tgt : vocab size
    # dim_emb : model dimension
    # dim_rep : representation dimension
    #
    # unk=0

    assert mode in ('train', 'valid', 'infer')
    self = Record(bos=bos, eos=eos)

    with scope('tgt'):
        tgt = self.tgt = placeholder(tf.int32, (None, None), tgt, 'tgt')
        tgt = tf.transpose(tgt) # time major order
        not_eos = tf.not_equal(tgt, eos)
        len_seq = tf.reduce_sum(tf.to_int32(not_eos), axis=0)
        max_len = tf.reduce_max(len_seq)
        # trims extra bos to make sure the lengths are right
        tgt, not_eos = tgt[:max_len], not_eos[:max_len]
        # tgt reversed and having eos replaced with bos
        gtg = tf.reverse_sequence(tgt, len_seq, seq_axis=0, batch_axis=1)
        gtg = tf.where(not_eos, gtg, tf.fill(tf.shape(gtg), bos))

    with scope('mask'):
        mask_dec = tf.pad(not_eos, ((1,0),(0,0)), constant_values=True)
        mask_enc = tf.pad(not_eos, ((0,1),(0,0)), constant_values=False)

    with scope('embed'): # (t, b) -> (t, b, dim_emb)
        embedding = tf.get_variable('embedding', (dim_tgt, dim_emb), initializer=init_embd)
        with scope('embed_enc'): # pads one eos or bos for the attention query
            tgt_enc = tf.pad(tgt, paddings=((0,1),(0,0)), constant_values=eos)
            gtg_enc = tf.pad(gtg, paddings=((0,1),(0,0)), constant_values=bos)
            embed_enc = tf.gather(embedding, tgt_enc)
            embed_gtg = tf.gather(embedding, gtg_enc)
        with scope('embed_dec'): # pads one bos for the initial step
            if 'train' == mode:
                with scope('drop_word'):
                    tgt *= tf.to_int32(drop_word <= tf.random_uniform(tf.shape(tgt)))
            tgt_dec = tf.pad(tgt, paddings=((1,0),(0,0)), constant_values=bos)
            embed_dec = tf.gather(embedding, tgt_dec)

    # t : seq length plus one padding, either eos or bos
    # b : batch size
    # d : dimension aka dim_emb
    #
    # len_seq   :  b  aka t-1
    # not_eos   : sb  where s = t-1
    #
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
        tgt, _ = layer_rnn(rnn_layers, dim_emb, name='rnn_fwd')(embed_enc) # tbd
        gtg, _ = layer_rnn(rnn_layers, dim_emb, name='rnn_bwd')(embed_gtg) # tbd
        with scope('cata'):
            # extract the final states which are the outputs from the first padding steps
            idx = tf.stack((len_seq, tf.range(tf.size(len_seq), dtype=tf.int32)), axis=1) # b2
            # forward is backward
            fwd = tf.gather_nd(gtg, idx) # bd <- tbd, b2
            bwd = tf.gather_nd(tgt, idx) # bd <- tbd, b2
            if attentive:
                # the values are the outputs from all non-padding steps;
                # the queries are the final states;
                # padding mask: b1t <- tb <- bt
                msk = tf.log(tf.to_float(tf.expand_dims(tf.transpose(mask_enc), axis=1)))
                # query: b1d <- bd
                fwd = tf.expand_dims(fwd, axis=1)
                bwd = tf.expand_dims(bwd, axis=1)
                # value: btd <- tbd
                tgt = tf.transpose(tgt, (1,0,2))
                gtg = tf.transpose(gtg, (1,0,2))
                # attend: b1d <- b1d, btd, b1t
                with scope('fwd'): fwd = attention(fwd, tgt, msk, dim_emb)
                with scope('bwd'): bwd = attention(bwd, gtg, msk, dim_emb)
                # bd <- b1d
                fwd = tf.squeeze(fwd, axis=1)
                bwd = tf.squeeze(bwd, axis=1)

    with scope('latent'): # (b, dim_emb), (b, dim_emb) -> (b, dim_rep) -> (b, dim_emb)
        fwd = layer_aff(fwd, dim_emb, name='in_fwd')
        bwd = layer_aff(bwd, dim_emb, name='in_bwd')
        h = fwd + bwd
        mu = self.mu = layer_aff(h, dim_rep, name='mu')
        lv = self.lv = layer_aff(h, dim_rep, name='lv')
        with scope('z'): h = self.z = mu + tf.exp(0.5 * lv) * tf.random_normal(shape=tf.shape(lv))
        h = layer_aff(h, dim_emb, name='ex')

    with scope('decode'): # (b, dim_emb) -> (t, b, dim_emb) -> (?, dim_emb)
        self.tgt_dec = tgt_dec
        h = self.state_in = tf.expand_dims(h, axis=0)
        h, _ = _, (self.state_ex,) = layer_rnn(1, dim_emb, name='rnn')(embed_dec, initial_state=(h,))
        # keep the decoder simple for now, just 1 rnn layer
        if 'infer' != mode: h = tf.boolean_mask(h, mask_dec)
        h = layer_aff(h, dim_emb, name='out')

    with scope('logits'): # (?, dim_emb) -> (?, dim_tgt)
        if logit_use_embed:
            logits = self.logits = tf.tensordot(h, (dim_emb ** -0.5) * tf.transpose(embedding), 1)
        else:
            logits = self.logits = layer_aff(h, dim_tgt)

    with scope('prob'): prob = self.prob = tf.nn.softmax(logits)
    with scope('pred'): pred = self.pred = tf.argmax(logits, -1, output_type=tf.int32)

    if 'infer' != mode:
        labels = tf.boolean_mask(tgt_enc, mask_dec, name='labels')
        with scope('acc'): acc = self.acc = tf.reduce_mean(tf.to_float(tf.equal(labels, pred)))
        step = self.step = tf.train.get_or_create_global_step()
        with scope('loss'):
            with scope('loss_gen'):
                loss_gen = self.loss_gen = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
            with scope('loss_kld'):
                anneal = self.anneal = tf.tanh(accelerate * tf.to_float(step))
                mu *= anneal
                lv *= anneal
                loss_kld = self.loss_kld = 0.5 * tf.reduce_mean(tf.square(mu) + tf.exp(lv) - lv - 1.0)
            loss = self.loss = loss_kld + loss_gen

    if 'train' == mode:
        with scope('train'):
            train_step = self.train_step = tf.train.AdamOptimizer().minimize(loss, step)

    return self


def encode(sess, vae, tgt):
    """returns latent state parameters `mu` and `lv` given `tgt`

    tgt : array i32 (b, t)
     mu : array f32 (b, dim_rep)
     lv : array f32 (b, dim_rep)

    """
    return sess.run((vae.mu, vae.lv), {vae.tgt: tgt})


def decode(sess, vae, z, steps= 256):
    """-> array i32 (b, t)

    z :  array f32 (b, dim_rep)
    t <= steps

    decodes latent states

    """
    x = np.full((1, len(z)), vae.bos, dtype=np.int32)
    s = vae.state_in.eval({vae.z: z})
    y = []
    for _ in range(steps):
        x, s = sess.run((vae.pred, vae.state_ex), {vae.tgt_dec: x, vae.state_in: s})
        if np.all(x == vae.eos): break
        y.append(x)
    return np.concatenate(y).T


def sample(mu, lv, size):
    """-> array f32 (size, dim_rep)

    mu : array f32 dim_rep
    lv : array f32 dim_rep

    samples latent states given mean `mu` and log variance `lv`

    """
    return mu + np.exp(0.5 * lv) * np.random.randn(size, len(lv))
