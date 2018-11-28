from model import vAe, encode, decode, sample
from util_io import pform
from util_np import np
from util_tf import tf
import util_sp as sp
sess = tf.InteractiveSession()


path_vocab = "../trial/data/vocab.model"
path_ckpt = "../trial/ckpt"
trial = "biatt"
ckpt = 4


# build model for inference
vae = vAe('infer')
# restore model from ckeckpoint
tf.train.Saver().restore(sess, pform(path_ckpt, trial, ckpt))
# load sentencepiece model
vocab = sp.load_spm(path_vocab)


def generate(mu, lv, n=8):
    assert 1 == len(mu) == len(lv)
    # sample latent states
    z = sample(mu.ravel(), lv.ravel(), n)
    # decode sampled z
    y = decode(sess, vae, z)
    # decode as sentences
    for s in sp.decode(vocab, y):
        print(s)


# encode one sentence as ids
tgt = sp.encode(vocab, ["This is a test."])
# encode as mu and lv
mu, lv = encode(sess, vae, tgt)
# generate sentences
generate(mu, lv)


# generate central sentences
mu = lv = np.zeros((1, int(vae.mu.shape[1])))
generate(mu, lv)
