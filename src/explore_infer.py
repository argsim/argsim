from model import vAe, encode, decode
from util_io import pform
from util_np import np
from util_tf import tf
import util_sp as sp


path_vocab = "../trial/data/vocab.model"
path_ckpt = "../trial/ckpt"


vocab = sp.load_spm(path_vocab)
s0 = "This is a test."
s1 = "Dragons have been portrayed in film and television in many different forms."
s2 = "His development of infinitesimal calculus opened up new applications of the methods of mathematics to science."
tgt = sp.encode(vocab, (s0, s1, s2))


vae = vAe('infer')
sess = tf.InteractiveSession()


def auto(z, steps=256):
    for s in sp.decode(vocab, decode(sess, vae, z, steps)):
        print(s)


for i in range(1, 7):
    print()
    ckpt = "master{}".format(i)
    tf.train.Saver().restore(sess, pform(path_ckpt, ckpt))
    auto(encode(sess, vae, tgt))
    auto(np.zeros((1, int(vae.mu.shape[1]))))
