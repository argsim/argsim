import tensorflow as tf
from model import vAe, decode
import util_sp as sp
from util_io import load_txt
import numpy as np


def analyze(z, use_dim=[], seed=25):
    ''' z =  np.array[2, dim], mu of two sentences'''
    ''' use_dim = list of int describing which dimension should be used '''

    # select random path from z1 to z2
    np.random.seed(seed)
    if use_dim == []:
        rdm_path = np.arange(len(z[0]))
    else:
        rdm_path = use_dim
    np.random.shuffle(rdm_path)

    # walk the path and print  at every step
    path = np.copy(z[0])
    for idx,dim in enumerate(rdm_path):
        path[dim] = z[1][dim]
        output = decode(sess, vae, [z[0], path, z[1]]).tolist()
        _ = [vocab.decode_ids(output[idx]) for idx in range(3)]
        print(idx,dim, _[1])
        #print("{}\n{}\n{}\n{}\n".format(idx,_[0],_[1],_[2])) #print: sentence1, path, sentence2


path_vocab = "../trial/data/vocab.model"
path_txt = "../data/test_data.txt"
path_ckpt = "../trial/ckpt/kudo18"
path_use_dim = "../data/useful_dimension.npy"

# load and restore model
vae = vAe('infer')
sess = tf.InteractiveSession()
tf.train.Saver().restore(sess, path_ckpt)

# load vocab and text
vocab = sp.load_spm(path_vocab)
text = list(load_txt(path_txt))

#pick 2 random sentences to explore
np.random.seed(23)
sen_idx = np.random.random_integers(0, len(text), 2)
sentences = [text[idx] for idx in sen_idx]
print("sentence 1: {}\nsentence 2: {}".format(sentences[0], sentences[1]))

# encode sentences with sentence piece model
data = sp.encode(vocab, sentences)

### full high dimensional space
z = vae.z.eval({vae.tgt: data})
analyze(z)

### only the dimensions that turned out usefull for our task
use_dim = np.load(path_use_dim)
analyze(z, use_dim)
