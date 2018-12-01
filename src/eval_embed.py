#!/usr/bin/env python3

path_ckpt = "../trial/ckpt/master7"
path_emb = "../trial/emb7.npy"

path_vocab = "../trial/data/vocab.model"
path_txt = "../data/test_data.txt"

from model import tf, vAe
from util_io import load_txt
from util_np import np, partition
import util_sp as sp

# load test sentences
text = list(load_txt(path_txt))

# encode text with sentence piece model
vocab = sp.load_spm(path_vocab)
data = sp.encode(vocab, text)

# Load the model
model = vAe('infer')

# Restore the session
sess = tf.InteractiveSession()
tf.train.Saver().restore(sess, path_ckpt)

# calculate z for the test data
inpt = [model.z.eval({model.tgt: data[i:j]}) for i, j in partition(len(data), 128)]
inpt = np.concatenate(inpt, axis=0)

# save embedded sentences
np.save(path_emb, inpt)
