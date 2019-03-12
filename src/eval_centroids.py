#!/usr/bin/env python3

path_ckpt = "../trial/ckpt/kudo396"
path_vocab = "../trial/data/vocab.model"

path_csv    = "../docs/results_iac/clustering.csv"
path_emb    = "../data/test_data_emb.npy"
path_emb_sp = "../data/test_data_emb_sample.npy"


from model import tf, vAe, encode, decode
from util import partial
from util_io import load_txt, save_txt
from util_np import np, partition, vpack
from util_np import vpack
import pandas as pd
import util_sp as sp

# load data
df = pd.read_csv(path_csv)
emb = np.load(path_emb)
emb_sp = np.load(path_emb_sp)

# load sentencepiece model
vocab = sp.load_spm(path_vocab)

# Load the model
model = vAe('infer')
# Restore the session
sess = tf.InteractiveSession()
tf.train.Saver().restore(sess, path_ckpt)

###########################
# generate from centroids #
###########################

for col in "euc euc_sp cos cos_sp".split():
   cluster = df["cls_{}".format(col)].values
   centroids = np.stack([np.mean(emb[cluster == c], axis= 0) for c in range(cluster.max() + 1)])
   y = decode(sess, model, centroids, steps= 512)
   save_txt("../trial/centroids_{}".format(col), sp.decode(vocab, y))
