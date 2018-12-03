#!/usr/bin/env python3

path_ckpt = "../trial/ckpt/kudo18"
path_emb = "../trial/kudo18.npy"

path_vocab = "../trial/data/vocab.model"
path_txt = "../data/test_data.txt"

from model import tf, vAe
from util_io import load_txt
from util_np import np, partition, vpack
import util_sp as sp

# load test sentences
text = list(load_txt(path_txt))
# load sentencepiece model
vocab = sp.load_spm(path_vocab)

# Load the model
model = vAe('infer')
# Restore the session
sess = tf.InteractiveSession()
tf.train.Saver().restore(sess, path_ckpt)

################################
# deterministic representation #
################################

# encode text with sentence piece model
data = sp.encode(vocab, text)
# calculate z for the test data in batches
inpt = [model.z.eval({model.tgt: data[i:j]}) for i, j in partition(len(data), 128)]
inpt = np.concatenate(inpt, axis=0)

#######################################################
# averaged representation with sentencepiece sampling #
#######################################################

def infer_avg(sent, samples=128):
    bat = [vocab.sample_encode_as_ids(sent, -1, 0.1) for _ in range(samples)]
    bat = vpack(bat, (len(bat), max(map(len, bat))), vocab.eos_id(), np.int32)
    z = model.z.eval({model.tgt: bat})
    return np.mean(z, axis=0)

inpt = np.stack(list(map(infer, text)), axis=0)

###########################
# save embedded sentences #
###########################

np.save(path_emb, inpt)
