#!/usr/bin/env python3

path_ckpt = "../trial/ckpt/kudo396"
path_emb = "../data/stance_emb.npy"

path_vocab = "../trial/data/vocab.model"
path_data = "../data/stance.npz"

from model import tf, vAe
from util import partial
from util_io import load_txt
from util_np import np, partition, vpack
import util_sp as sp
from util_np import vpack

# load test sentences
text = np.load(path_data)['text']
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
data = list(map(partial(sp.encode_capped, vocab), text))
data = vpack(data, (len(data), max(map(len, data))), vocab.eos_id(), np.int32)

# calculate z for the test data in batches
inpt = [model.z.eval({model.src: data[i:j]}) for i, j in partition(len(data), 128)]
inpt = np.concatenate(inpt, axis=0)

#######################################################
# averaged representation with sentencepiece sampling #
#######################################################

# def infer_avg(sent, samples=128):
#    bat = [sp.encode_capped_sample(vocab, sent) for _ in range(samples)]
#    bat = vpack(bat, (len(bat), max(map(len, bat))), vocab.eos_id(), np.int32)
#    z = model.z.eval({model.src: bat})
#    return np.mean(z, axis=0)

# inpt = np.stack(list(map(infer_avg, text)), axis=0)

###########################
# save embedded sentences #
###########################

np.save(path_emb, inpt)
