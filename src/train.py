#!/usr/bin/env python3


seed = 0
path_vocab = "../trial/data/vocab.model"
path_train = "../trial/data/train.txt"
path_valid = "../trial/data/valid.npy"

batch_train = 64
batch_valid = 512


from util_io import load_txt
from util_np import np, vpack, sample
from util_sp import load_spm, encode, decode
from util_tf import tf, pipe
tf.set_random_seed(seed)


#############
# load data #
#############


vocab = load_spm(path_vocab)
valid = np.load(path_valid)


def batch(size= batch_train, path= path_train, vocab= vocab, seed= seed, eos= 1):
    raw = tuple(load_txt(path))
    bat = []
    for i in sample(len(raw), seed):
        if size == len(bat):
            yield vpack(bat, (size, max(map(len, bat))), eos, np.int32)
            bat = []
        bat.append(vocab.sample_encode_as_ids(raw[i], -1, 0.1))


## try this
# bat = pipe(batch, tf.int32, prefetch= 4)
# sess = tf.InteractiveSession()
# bat.eval()


###############
# build model #
###############


############
# training #
############
