#!/usr/bin/env python3


path_csv = "../data/ibm_claim"
path_txt = "../data/ibm_claim.txt"

path_vocab = "../trial/data/vocab"
path_train = "../trial/data/train.txt"
path_valid = "../trial/data/valid.npy"
valid_size = 4096


from util_io import load_txt, save_txt
from util_np import np, vpack
from util_sp import load_spm, spm, encode
import csv

def load_ibm_claim(path):
    rows = csv.reader(load_txt(path))
    next(rows)
    for row in rows:
        yield row[3]

def load_all():
    for split in "q_mc_heldout.csv", "q_mc_test.csv", "q_mc_train.csv", "test_set.csv":
        yield from load_ibm_claim("{}/{}".format(path_csv, split))

# extract all sentences
save_txt(path_txt, load_all())

# train a sentence piece model
spm(name= path_vocab, path= path_txt)

# load the trained sentence piece model
vocab = load_spm(path_vocab + ".model")

# load and shuffle sentences
sents = list(load_txt(path_txt))
np.random.seed(0)
np.random.shuffle(sents)

# train valid split
valid = sents[:valid_size]
train = sents[valid_size:]

# save train and valid data
save_txt(path_train, train)
np.save(path_valid, encode(vocab, valid))
