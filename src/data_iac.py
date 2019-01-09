#!/usr/bin/env python3

path_raw = "../data/iac_v1.1/data/fourforums/discussions"
path_txt = "../data/iac.txt"

path_vocab = "../trial/data/vocab"
path_train = "../trial/data/train.txt"
path_valid = "../trial/data/valid.npy"
valid_size = 4096

from util_io import pform, load_json, clean, save_txt
from util_np import np, vpack
from util_sp import load_spm, spm, encode
import json
import os

posts = tuple(
    clean(post[3])
    # extract the cleaned raw texts
    for filename in sorted(os.listdir(path_raw))
    # each json: posts, annotations, metadata
    for post in load_json(pform(path_raw, filename))[0]
    # each post: id, side(unused), author, raw text, annotations, parent post id, category (unused), timestamp
)

# removes empty posts
posts = tuple(post for post in posts if 0 < len(post))

# saves raw texts
save_txt(path_txt, posts)

# train a sentence piece model
spm(name= path_vocab, path= path_txt)

# load the trained sentence piece model
vocab = load_spm(path_vocab + ".model")


# length control

encoded = tuple(vocab.encode_as_ids(post) for post in posts)

x = np.array(list(map(len, encoded)))
# length coverage
#  256   0.7170
#  512   0.8865
# 1024   0.9644
# 2048   0.9924

# todo check the length of posts in reason to determine threshold


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
