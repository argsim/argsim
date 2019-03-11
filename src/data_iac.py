#!/usr/bin/env python3

path_raw = "../data/iac_v1.1/data/fourforums/discussions"
path_txt = "../data/iac.txt"
path_val = "../data/val.txt"

path_vocab = "../trial/data/vocab"
path_train = "../trial/data/train.txt"
path_valid = "../trial/data/valid.npy"
valid_size = 4096

from util_io import pform, load_json, clean, save_txt, load_txt
from util_np import np, vpack
from util_sp import load_spm, spm, encode, encode_capped
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
posts = [encode_capped(vocab, post, cap= 512) for post in posts]
save_txt(path_train, map(vocab.decode_ids, posts))

# validation data
posts = tuple(map(clean, load_txt(path_val)))
posts = [encode_capped(vocab, post, cap= 512) for post in posts]
np.save(path_valid, vpack(posts, (len(posts), 512), vocab.eos_id(), np.int32))
