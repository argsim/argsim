#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(description="""
trains a variational autoencoder on text.
logs validation statistics per 100 steps;
saves a ckeckpoint per 10000 steps aka one round;
the ckeckpoints are named after the trial name and the training round.
details are specified in the config file.
""", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--trial',    default="master",      help="the trial name")
parser.add_argument('--config',   default="config.json", help="the config file")
parser.add_argument('--ckpt',     default=None,          help="the ckeckpoint to resume")
parser.add_argument('--gpu',      default="0",           help="the gpu to use")
parser.add_argument('--rounds',   default=0,  type=int,  help="numbers of training rounds")
parser.add_argument('--prefetch', default=16, type=int,  help="numbers of batches to prefetch")
parser.add_argument('--sample',   action='store_true',   help="train with sentencepiece sampling")
parser.add_argument('--profile',  action='store_true',   help="run tensorboard profile")
A = parser.parse_args()

import sys
if not A.rounds and not A.profile: sys.exit("nothing to do")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = A.gpu

###############
# preparation #
###############

from util import Record
from model import vAe as vae
from tqdm import tqdm
from util_io import pform, load_txt, load_json
from util_np import np, vpack, sample, partition
from util_sp import load_spm, encode, decode
from util_tf import tf, pipe

config = load_json(A.config)
P = Record(config['paths'])
C = Record(config['model'])
T = Record(config['train'])

tf.set_random_seed(T.seed)

#############
# load data #
#############

vocab = load_spm(P.vocab)
valid = np.load(P.valid)

def batch(size=T.batch_train, path=P.train, vocab=vocab, seed=T.seed, kudo=A.sample, max_len=T.max_len):
    raw = tuple(load_txt(path))
    eos = vocab.eos_id()
    bat = []
    for i in sample(len(raw), seed):
        if size == len(bat):
            yield vpack(bat, (size, max(map(len, bat))), eos, np.int32)
            bat = []
        s = vocab.sample_encode_as_ids(raw[i], -1, 0.1) if kudo else \
            vocab.encode_as_ids(raw[i])
        if 0 < len(s) < max_len:
            bat.append(s)

###############
# build model #
###############

model_valid = vae('valid', **C)

if A.profile:
    from util_tf import profile
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        with tf.summary.FileWriter(pform(P.log, A.trial), sess.graph) as wtr:
            profile(sess, wtr, model_valid.loss, feed_dict= {model_valid.tgt: valid[:32]})
if not A.rounds: sys.exit("profiling done")

tgt = pipe(batch, tf.int32, prefetch= A.prefetch)
model_train = vae('train', tgt=tgt, **C)

############
# training #
############

sess = tf.InteractiveSession()
saver = tf.train.Saver()
if A.ckpt:
    saver.restore(sess, pform(P.ckpt, A.ckpt))
else:
    tf.global_variables_initializer().run()

wtr = tf.summary.FileWriter(pform(P.log, A.trial))
summary = tf.summary.merge(
    (tf.summary.scalar('step_acc',      model_valid['acc']),
     tf.summary.scalar('step_loss_gen', model_valid['loss_gen']),
     tf.summary.scalar('step_loss_kld', model_valid['loss_kld'])))

def summ(step, model=model_valid):
    fetches = model.acc, model.loss_gen, model.loss_kld
    results = map(np.mean, zip(*(
        sess.run(fetches, {model.tgt: valid[i:j]})
        for i, j in partition(len(valid), T.batch_valid, discard= False))))
    wtr.add_summary(sess.run(summary, dict(zip(fetches, results))), step)

for _ in range(A.rounds):
    for _ in range(100):
        for _ in tqdm(range(100), ncols= 70):
            sess.run(model_train.train_step)
        step = sess.run(model_train.step)
        summ(step)
    saver.save(sess, pform(P.ckpt, A.trial, step // 10000), write_meta_graph= False)
