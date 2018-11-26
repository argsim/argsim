#!/usr/bin/env python3

import sys
if 1 == len(sys.argv): ckpt = None
if 2 == len(sys.argv): ckpt = sys.argv[1]
if 3 <= len(sys.argv): sys.exit("wrong args")
trial = "cudnn"

path_vocab = "../trial/data/vocab.model"
path_train = "../trial/data/train.txt"
path_valid = "../trial/data/valid.npy"
path_ckpt = "../trial/ckpt"
path_log = "../trial/log" # on colab
# path_log = "/cache/tensorboard-logdir/argsim" # on jarvis

seed = 0
batch_train = 256
batch_valid = 1024
# batch_train = 64
# batch_valid = 256

from model import vAe as vae
from tqdm import tqdm
from util_io import pform, load_txt
from util_np import np, vpack, sample, partition
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
        s = vocab.sample_encode_as_ids(raw[i], -1, 0.1)
        if 1 < len(s) <= 256:
            bat.append(s)

###############
# build model #
###############

model_valid = vae('valid')

# # for profiling
# from util_tf import profile
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     with tf.summary.FileWriter(pform(path_log, trial), sess.graph) as wtr:
#         profile(sess, wtr, model_valid['loss'], feed_dict= {model_valid['tgt']: valid[:32]})

tgt = pipe(batch, tf.int32, prefetch= 16)
model_train = vae('train', tgt=tgt)

############
# training #
############

sess = tf.InteractiveSession()
saver = tf.train.Saver()
if ckpt:
    saver.restore(sess, pform(path_ckpt, trial, ckpt))
else:
    tf.global_variables_initializer().run()

wtr = tf.summary.FileWriter(pform(path_log, trial))
summary = tf.summary.merge(
    (tf.summary.scalar('step_acc',      model_valid['acc']),
     tf.summary.scalar('step_loss',     model_valid['loss']),
     tf.summary.scalar('step_loss_gen', model_valid['loss_gen']),
     tf.summary.scalar('step_loss_kld', model_valid['loss_kld'])))

def summ(step, model=model_valid):
    fetches = model.acc, model.loss, model.loss_gen, model.loss_kld
    results = map(np.mean, zip(*(
        sess.run(fetches, {model.tgt: valid[i:j]})
        for i, j in partition(len(valid), batch_valid, discard= False))))
    wtr.add_summary(sess.run(summary, dict(zip(fetches, results))), step)

# with the current dataset and batch size, about 5k steps per epoch
for epoch in range(2): # train for 2 epochs at a time
    for _ in range(50):
        for _ in tqdm(range(100), ncols= 70):
            sess.run(model_train.train_step)
        step = sess.run(model_train.step)
        summ(step)
    saver.save(sess, pform(path_ckpt, trial, step // 5000), write_meta_graph= False)
