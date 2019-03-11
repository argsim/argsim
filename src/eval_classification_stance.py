path_inputs = "../data/stance_emb_sample.npy"
# path_inputs = "../data/stance_emb.npy"
path_stance = "../data/stance.npz"


from collections import defaultdict, Counter
from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from util import partial
import numpy as np

#############
# load data #
#############

dataset = np.load(path_stance)
fold = dataset['fold']
top = dataset['top']
stn = dataset['stn']

inputs = np.load(path_inputs)

# group labels by topic then fold then stance
topic2fold2stance2idxs = defaultdict(partial(defaultdict, partial(defaultdict, list)))
for i, (topic, stance, f) in enumerate(zip(top, stn, fold)):
    topic2fold2stance2idxs[topic][f][stance].append(i)

# # print label counts for each topic and each fold
# for topic, fold2stance2idxs in topic2fold2stance2idxs.items():
#     print(topic)
#     for stance in {stance for stances in fold2stance2idxs.values() for stance in stances}:
#         print("| {} ".format(stance), end="")
#         for fold in range(5):
#             print("| {} ".format(len(topic2fold2stance2idxs[topic][fold][stance])), end="")
#         print("|")

# group instances by topic then fold
topic2fold2idxs = defaultdict(partial(defaultdict, list))
for topic, fold2stance2idxs in topic2fold2stance2idxs.items():
     for fold, stance2idxs in fold2stance2idxs.items():
          for idxs in stance2idxs.values():
               topic2fold2idxs[topic][fold].extend(idxs)
# dict str (list (array int))
topic2fold2idxs = {topic: tuple(np.array(idxs) for idxs in fold2idxs.values())
                   for topic, fold2idxs in topic2fold2idxs.items()}

##########################
# 5-fold crossvalidation #
##########################

f1_micro = partial(f1_score, average='micro')

def crossvalidation(fold2idxs, labels=stn, inputs=inputs, score=f1_micro, cost=0.001):
    scores = []
    for fold in range(5):
         i_valid = fold2idxs[fold]
         i_train = np.concatenate(fold2idxs[:fold] + fold2idxs[1+fold:])
         x_valid, y_valid = inputs[i_valid], labels[i_valid]
         x_train, y_train = inputs[i_train], labels[i_train]
         model = LogisticRegression(
             C=cost,
             penalty='l2',
             solver='liblinear',
             multi_class='auto',
             class_weight='balanced'
         ).fit(x_train, y_train)
         scores.append(score(y_valid, model.predict(x_valid)))
    return np.mean(scores)

# topic classification
fold2idxs = tuple(map(np.concatenate, zip(*topic2fold2idxs.values())))
print(crossvalidation(fold2idxs, labels= top, cost= 0.01))

# stance classification per topic
scores = []
for topic, fold2idxs in topic2fold2idxs.items():
     score = crossvalidation(fold2idxs, cost= 0.1)
     print(topic, "{:.2f}".format(score * 100))
     scores.append(score)
print(np.mean(scores))
