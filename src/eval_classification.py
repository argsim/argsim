path_inputs = "../trial/master17.npy"
path_labels = "../data/test_labels.npy"
path_folds  = "../data/test_folds.npy"


from collections import defaultdict, Counter
from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from util import partial
import numpy as np

#############
# load data #
#############

inputs = np.load(path_inputs)
labels = np.load(path_labels)
folds  = np.load(path_folds)

# group labels by topic then fold then reason
topic2fold2reason2idxs = defaultdict(partial(defaultdict, partial(defaultdict, list)))
for i, (label, fold) in enumerate(zip(labels, folds)):
    topic, stance, reason = label.split("-")
    # topic = "{}-{}".format(topic, stance) # per topic and stance
    reason = "{}-{}".format(stance, reason)
    topic2fold2reason2idxs[topic][fold][reason].append(i)

# # print label counts for each topic and each fold
# for topic, fold2reason2idxs in topic2fold2reason2idxs.items():
#     print(topic)
#     for reason in {reason for reasons in fold2reason2idxs.values() for reason in reasons}:
#         print("| {} ".format(reason), end="")
#         for fold in range(5):
#             print("| {} ".format(len(topic2fold2reason2idxs[topic][fold][reason])), end="")
#         print("|")

# group instances by topic then fold
topic2fold2idxs = defaultdict(partial(defaultdict, list))
for topic, fold2reason2idxs in topic2fold2reason2idxs.items():
     for fold, reason2idxs in fold2reason2idxs.items():
          for idxs in reason2idxs.values():
               topic2fold2idxs[topic][fold].extend(idxs)
# dict str (list (array int))
topic2fold2idxs = {topic: tuple(np.array(idxs) for idxs in fold2idxs.values())
                   for topic, fold2idxs in topic2fold2idxs.items()}

##########################
# 5-fold crossvalidation #
##########################

f1_micro = partial(f1_score, average='micro')

def crossvalidation(fold2idxs, inputs=inputs, labels=labels, score=f1_micro, cost=0.001):
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

scores = []
for topic, fold2idxs in topic2fold2idxs.items():
     score = crossvalidation(fold2idxs)
     print(topic, "{:.2f}".format(score * 100))
     scores.append(score)
print(np.mean(scores))
