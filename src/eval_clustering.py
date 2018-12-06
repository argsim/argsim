path_lbls = "../data/test_labels.npy"
path_inpt = "../trial/kudo18.npy"

############
# analysis #
############

from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

inpt = np.load(path_inpt)

# load labels
true_lbl = np.load(path_lbls)

topic, stance, reason = zip(*[lbl.split("-") for lbl in true_lbl])
labels = tuple(zip(topic, stance, reason))
stance = tuple(zip(topic, stance))
reason = tuple(zip(topic, reason))

top2idx = {lbl: idx for idx, lbl in enumerate(sorted(set(topic)))}
top_ids = np.array([top2idx[top] for top in topic])

stn2idx = {stn: idx for idx, stn in enumerate(sorted(set(stance)))}
stn_ids = np.array([stn2idx[stn] for stn in stance])

rsn2idx = {rsn: idx for idx, rsn in enumerate(sorted(set(reason)))}
rsn_ids = np.array([rsn2idx[rsn] for rsn in reason])

lbl2idx = {lbl: idx for idx, lbl in enumerate(sorted(set(labels)))}
lbl_ids = np.array([lbl2idx[lbl] for lbl in labels])

########################################
# pick the partition for analysis
gold, gold2idx = top_ids, top2idx
########################################


###################
# classifiication #
###################

from sklearn.linear_model import LogisticRegression

# train logistic classifier with l1 regularization
log_reg = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', multi_class='auto')
log_reg.fit(inpt, gold)

pred = log_reg.predict(inpt)
metrics.f1_score(gold, pred, average='weighted')

conf = metrics.confusion_matrix(gold, pred)
plt.imshow(conf, cmap='gray')
plt.show()

# pick dimensions that the classifier sees as useful
use_dim = np.array([idx for idx,val in enumerate(log_reg.coef_.T) if not np.allclose(val, 0.0)])
np.save("../data/useful_dimension.npy",use_dim)
# reduce the dimensions of the clustering input
new_inpt = np.array([instance[use_dim] for instance in inpt])


############################
# embedding visualizatioin #
############################

#from sklearn.manifold import TSNE
#
#x = new_inpt
#x = new_inpt / np.linalg.norm(new_inpt, axis=-1, keepdims=True)
#
#e = TSNE(n_components=2).fit_transform(x)
#
#plt.scatter(e[:,0], e[:,1], c=gold)
#plt.show()


##############
# clustering #
##############

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, v_measure_score

# clustering by selection (topic, stance, reason)
for sel in gold2idx:
     if type(sel) == str:
          points = np.array([i for i,(t,_,_) in enumerate(labels) if t == sel])
          print("clustering by topic: ")
     elif len(sel[1]) == 1:
          points = np.array([i for i,(t,s,_) in enumerate(labels) if (t,s) == sel])
          print("clustering by stance: ")
     else:
          points = np.array([i for i,(t,_,r) in enumerate(labels) if (t,r) == sel])
          print("clustering by reason: ")

     x, y = new_inpt[points], lbl_ids[points]

     agglo_cluster = AgglomerativeClustering(n_clusters=len(set(y)))
     agglo_cluster.fit(x)
     pred_lbl = agglo_cluster.labels_

     ars = adjusted_rand_score(y, pred_lbl) #  [-1,1], 1 is perfect, 0 is random
     v_msr = v_measure_score(y, pred_lbl) # [0,1], 1 is perfect
     print("{:<20s}\tARS: {:.4F}\tV_MSR: {:.4f}".format(str(sel), ars, v_msr))
