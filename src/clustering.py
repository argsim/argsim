from util_tf import tf
from util_io import pform
import numpy as np
from model import vAe
import util_sp as sp
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, v_measure_score

path_lbls = "../argsim/data/test_labels.npy"
path_txt = "..argsim/data/test_data.txt"
path_vocab = "..argsim/trial/data/vocab"
path_ckpt = "..argsim/trial/ckpt/master5"


# load test data and labels
text = open(path_txt).read()
text = [line for line in text.split("\n")]
true_lbl = np.load(path_lbls)

# encode text with sentence piece model
vocab = sp.load_spm(path_vocab + ".model")
data = sp.encode(vocab, text)

# Load the model
model = vAe('infer')


# Restore the session
sess = tf.InteractiveSession()
tf.train.Save.restore(sess, path_ckpt)

# calculate mu for the test data
# ????????? doesnt work as one file or from a certain size one
inpt = []
inpt.extend(sess.run(model["mu"], {model["tgt"]: data[:1000]}))
inpt.extend(sess.run(model["mu"], {model["tgt"]: data[1000:2000]}))
inpt.extend(sess.run(model["mu"], {model["tgt"]: data[2000:3000]}))
inpt.extend(sess.run(model["mu"], {model["tgt"]: data[3000:]}))


###########################################
##############  CLUSTERING ################
###########################################

# train logistic classifier with l1 regularization
log_reg = LogisticRegression(penalty='l1', solver='liblinear', multi_class='auto')
log_reg.fit(inpt, true_lbl)

# pick dimensions that the classifier sees as useful
use_dim = [idx for idx,val in enumerate(log_reg.coef_[0]) if val != 0]
# reduce the dimensions of the clustering input
new_inpt = []
for instance in inpt:
     new_inpt.append([instance[dim] for dim in use_dim])

agglo_cluster = AgglomerativeClustering(n_clusters=len(set(true_lbl)))
agglo_cluster.fit(new_inpt)
pred_lbl = agglo_cluster.labels_

# Evaluate
ars = adjusted_rand_score(true_lbl, pred_lbl) #  [-1,1], 1 is perfect, 0 is random
v_msr = v_measure_score(true_lbl, pred_lbl) # [0,1], 1 is perfect
print("ARS: ", ars, "V_MSR: ", v_msr)
