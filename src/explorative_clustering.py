import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, v_measure_score
from sklearn.cluster import AffinityPropagation
from collections import Counter
import csv

def analyze(cluster, cluster_lbl, threshold, counts, max_acc=1):
    """
         cluster: sklearn clustering algorithm that has been fit on the cluster data
     cluster_lbl: the labels of the cluster_data
        treshold: 0.xx how many % of a cluster have to be of the chosen code for it to become relevant
          counts: how many instances must be in a cluster>counts
         max_acc: only clusters with an accuracy smaller than max_acc are selected

     return: pred_counts = number of instances per cluster
             pred_acc    = % of instances in the cluster being of the code
             best_pred   = cluster above treshold and count
    """
    labels = cluster.labels_
    n_clusters = len(set(labels))

    pred_match = np.zeros(n_clusters)
    for idx, label in enumerate(labels):
        if cluster_lbl[idx] == 1:
            pred_match[label] += 1

    # array that counts how many instances are in the cluster
    pred_counts = np.array([x[1] for x in list(Counter(labels).items())])

    pred_acc = np.array([(b/a) if b!=0 else 0 for a, b in zip(pred_counts, pred_match)])

    best_pred = [[idx, x, pred_counts[idx]] for idx, x in enumerate(pred_acc)
                 if x>threshold and pred_counts[idx]>counts and x<max_acc]

    return pred_counts, pred_acc, best_pred, labels


############
# GET DATA #
############

### Labels
# all labels
labels = [" ".join(sorted(l.split())) for l in  np.load("../data/test_data.npz")['labels']]
# only first labels
#labels = [l.split()[0] if l else "" for l in  np.load("../data/test_data.npz")['labels']]

topics = np.load("../data/test_data.npz")['topics']
posts = np.load("../data/test_data.npz")["posts"]

### Embeddings
# normal
#embed = np.load("../data/test_data_emb.npy")
# sentenc epiece sample
embed = np.load("../data/test_data_emb_sample.npy")


########################
# VISUALIZE EMBEDDINGS #
########################
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=20000)
tsne_results = tsne.fit_transform(embed)
xx = [x[0] for x in tsne_results]
yy = [x[1] for x in tsne_results]

tpc_dic = {"obama": 'r',
           "gayRights": "b",
           "abortion": "g",
           "marijuana": "k"}
colours = [tpc_dic[x] for x in topics]


for (i,cla) in enumerate(set(topics)):
    xc = [p for (j,p) in enumerate(xx) if topics[j]==cla]
    yc = [p for (j,p) in enumerate(yy) if topics[j]==cla]
    cols = [c for (j,c) in enumerate(colours) if topics[j]==cla]
    plt.scatter(xc,yc,c=cols,label=cla)
plt.legend(loc=4)
plt.show()


#######################
# AFFINITY PROPAGTION #
#######################
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
precomp_dist = squareform(pdist(embed, 'cosine'))

### TOPIC
af = AffinityPropagation(affinity="precomputed", max_iter=200, verbose=True).fit(precomp_dist)
cluster_labels = af.labels_


### save all info to csv
all_data=[]
for p,t,l,c in zip(posts, topics, labels, cluster_labels):
    all_data.append([p,t,l,c])
df = pd.DataFrame(all_data)
df.to_csv("../data/no_sample_cosine.csv")

### analyze for different topics
tpc = [1 if t =="obama" else 0 for t in topics]
#tpc = [1 if t =="abortion" else 0 for t in topics]
#tpc = [1 if t =="gayRights" else 0 for t in topics]
#tpc = [1 if t =="marijuana" else 0 for t in topics]

pred_counts, pred_acc, best_pred, cluster_lbl = analyze(af, tpc, 0.5, 1.1)

#############################################################################################
#########################
# TOPIC WISE CLUSTERING #
#########################
agglo_cluster = AgglomerativeClustering(n_clusters=len(set(topics)))
agglo_cluster.fit(embed)
pred_lbl = agglo_cluster.labels_

ars = adjusted_rand_score(topics, pred_lbl) #  [-1,1], 1 is perfect, 0 is random
v_msr = v_measure_score(topics, pred_lbl) # [0,1], 1 is perfect
print("TOPIC:  ARS: {:.4F}\tV_MSR: {:.4f}".format(ars, v_msr))

#########################
# ALL REASON CLUSTERING #
#########################
agglo_cluster = AgglomerativeClustering(n_clusters=len(set(labels)))
agglo_cluster.fit(embed)
pred_lbl = agglo_cluster.labels_

ars = adjusted_rand_score(labels, pred_lbl) #  [-1,1], 1 is perfect, 0 is random
v_msr = v_measure_score(labels, pred_lbl) # [0,1], 1 is perfect
print("REASON: ARS: {:.4F}\tV_MSR: {:.4f}".format(ars, v_msr))


###############################
# TOPICWISE REASON CLUSTERING #
###############################

for tpc  in ["obama", "abortion", "marijuana", "gayRights"]:
    lbl, inst = [], []
    for l, e, t in zip(labels, embed, topics):
        if t==tpc:
            lbl.append(l)
            inst.append(e)

            #agglo_cluster = AgglomerativeClustering(n_clusters=332)
            agglo_cluster = AgglomerativeClustering(n_clusters=len(set(lbl)))
    agglo_cluster.fit(inst)
    pred_lbl = agglo_cluster.labels_

    ars = adjusted_rand_score(lbl, pred_lbl) #  [-1,1], 1 is perfect, 0 is random
    v_msr = v_measure_score(lbl, pred_lbl) # [0,1], 1 is perfect
    print("{:<20s}: ARS: {:.4F}\tV_MSR: {:.4f}".format(tpc, ars, v_msr))
