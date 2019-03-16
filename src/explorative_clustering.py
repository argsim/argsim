import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, v_measure_score
from sklearn.cluster import AffinityPropagation
from collections import Counter
import csv


def analyze(pred, mask, min_acc= 0.5, min_cnt= 2):
    lbl2hit = Counter(pred[mask])
    lbl2cnt = Counter(pred)
    lbl_cnt_acc = [(lbl, cnt, lbl2hit[lbl] / cnt) for lbl, cnt in lbl2cnt.items()]
    lbl_cnt_acc.sort(key= lambda x: -x[-1])
    for lbl, cnt, acc in lbl_cnt_acc:
        if min_acc <= acc and min_cnt <= cnt:
            yield lbl, cnt, acc


############
# GET DATA #
############

test_data = np.load("../data/test_data.npz")

### Labels
# all labels
labels = np.array([" ".join(sorted(l.split())) for l in test_data['labels']])
# only first labels
# labels = np.array([l.split()[0] if l else "" for l in  test_data['labels']])

topics = test_data['topics']
posts = test_data["posts"]

### Embeddings
# normal
embed = np.load("../data/test_data_emb.npy")
# sentenc epiece sample
embed_sp = np.load("../data/test_data_emb_sample.npy")


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
from scipy.spatial import distance
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

### analyze for different topics
def analyze2(cluster_labels):
    for topic in "obama gayRights abortion marijuana".split():
        cls_cnt_acc = analyze(cluster_labels, topics == topic)
        print(topic.ljust(9), *["{:.3f}|{:03d}|{:03d}".format(a, n, c) for c, n, a in cls_cnt_acc])
    print(len(set(cluster_labels)), "clusters")

### euclidean
af = AffinityPropagation(affinity="euclidean", max_iter=200, verbose=True).fit(embed)
cluster_labels_euc = af.labels_
analyze2(cluster_labels_euc)

### euclidean with sampled embed
af = AffinityPropagation(affinity="euclidean", max_iter=200, verbose=True).fit(embed_sp)
cluster_labels_euc_sp = af.labels_
analyze2(cluster_labels_euc_sp)

### cosine
precomp_dist = squareform(- pdist(embed, 'cosine'))
af = AffinityPropagation(affinity="precomputed", max_iter=200, verbose=True).fit(precomp_dist)
cluster_labels_cos = af.labels_
analyze2(cluster_labels_cos)

### cosine with sampled embed
precomp_dist = squareform(- pdist(embed_sp, 'cosine'))
af = AffinityPropagation(affinity="precomputed", max_iter=200, verbose=True).fit(precomp_dist)
cluster_labels_cos_sp = af.labels_
analyze2(cluster_labels_cos_sp)

### distance to centroids
def dist_to_centroids(x, cluster_labels, dist_fn):
    row = np.arange(len(x)) # original row ids
    cls = [cluster_labels == c for c in set(cluster_labels)] # cluster masks
    row, cls = zip(*((row[c], x[c]) for c in cls)) # rows, clusters
    row = np.concatenate(row) # current row id -> original row id
    ctr = np.stack([np.mean(zs, axis= 0) for zs in cls]) # centroids
    dis = np.concatenate([np.array([dist_fn(z, c) for z in zs]) for zs, c in zip(cls, ctr)]) # distances
    return dis[np.argsort(row)] # restore original row ordering

dist_euc = dist_to_centroids(embed, cluster_labels_euc, distance.sqeuclidean)
dist_cos = dist_to_centroids(embed, cluster_labels_cos, distance.cosine)
dist_euc_sp = dist_to_centroids(embed_sp, cluster_labels_euc_sp, distance.sqeuclidean)
dist_cos_sp = dist_to_centroids(embed_sp, cluster_labels_cos_sp, distance.cosine)

### save all info to csv
df = pd.DataFrame({
    "post": posts, "topic": topics, "labels": labels
    , "cls_euc": cluster_labels_euc, "dist_euc": dist_euc
    , "cls_cos": cluster_labels_cos, "dist_cos": dist_cos
    , "cls_euc_sp": cluster_labels_euc_sp, "dist_euc_sp": dist_euc_sp
    , "cls_cos_sp": cluster_labels_cos_sp, "dist_cos_sp": dist_cos_sp})

df.to_csv("../trial/clustering.csv"
          , index_label= "id"
          , columns= "post topic labels \
          cls_euc dist_euc cls_euc_sp dist_euc_sp \
          cls_cos dist_cos cls_cos_sp dist_cos_sp".split())

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
