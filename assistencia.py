# -*- coding: utf-8 -*-
import csv
import unicodedata
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from simple_ga import Simple_GA


with open('estudantesAssistencia.csv', 'r') as csvfile:
    csv_ = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)   
    data = [row for row in csv_]


def standard_chars(text):
    std_text = unicodedata.normalize('NFD', text) # remove letter accents
    return u''.join(ch for ch in std_text if unicodedata.category(ch) != 'Mn')

from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler

def cluster_accuracy(true_labels, clusters):
    labels = np.array([uid for uid, ulab in enumerate(np.unique(true_labels)) for l in true_labels if ulab == l])
    cluster_classes = np.zeros(len(labels))
    for c in np.unique(clusters):
        cltr_id = [sid for sid, cid in enumerate(clusters) if cid == c]
        cltr_labels = labels[cltr_id]
        cltr_class = np.bincount(cltr_labels).argmax()
        for sid in cltr_id:
            cluster_classes[sid] = cltr_class
    return accuracy_score(labels, cluster_classes)

def cluster_information(true_labels, clusters, distance_mtx):
    labels = np.array([uid for uid, ulab in enumerate(np.unique(true_labels)) for l in true_labels if ulab == l])
    cluster_classes = np.zeros(len(labels))
    """for c in np.unique(clusters):
      
        cltr_id = [sid for sid, cid in enumerate(clusters) if cid == c]
        cltr_labels = labels[cltr_id]
        
        cltr_class = np.bincount(cltr_labels).argmax()
        for sid in cltr_id:
            cluster_classes[sid] = cltr_class
    """
    for c in np.unique(clusters):
        dataset_indices = [sid for sid, cid in enumerate(clusters) if cid == c]
        print(str(len(dataset_indices))+" in cluster "+str(c))
        if len(dataset_indices) > 1:
            
            cluster_distances = distance_mtx[np.ix_(dataset_indices, dataset_indices)]

            # otimizar esse processo por numpy (diag)
            cluster_distances = np.array([[cluster_distances[i, j] if i != j else -1 for i, _ in enumerate(dataset_indices)] for j, _ in enumerate(dataset_indices)])
            
            items = check_distance(cluster_distances, items="max")
            max_val = cluster_distances[items[0], items[1]]
            maxid_a, maxid_b = (dataset_indices[items[0]], dataset_indices[items[1]])
            items = check_distance(cluster_distances, items="min")
            min_val = cluster_distances[items[0], items[1]]
            minid_a, minid_b = (dataset_indices[items[0]], dataset_indices[items[1]])

            print("Cluster : "+str(c))
            
            for l_ in np.unique(true_labels):
                print("Class : "+str(l_)+" - "+str(len([True for sid in dataset_indices if true_labels[sid] == l_]))+" items")
            
            print("Mean Dist. : "+str(np.mean(distance_mtx[np.ix_(dataset_indices, dataset_indices)])))
            print("Max Dist. : "+str(maxid_a)+"/"+str(maxid_b)+" : "+str(max_val))
            print("Min Dist. : "+str(minid_a)+"/"+str(minid_b)+" : "+str(min_val))

            if labels[maxid_a] == labels[maxid_b]:
                print(str(c)+" : closed cluster")
                for i in dataset_indices:
                    cluster_classes[i] = labels[maxid_b]
            else:
                print(str(c)+" : open cluster")
                for idx, pos in enumerate(dataset_indices):
                    
                    a = dataset_indices.index(maxid_a)
                    b = dataset_indices.index(maxid_b)
                    if cluster_distances[idx][a] >= cluster_distances[idx][b]:
                        cluster_classes[pos] = labels[maxid_a]
                    else:
                        cluster_classes[pos] = labels[maxid_b]
        else:
            cluster_classes[i] = labels[dataset_indices[0]]
    return accuracy_score(labels, cluster_classes)


def check_distance(dist_mtx, items="max"):
    if items == "max":
        np.fill_diagonal(dist_mtx, -np.inf)
        return np.unravel_index(np.argmax(dist_mtx), dist_mtx.shape) 
    if items == "min":
        np.fill_diagonal(dist_mtx, np.inf)
        return np.unravel_index(np.argmin(dist_mtx), dist_mtx.shape)
    return None

df = pd.DataFrame(data[1:], columns=data[0])

print(data[0])
data_id = [did for did in df.CODIGO if did != ""]
df.RENDA_MENSAL = [val.replace("","0") if val == "" else val.replace(",",".") for val in df.RENDA_MENSAL]
df.RENDA_MENSAL = df.RENDA_MENSAL.astype(float)

df.IDADE = df.IDADE.astype(int)
df.ESTADO_CIVIL = [standard_chars(rel.lower()) for rel in df.ESTADO_CIVIL]
df.GRAU_PARENTESCO = [standard_chars(rel.lower()) for rel in df.GRAU_PARENTESCO]


df["CONTRIBUINTE"] = [1 if renda_ind > 0 else 0 for renda_ind in df.RENDA_MENSAL] 

# df.RENDA_MENSAL

df['CNT_FAMILIA'] = [1 for _ in df.CODIGO]
group = df.groupby(['CODIGO'])
def_= group.DEFERIMENTO
group = pd.DataFrame(group.sum())


group["DEFERIMENTO"] = ["S" if "S" in list(i[1]) else "N" for i in def_]
group['RENDA_MEDIA'] = group["RENDA_MENSAL"] / group["CNT_FAMILIA"]


np.mean(group.RENDA_MEDIA)


print("DEFERIDOS - "+str(len(group[group.DEFERIMENTO == "S"])))
print("INDEFERIDOS - "+str(len(group[group.DEFERIMENTO == "N"])))

# print(group[group.DEFERIMENTO == "N"].RENDA_MEDIA.min())
print(group[group.DEFERIMENTO == "N"].RENDA_MEDIA.mean())
# print(group[group.DEFERIMENTO == "N"].RENDA_MEDIA.max())

# print(group[group.DEFERIMENTO == "S"].RENDA_MEDIA.min())
print(group[group.DEFERIMENTO == "S"].RENDA_MEDIA.mean())
# print(group[group.DEFERIMENTO == "S"].RENDA_MEDIA.max())

print(np.mean(group))


# import re

# rep = {",":".", "": "0"} # define desired replacements here

# # use these three lines to do the replacement
# rep = dict((re.escape(k), v) for k, v in rep.items())
# pattern = re.compile("|".join(rep.keys()))
# df.RENDA_MENSAL = [pattern.sub(lambda m: rep[re.escape(m.group(0))], val) for val in df.RENDA_MENSAL] 
# print
# df.RENDA_MENSAL = df.RENDA_MENSAL.astype(float)
# print(df.RENDA_MENSAL)

scaler = MinMaxScaler()
fnames = ["CONTRIBUINTE","CNT_FAMILIA", "RENDA_MEDIA", "IDADE"]
dataset = group[fnames]

import scipy as sp
#from scipy.io import mmwrite

#mmwrite("ass_estudantil.mtx", sp.sparse.csr_matrix(dataset.values).astype(float))

with open("ass_estudantil.mtx", "w") as mmtx:
    A = sp.sparse.csr_matrix(dataset.values)
    mmtx.write("% Matrix Market \n")
    mmtx.write(str(A.shape[0])+" "+str(A.shape[1])+" "+str(round(A.sum(), 3))+"\n")
    for line, col in zip(*A.nonzero()):
        mmtx.write(str(line+1)+" "+str(col+1)+" "+str(round(A[line, col],3))+"\n")

dataset = dataset.values
metric_ = "cosine"
# metric_ = "euclidean"
labels_ = group["DEFERIMENTO"].values

dist_mtx = pairwise_distances(dataset, metric=metric_, n_jobs=None)
print("Mean Dist. : "+str(np.mean(dist_mtx)))
items = check_distance(dist_mtx, items="max")
print("Max Dist. : "+str(np.max(dist_mtx)))
print(items[0], dataset[items[0]], labels_[items[0]])
print(items[1], dataset[items[1]], labels_[items[1]])
items = check_distance(dist_mtx, items="min")
print("Min Dist. : "+str(np.min(dist_mtx)))
print(items[0], dataset[items[0]], labels_[items[0]])
print(items[1], dataset[items[1]], labels_[items[1]])

dataset = scaler.fit_transform(dataset)

pred_ = []

labels_ = LabelBinarizer().fit_transform(labels_)

dist_mtx = pairwise_distances(dataset, metric=metric_, n_jobs=None)
colors = ["red", "blue"]

#add plot

for f1 in range(4):
    for f2 in range(f1+1, 4):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for lid, l_ in enumerate(np.unique(labels_)):
            subset = dataset[[did for did in range(dataset.shape[0]) if labels_[did] == l_], :]
            ax.scatter(subset[:, f1], subset[:, f2], alpha=0.3, label=str(l_))
        plt.xlabel(fnames[f1])
        plt.ylabel(fnames[f2])
        plt.title('Feature '+fnames[f1]+" vs Feature "+fnames[f2])
        plt.savefig("graf1"+fnames[f1]+"x"+fnames[f2]+".png")
        plt.close()

# CLUSTERING STEP
# for k in range(2, 6):
k = 2
pred_ = KMeans(n_clusters=k).fit_predict(dataset)
print("KMEANS: ["+str(k)+"] "+str(cluster_accuracy(labels_, pred_)))
print("Acc: "+str(cluster_information(labels_, pred_, dist_mtx)))

weighted_dataset = []
for c in np.unique(pred_):
    subset = dataset[[pid for pid, p in enumerate(pred_) if p == c], :]
    ga = Simple_GA(1000, 150, dataset, [p for pid, p in enumerate(pred_) if p == c]) #unique class
    best = ga.executeGA()
    print(best)
    wset = np.multiply(subset, best)
    [weighted_dataset.append(w) for w in wset if w != []]
    
weighted_mtx = pairwise_distances(weighted_dataset, metric=metric_, n_jobs=None)
print("Acc: "+str(cluster_information(labels_, pred_, weighted_mtx)))


#add plot
weighted_dataset = np.array(weighted_dataset)

for f1 in range(4):
    for f2 in range(f1+1, 4):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for lid, l_ in enumerate(np.unique(labels_)):
            subset = weighted_dataset[[did for did in range(weighted_dataset.shape[0]) if labels_[did] == l_], :]
            ax.scatter(subset[:, f1], subset[:, f2], alpha=0.3, label=str(l_))
        plt.xlabel(fnames[f1])
        plt.ylabel(fnames[f2])
        plt.title('Feature '+fnames[f1]+" vs Feature "+fnames[f2])
        plt.savefig("graf2"+fnames[f1]+"x"+fnames[f2]+".png")
        plt.close()

# pred_ = SpectralClustering(n_clusters=k).fit_predict(dataset)
# print("SPECTRAL: ["+str(k)+"] "+str(cluster_accuracy(labels_, pred_)))
# pred_ = AgglomerativeClustering(n_clusters=k).fit_predict(dataset)
# print("AGGLOMERATIVE: ["+str(k)+"] "+str(cluster_accuracy(labels_, pred_)))

# 
# print(np.unique(df.GRAU_PARENTESCO))
# print(np.unique(df.ESTADO_CIVIL))

# ga = Simple_GA(1000, 150, dataset, [1 for _ in range(dataset.shape[0])]) #unique class
# best = ga.executeGA()
# print(best)