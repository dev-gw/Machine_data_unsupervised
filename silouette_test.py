import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.metrics import silhouette_score, silhouette_samples

original = np.array([[1,2],[2,1],[5,6],[5,7],[6,6],[9,1],[9,2],[10,1],[10,2]])

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0)
kmeans.fit(original)
labels = np.array(kmeans.labels_)
print(silhouette_score(original, labels))

def get_sum_distance(target_x, target_cluster):
    res = np.sum([np.linalg.norm(target_x - x) for x in target_cluster])
    return res


def get_silhouette_results(X, labels):
    uniq_labels = np.unique(labels)
    silhouette_val_list = []
    for i in range(len(labels)):
        target_data = X[i]

        ## calculate a(i)
        target_label = labels[i]
        target_cluster_data_idx = np.where(labels == target_label)[0]

        if len(target_cluster_data_idx) == 1:
            silhouette_val_list.append(0)
            continue
        else:
            target_cluster_data = X[target_cluster_data_idx]
            a_i_sum = get_sum_distance(target_data, target_cluster_data)
            a_i = a_i_sum / (target_cluster_data.shape[0] - 1)
            print('a',a_i)
        ## calculate b(i)
        b_i_list = []
        # select different label
        label_list = uniq_labels[np.unique(labels) != target_label]
        for ll in label_list:
            other_cluster_data_idx = np.where(labels == ll)[0]
            other_cluster_data = X[other_cluster_data_idx]
            b_i_sum = get_sum_distance(target_data, other_cluster_data)
            temp_b_i = b_i_sum / other_cluster_data.shape[0]
            b_i_list.append(temp_b_i)

        b_i = min(b_i_list)
        print('b', b_i)
        s_i =(b_i - a_i) / max(a_i, b_i)
        print('s', s_i)
        silhouette_val_list.append(s_i)

    silhouette_coef = np.mean(silhouette_val_list)
    return (silhouette_coef, np.array(silhouette_val_list))

get_silhouette_results(original, labels)