# Module
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from mst_clustering import MSTClustering
from sklearn.cluster import KMeans
from scipy.stats import skew, kurtosis
import antropy as ant
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.layers import Layer, Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# 시각화와 실루엣 출력
def draw_plot(data,pcscore, model, cluster_col, groups):
    print("=====" + model + '=====')
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.plot(group['PC1'],
                group['PC2'],
                marker='o',
                linestyle='',
                label=name)
    ax.legend(fontsize=12, loc='upper left')
    plt.xlabel('PC1', fontsize=14)
    plt.ylabel('PC2', fontsize=14)
    plt.show()
    sil_score = silhouette_score(data, pcscore[cluster_col], metric='euclidean')
    print(model+'_silhouette : {0:.3f}'.format(sil_score))
    print(pcscore[cluster_col].value_counts())

# PCA
def pca(data):
    pca = PCA(n_components = 2)
    pcscore = pca.fit_transform(data)

    dat_hat = pca.inverse_transform(pcscore)
    diff = pd.DataFrame(np.square(data - dat_hat))
    anomaly_score1 = diff.sum(axis=1)
    robust_cov = MinCovDet().fit(pcscore)
    anomaly_score2 = robust_cov.mahalanobis(pcscore)
    return anomaly_score1, anomaly_score2, pcscore

# AutoEncoder
def create_autoencoder(INPUT_SIZE):
    input_tensor = Input(shape=(INPUT_SIZE))
    #x = Dense(128, activation='sigmoid',kernel_initializer = 'he_normal')(input_tensor)
    x = Dense(64, activation='relu',kernel_initializer = 'he_normal')(input_tensor)
    x = Dense(32, activation='relu', kernel_initializer = 'he_normal')(x)
    x = Dense(16, activation='relu', kernel_initializer = 'he_normal')(x)
    x = Dense(4, activation='relu',kernel_initializer = 'he_normal')(x)
    x = Dense(16, activation='relu',kernel_initializer = 'he_normal')(x)
    x = Dense(32, activation='relu',kernel_initializer = 'he_normal')(x)
    x = Dense(64, activation='relu',kernel_initializer = 'he_normal')(x)
    #x = Dense(128, activation='relu')(x)
    #x = Dropout(rate=0.3)(x)

    output = Dense(INPUT_SIZE, activation='sigmoid')(x)

    model = Model(inputs=input_tensor, outputs=output)
    return model

# K-means Clustering
def kmeans_base(n_cluster, train_data, test_data, predict=False):
    kmeans = KMeans(n_clusters=n_cluster,init='k-means++',max_iter=300, random_state=0)
    if predict == True:
        clust = kmeans.fit(train_data)
        cluster_label = clust.predict(test_data)
    else:
        clust = kmeans.fit(train_data)
        cluster_label = clust.labels_
    return cluster_label

# BIRCH Clustering
def BIRCH_base(n_cluster, train_data, test_data, predict=False):
    birch = cluster.Birch(n_clusters=n_cluster, threshold=0.5)
    if predict == True:
        clust = birch.fit(train_data)
        cluster_label = clust.predict(test_data)
    else:
        clust = birch.fit(train_data)
        cluster_label = clust.labels_
    return cluster_label

# GMM Clustering
def gmm_base(n_cluster, train_data, test_data, predict=False):
    gmm = mixture.GaussianMixture(n_components=n_cluster, covariance_type='full')
    if predict == True:
        clust = gmm.fit(train_data)
        cluster_label = clust.predict(test_data)
    else:
        clust = gmm.fit(train_data)
        cluster_label = clust.predict(train_data)
    return cluster_label

# No of Cluster TEST
def cluster_number_test(data):
    cluster_num_candidates = range(2,8)
    kmeans_list = []
    birch_list = []
    gmm_list=[]
    for k in cluster_num_candidates:
        kmeans_labels = kmeans_base(k, data, data)
        birch_labels = BIRCH_base(k, data, data)
        gmm_labels = gmm_base(k, data, data)
        ss = silhouette_score(data, kmeans_labels, metric='euclidean')
        birch_ss = silhouette_score(data, birch_labels, metric='euclidean')
        gmm_ss = silhouette_score(data, gmm_labels, metric='euclidean')
        birch_list.append(birch_ss)
        kmeans_list.append(ss)
        gmm_list.append(gmm_ss)

    fig = plt.figure(figsize=(9, 6))
    fig.set_facecolor('white')

    plt.plot(cluster_num_candidates,kmeans_list, marker='o', label='kmeans')
    plt.plot(cluster_num_candidates, birch_list, marker='o', label='birch')
    plt.plot(cluster_num_candidates, gmm_list, marker='o', label='gmm')

    plt.legend()
    plt.xlabel('The Number of Cluster')
    plt.ylabel('Silhouette Value')
    plt.show()
    #print('kmeans:', kmeans_list)
    #print('birch:', birch_list)
    #print('gmm:', gmm_list)
    return kmeans_list, birch_list, gmm_list

# Clustering Compare TEST
def cluster_compare_test(data, params):
    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(data, n_neighbors=params["n_neighbors"], include_self=False)

    X = data
    # Create cluster objects
    # ============
    two_means = cluster.KMeans(init='k-means++', n_clusters=params["n_clusters"], random_state=0)
    birch = cluster.Birch(n_clusters=params["n_clusters"])
    gmm = mixture.GaussianMixture(random_state=0,
                                  n_components=params["n_clusters"], covariance_type="full")
    dbscan = cluster.DBSCAN(eps=params["eps"])
    mst = MSTClustering(cutoff_scale=3)
    ward = cluster.AgglomerativeClustering(
        n_clusters=params["n_clusters"], linkage="ward", connectivity=connectivity
    )

    clustering_algorithms = (
        ("KMeans", two_means),
        ("BIRCH", birch),
        ("GMM", gmm),
        ("DBSCAN", dbscan),
        ("MST", mst),
        #("Ward", ward)

    )
    dict = {}
    time_dict = {}
    for name, algorithm in clustering_algorithms:
        if (name == 'CLIQUE') or (name == 'SOM'):
            start_time = time.time()
            algorithm.process()
            clust = algorithm.get_clusters()
            end_time = time.time() - start_time
        elif (name == 'MST'):
            start_time = time.time()
            clust = algorithm.fit_predict(X)
            end_time = time.time() - start_time
        else:
            start_time = time.time()
            algorithm.fit(X)
            if hasattr(algorithm, "labels_"):
                clust = algorithm.labels_
            else:
                clust = algorithm.predict(X)
            end_time = time.time() - start_time
        ss = silhouette_score(data, clust, metric='euclidean')
        dict[name] = ss
        time_dict[name] = end_time
    #print(dict)
    #print(time_dict)
    return dict, time_dict