#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from fcmeans import FCM
from mst_clustering import MSTClustering
from sklearn.cluster import KMeans
from scipy.stats import skew, kurtosis
import antropy as ant



def draw_plot(model, cluster_col, groups):
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
    sil_score = silhouette_score(dat_scaled, pcscore[cluster_col], metric='euclidean')
    print(model+'_silhouette : {0:.3f}'.format(sil_score))
    print(pcscore[cluster_col].value_counts())

raw_data1 = pd.read_csv("/UHome/kgw32395/kgw/dataset/drying_actuator1.csv")
raw_data2 = pd.read_csv("/UHome/kgw32395/kgw/dataset/drying_actuator2.csv")
raw_data = pd.concat([raw_data1, raw_data2], axis = 0)
raw_data['index'] = raw_data.groupby(['Date']).cumcount() + 1
raw_data = raw_data.reset_index(drop = True)

print(raw_data)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import silhouette_score, silhouette_samples

raw_data.Date = raw_data.Date.str[2:16]
counts = raw_data.groupby(raw_data.Date).count()
curr_date = counts[counts.Sensor == 1024].index
flt_data = raw_data[raw_data.Date.isin(curr_date)]

dat_arr = flt_data.pivot(index='Date', columns='index', values='Sensor')
print(dat_arr)

# extraction
abs_mean = np.abs(dat_arr).mean(axis = 1)
std = dat_arr.std(axis = 1)
q0 = dat_arr.min(axis = 1)
q1 = dat_arr.quantile(0.25, axis = 1)
q3 = dat_arr.quantile(0.75, axis = 1)
q4  = dat_arr.max(axis = 1)
peak = np.abs(dat_arr).max(axis=1)
minmax = dat_arr.max(axis = 1) - dat_arr.min(axis = 1)
Impulse = np.abs(dat_arr).max(axis=1) / np.abs(dat_arr).mean(axis=1)
RMS = np.sqrt((dat_arr**2).mean(axis=1))
Shape = np.sqrt((dat_arr**2).mean(axis=1)) / np.sqrt(np.abs(dat_arr).mean(axis=1))
iqr = dat_arr.quantile(0.75, axis = 1) - dat_arr.quantile(0.25, axis = 1)

dat_final_raw = pd.DataFrame({
    "mean": dat_arr.mean(axis = 1),
    "std" : dat_arr.std(axis = 1),
    "q0" : dat_arr.min(axis = 1),
    "q1" : dat_arr.quantile(0.25, axis = 1),
    "q2" : dat_arr.median(axis = 1),
    "q3" : dat_arr.quantile(0.75, axis = 1),
    "q4" : dat_arr.max(axis = 1),
    "minmax" : dat_arr.max(axis = 1) - dat_arr.min(axis = 1)})

dat_final = pd.DataFrame({
    "abs_mean": abs_mean,
    "std" : std,
    "q0" : q0,
    "q1" : q1,
    "q3" : q3,
    "q4" : q4,
    "peak": peak,
    "minmax" : minmax,
    "Impulse" : Impulse,
    "RMS" : RMS,
    "Shape" : Shape,
    "iqr": iqr,
    "mad" : dat_arr.mad(axis=1),
    #"f1": dat_arr.quantile(0.25, axis=1) - dat_arr.min(axis=1),
    #"f2": dat_arr.max(axis = 1) - dat_arr.quantile(0.25, axis = 1),
    #"f3": np.sqrt((dat_arr**2).mean(axis=1)) - dat_arr.mean(axis=1),
    #"f4": dat_arr.max(axis=1) - dat_arr.quantile(0.75, axis=1),
    "f1": peak - abs_mean
    })

# columns = ['abs_mean', 'std', 'q0','q1', 'q3','q4','peak','minmax','Impulse','RMS','Shape','iqr','mad','f1']
# for column in columns:
#     print('skew_'+ column, skew(dat_final[column]))
# print('\n')
# for column in columns:
#     print('kurtosis_'+ column, kurtosis(dat_final[column]))

# dat_final['fourier'] = 0
# for i in range(len(dat_arr)):
#     dat_final.iloc[i,12] = np.fft.fft(np.array(dat_arr.iloc[i,:])) / 1024

# print(dat_final)


# columns = ['abs_mean', 'std', 'q0','q1', 'q3','q4','peak','minmax','Impulse','RMS','Shape','iqr','mad','f1']
# for column in columns:
#     print('skew_'+ column, skew(dat_final[column]))
# print('\n')
# for column in columns:
#     print('kurtosis_'+ column, kurtosis(dat_final[column]))

# #진동데이터 시각화
# f, axes = plt.subplots(3,2, squeeze=False)
# f.set_size_inches(15,10)
# plt.subplots_adjust(wspace=0.3, hspace=0.3)
# axes[0][0].hist(abs_mean)
# axes[0][0].set_title('abs_mean')
# axes[0][1].hist(std)
# axes[0][1].set_title('std')
# axes[1][0].hist(q0)
# axes[1][0].set_title('q0')
# axes[1][1].hist(q1)
# axes[1][1].set_title('q1')
# axes[2][0].hist(q3)
# axes[2][0].set_title('q3')
# axes[2][1].hist(q4)
# axes[2][1].set_title('q4')
# plt.show()
#
# f, axes = plt.subplots(3,2, squeeze=False)
# f.set_size_inches(15,10)
# plt.subplots_adjust(wspace=0.3, hspace=0.3)
# axes[0][0].hist(peak)
# axes[0][0].set_title('peak')
# axes[0][1].hist(minmax)
# axes[0][1].set_title('minmax')
# axes[1][0].hist(RMS)
# axes[1][0].set_title('rms')
# axes[1][1].hist(Shape)
# axes[1][1].set_title('shape')
# axes[2][0].hist(iqr)
# axes[2][0].set_title('iqr')
# plt.show()


dat_final['peak'] = np.log1p(dat_final['peak'])
dat_final['Impulse'] = np.log1p(dat_final['Impulse'])
dat_final['minmax'] = np.log1p(dat_final['minmax'])
#dat_final['RMS'] = np.log1p(dat_final['RMS'])
#dat_final['Shape'] = np.log1p(dat_final['Shape'])
dat_final['q4'] = np.log1p(dat_final['q4'])
#dat_final['q0'] = np.log1p(dat_final['q0']+5.8)
#dat_final['iqr'] = np.log1p(dat_final['iqr'])
#dat_final['abs_mean'] = np.log1p(dat_final['abs_mean'])
#dat_final['q1'] = np.log1p(dat_final['q1'])
#dat_final['q3'] = np.log1p(dat_final['q3'])
#dat_final['f2'] = np.log1p(dat_final['f2'])
#dat_final['f5'] = np.log1p(dat_final['f4'])
#dat_final['mad'] = np.log1p(dat_final['mad'])
dat_final['f1'] = np.log1p(dat_final['f1'])



print(dat_final.isna().sum())

INPUT_SIZE = dat_final.shape[1]
print(INPUT_SIZE)
std1 = StandardScaler()
dat_scaled = std1.fit_transform(dat_final)

# print(dat_arr)
# plt.figure(figsize = (50,50))
# plt.plot(dat_arr.iloc[:,100])
# plt.show()


# PCA 분석(기존연구)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

scov = np.cov(dat_scaled.T)
eigen_vals, eigen_vecs=np.linalg.eig(scov)
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1,INPUT_SIZE+1), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1,INPUT_SIZE+1), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc = 'best')
plt.show()

pca = PCA(n_components = 2)
pcscore = pca.fit_transform(dat_scaled)
# plt.scatter(pcscore[:,0], pcscore[:,1])

from sklearn.covariance import EmpiricalCovariance, MinCovDet

dat_hat = pca.inverse_transform(pcscore)
diff = pd.DataFrame(np.square(dat_scaled - dat_hat))
anomaly_score1 = diff.sum(axis=1)
print(anomaly_score1)
robust_cov = MinCovDet().fit(pcscore)
anomaly_score2 = robust_cov.mahalanobis(pcscore)

# #새로 적용
pca_threshold_1 = np.mean(anomaly_score1) + 1 * np.std(anomaly_score1)
pca_threshold_2 = np.mean(anomaly_score2) + 1 * np.std(anomaly_score2)
#
ind = pd.DataFrame({"ind":
    np.where((anomaly_score1 > pca_threshold_1) | (anomaly_score2 > pca_threshold_2),
                "2. Anomaly", "1. Normal")})
pcscore = pd.concat([pd.DataFrame(pcscore, columns=['PC1','PC2']),ind], axis = 1)
groups = pcscore.groupby('ind')

draw_plot('pca','ind', groups)

# # # 정상 데이터만 분류
# dat_final['ind'] = ind.to_numpy()
# dat_filtered = dat_final[dat_final['ind'] == '1. Normal']
# dat_filtered.drop('ind', axis=1, inplace=True)
# std2 = StandardScaler()
# dat_filtered = std2.fit_transform(dat_filtered)
# print(dat_filtered.shape)
#
#
# AutoEncoder
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.layers import Layer, Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# def create_model():
#     input_tensor = Input(shape=(INPUT_SIZE))
#     #x = Dense(128, activation='sigmoid',kernel_initializer = 'he_normal')(input_tensor)
#     x = Dense(64, activation='relu',kernel_initializer = 'he_normal')(input_tensor)
#     x = Dense(32, activation='relu', kernel_initializer = 'he_normal')(x)
#     x = Dense(16, activation='relu', kernel_initializer = 'he_normal')(x)
#     x = Dense(4, activation='relu',kernel_initializer = 'he_normal')(x)
#     x = Dense(16, activation='relu',kernel_initializer = 'he_normal')(x)
#     x = Dense(32, activation='relu',kernel_initializer = 'he_normal')(x)
#     x = Dense(64, activation='relu',kernel_initializer = 'he_normal')(x)
#     #x = Dense(128, activation='relu')(x)
#     #x = Dropout(rate=0.3)(x)
#
#     output = Dense(INPUT_SIZE, activation='sigmoid')(x)
#
#     model = Model(inputs=input_tensor, outputs=output)
#     return model
#
# model = create_model()
# model.summary()
# model.compile(optimizer=Adam(0.0001), loss='mse')
# history = model.fit(dat_scaled, dat_scaled,
#           epochs=50,
#           batch_size=64,
#           shuffle=True)
#
# reconstructions = model.predict(dat_scaled)
# train_loss = losses.mse(reconstructions, dat_scaled).numpy()
# data2 = dat_final.copy()
# data2['loss'] = train_loss.reshape(-1,1)
# dat_scaled = std1.fit_transform(data2)
# print(data2)
#
# plt.hist(train_loss, bins=50)
# plt.xlabel("Mse")
# plt.ylabel("No. of examples")
# plt.show()
#
# threshold = np.mean(train_loss) + 1 * np.std(train_loss)
# print("Threshold : ",threshold)
#
# ind = pd.DataFrame({'ind': np.where((train_loss > threshold), '2. Anomaly', '1. Normal')})
# pcscore = pd.concat([pd.DataFrame(pcscore, columns=['PC1','PC2']),ind], axis = 1)
# groups = pcscore.groupby('ind')
#
# draw_plot('autoencoder', 'ind', groups)
#

# K-means Clustering

# kmeans = KMeans(n_clusters=3,init='k-means++',max_iter=300, random_state=0)
# cluster = kmeans.fit(dat_scaled)
# clust_df = pd.DataFrame(dat_scaled).copy()
# clust_df['clust'] = cluster.labels_
# print(clust_df.head())
#
# pcscore = pd.concat([pd.DataFrame(pcscore, columns=['PC1','PC2']),clust_df['clust']], axis = 1)
# groups = pcscore.groupby('clust')
# draw_plot('K-means', 'clust', groups)
#
#
# # BIRCH Clustering
# birch = cluster.Birch(n_clusters=3).fit(dat_scaled)
# cluster = birch.fit(dat_scaled)
# clust_df = pd.DataFrame(dat_scaled).copy()
# clust_df['clust'] = cluster.labels_
# pcscore = pd.concat([pd.DataFrame(pcscore, columns=['PC1','PC2']),clust_df['clust']], axis = 1)
# groups = pcscore.groupby('clust')
# draw_plot('Birch', 'clust', groups)


# TEST
cluster_num_candidates = range(2,8)
kmeans_list = []
birch_list = []
gmm_list=[]
for k in cluster_num_candidates:
    kmeans = cluster.KMeans(n_clusters=k, init='k-means++', random_state=0).fit(dat_scaled)
    kmeans_labels = kmeans.labels_
    birch = cluster.Birch(n_clusters=k, threshold=0.5).fit(dat_scaled)
    birch_labels = birch.labels_
    gmm = mixture.GaussianMixture(n_components=k, covariance_type="full")
    gmm_labels = gmm.fit_predict(dat_scaled)
    ss = silhouette_score(dat_scaled, kmeans_labels, metric='euclidean')
    birch_ss = silhouette_score(dat_scaled, birch_labels, metric='euclidean')
    gmm_ss = silhouette_score(dat_scaled, gmm_labels, metric='euclidean')
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
print('kmeans:', kmeans_list)
print('birch:', birch_list)
print('gmm:', gmm_list)




# # Cluster test 1
# params = {
#     "quantile": 0.3,
#     "eps": 0.6,
#     "damping": 0.9,
#     "preference": -200,
#     "n_neighbors": 3,
#     "n_clusters": 3,
#     "min_samples": 7,
#     "xi": 0.05,
#     "min_cluster_size": 0.1,
# }
#
# # connectivity matrix for structured Ward
# connectivity = kneighbors_graph(dat_scaled, n_neighbors=params["n_neighbors"], include_self=False)
#
# X = dat_scaled
# # Create cluster objects
# # ============
# two_means = cluster.KMeans(init='k-means++', n_clusters=params["n_clusters"], random_state=0)
# birch = cluster.Birch(n_clusters=params["n_clusters"])
# #fcm = FCM(n_clusters=params["n_clusters"])
# gmm = mixture.GaussianMixture(random_state=0,
#     n_components=params["n_clusters"], covariance_type="full")
# dbscan = cluster.DBSCAN(eps=params["eps"])
# mst = MSTClustering(cutoff_scale=3)
# ward = cluster.AgglomerativeClustering(
#          n_clusters=params["n_clusters"], linkage="ward", connectivity=connectivity
#      )
#
#
# clustering_algorithms = (
#         ("KMeans", two_means),
#         ("BIRCH", birch),
#         ("GMM", gmm),
#         ("DBSCAN",dbscan),
#         ("MST", mst),
#         ("Ward", ward)
#
# )
# dict = {}
# time_dict = {}
# for name, algorithm in clustering_algorithms:
#     if (name == 'CLIQUE') or (name == 'SOM'):
#         start_time = time.time()
#         algorithm.process()
#         cluster = algorithm.get_clusters()
#         end_time = time.time()-start_time
#     elif (name == 'MST'):
#         start_time = time.time()
#         cluster = algorithm.fit_predict(X)
#         end_time = time.time()-start_time
#     else:
#         start_time = time.time()
#         algorithm.fit(X)
#         if hasattr(algorithm, "labels_"):
#             cluster = algorithm.labels_
#         else:
#             cluster = algorithm.predict(X)
#         end_time = time.time()-start_time
#     ss = silhouette_score(dat_scaled, cluster, metric='euclidean')
#     dict[name] = ss
#     time_dict[name] = end_time
# print(dict)
# print(time_dict)


# for i_dataset, (dataset, algo_params) in enumerate(datasets):
#     # update parameters with dataset-specific values
#     params = default_base.copy()
#     params.update(algo_params)
#
#     X, y = dataset
#
#     # normalize dataset for easier parameter selection
#     X = StandardScaler().fit_transform(X)
#
#     # estimate bandwidth for mean shift
#     bandwidth = cluster.estimate_bandwidth(X, quantile=params["quantile"])
#
#
#     # make connectivity symmetric
#     connectivity = 0.5 * (connectivity + connectivity.T)
#
#     # ============
#     # Create cluster objects
#     # ============
#     ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
#     two_means = cluster.MiniBatchKMeans(n_clusters=params["n_clusters"])
#     ward = cluster.AgglomerativeClustering(
#         n_clusters=params["n_clusters"], linkage="ward", connectivity=connectivity
#     )
#     spectral = cluster.SpectralClustering(
#         n_clusters=params["n_clusters"],
#         eigen_solver="arpack",
#         affinity="nearest_neighbors",
#     )
#     dbscan = cluster.DBSCAN(eps=params["eps"])
#     optics = cluster.OPTICS(
#         min_samples=params["min_samples"],
#         xi=params["xi"],
#         min_cluster_size=params["min_cluster_size"],
#     )
#     affinity_propagation = cluster.AffinityPropagation(
#         damping=params["damping"], preference=params["preference"], random_state=0
#     )
#     average_linkage = cluster.AgglomerativeClustering(
#         linkage="average",
#         affinity="cityblock",
#         n_clusters=params["n_clusters"],
#         connectivity=connectivity,
#     )
#     birch = cluster.Birch(n_clusters=params["n_clusters"])
#     gmm = mixture.GaussianMixture(
#         n_components=params["n_clusters"], covariance_type="full"
#     )
#
#     clustering_algorithms = (
#         ("MiniBatch\nKMeans", two_means),
#         ("Affinity\nPropagation", affinity_propagation),
#         ("MeanShift", ms),
#         ("Spectral\nClustering", spectral),
#         ("Ward", ward),
#         ("Agglomerative\nClustering", average_linkage),
#         ("DBSCAN", dbscan),
#         ("OPTICS", optics),
#         ("BIRCH", birch),
#         ("Gaussian\nMixture", gmm),
#     )