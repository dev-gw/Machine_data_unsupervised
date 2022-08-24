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
import module

# Data Loading
raw_data1 = pd.read_csv("/UHome/kgw32395/kgw/dataset/drying_actuator1.csv")
raw_data2 = pd.read_csv("/UHome/kgw32395/kgw/dataset/drying_actuator2.csv")
raw_data = pd.concat([raw_data1, raw_data2], axis = 0)
raw_data['index'] = raw_data.groupby(['Date']).cumcount() + 1
raw_data = raw_data.reset_index(drop = True)
print(raw_data)

# Data aggregation
raw_data.Date = raw_data.Date.str[2:16]
counts = raw_data.groupby(raw_data.Date).count()
curr_date = counts[counts.Sensor == 1024].index
flt_data = raw_data[raw_data.Date.isin(curr_date)]

dat_arr = flt_data.pivot(index='Date', columns='index', values='Sensor')
print(dat_arr)

# Feature extraction
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

# Original Data
dat_final_original = pd.DataFrame({
    "mean": dat_arr.mean(axis = 1),
    "std" : dat_arr.std(axis = 1),
    "q0" : dat_arr.min(axis = 1),
    "q1" : dat_arr.quantile(0.25, axis = 1),
    "q2" : dat_arr.median(axis = 1),
    "q3" : dat_arr.quantile(0.75, axis = 1),
    "q4" : dat_arr.max(axis = 1),
    "minmax" : dat_arr.max(axis = 1) - dat_arr.min(axis = 1)})

# After Feature Extraction
dat_final = pd.DataFrame({
    "abs_mean": abs_mean,
    "std": dat_arr.std(axis=1),
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
    "f1" : (np.sqrt(np.abs(dat_arr)).mean(axis=1)) ** 2,
    "f2" : peak - abs_mean,
    })

# Check Skewness
columns = dat_final.columns
for column in columns:
    print('skew_'+ column, skew(dat_final[column]))
print('\n')
for column in columns:
    print('kurtosis_'+ column, kurtosis(dat_final[column]))

# Skew Features
skew_col = ['q4','peak', 'minmax', 'Impulse', 'f1']
for col in skew_col:
    dat_final[col] = np.log1p(dat_final[col])
print(dat_final.isna().sum())

# Define Input
INPUT_SIZE = dat_final.shape[1]
print(INPUT_SIZE)

# Scale
std1 = StandardScaler()
dat_scaled = std1.fit_transform(dat_final)

# PCA 분석(기존연구)
anomaly_score1, anomaly_score2, pcscore = module.pca(dat_scaled)
## 임계값 새롭게 적용
pca_threshold_1 = np.mean(anomaly_score1) + 1 * np.std(anomaly_score1)
pca_threshold_2 = np.mean(anomaly_score2) + 1 * np.std(anomaly_score2)
ind = pd.DataFrame({"ind":
    np.where((anomaly_score1 > pca_threshold_1) | (anomaly_score2 > pca_threshold_2),
                "2. Anomaly", "1. Normal")})
pcscore = pd.concat([pd.DataFrame(pcscore, columns=['PC1','PC2']),ind], axis = 1)
groups = pcscore.groupby('ind')

module.draw_plot(dat_scaled,pcscore, 'pca','ind', groups)

# # # 정상 데이터만 분류
# dat_final['ind'] = ind.to_numpy()
# dat_filtered = dat_final[dat_final['ind'] == '1. Normal']
# dat_filtered.drop('ind', axis=1, inplace=True)
# std2 = StandardScaler()
# dat_filtered = std2.fit_transform(dat_filtered)
# print(dat_filtered.shape)

# AutoEncoder
model = module.create_autoencoder(INPUT_SIZE)
model.summary()
model.compile(optimizer=Adam(0.0001), loss='mse')
history = model.fit(dat_scaled, dat_scaled,
          epochs=50,
          batch_size=64,
          shuffle=True)

reconstructions = model.predict(dat_scaled)
train_loss = losses.mse(reconstructions, dat_scaled).numpy()
data_auto = dat_final.copy()
data2_auto['loss'] = train_loss.reshape(-1,1)
dat_scaled = std1.fit_transform(data2_auto)

# plt.hist(train_loss, bins=50)
# plt.xlabel("Mse")
# plt.ylabel("No. of examples")
# plt.show()

threshold = np.mean(train_loss) + 1 * np.std(train_loss)

ind = pd.DataFrame({'ind': np.where((train_loss > threshold), '2. Anomaly', '1. Normal')})
pcscore = pd.concat([pd.DataFrame(pcscore, columns=['PC1','PC2']),ind], axis = 1)
groups = pcscore.groupby('ind')

module.draw_plot('autoencoder',pcscore,'ind', groups)

# No of Cluster TEST
# kmeans_list, birch_list, gmm_list = module.cluster_number_test(dat_scaled)

# Clustering Compare TEST
params = {
    "quantile": 0.3,
    "eps": 0.6,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 2,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
}

dict, time_dict = cluster_compare_test(dat_scaled, params)

# Conclusion Result