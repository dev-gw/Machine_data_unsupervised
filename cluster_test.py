import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.metrics import silhouette_score, silhouette_samples

# Test Data
original = pd.DataFrame([[1,2],[2,1],[5,6],[5,7],[6,6],[9,1],[9,2],[10,1],[10,2]], columns=['x','y'])
ex1_1 = pd.DataFrame([[1,2],[2,1],[5,7],[5,8],[6,7],[9,1],[9,2],[10,1],[10,2]], columns=['x','y'])
ex1_2 = pd.DataFrame([[1,2],[2,1],[5,5],[5,6],[6,5],[9,1],[9,2],[10,1],[10,2]], columns=['x','y'])
ex2_1 = pd.DataFrame([[1,2],[3,1],[5,6],[5,7],[7,6],[9,1],[9,2],[11,1],[11,2]], columns=['x','y'])
ex2_2 = pd.DataFrame([[1,2],[1,1],[5,6],[5,7],[5,6],[9,1],[9,2],[9,1],[9,2]], columns=['x','y'])
ex3 = pd.DataFrame([[1,3],[2,2],[1,2],[2,1],[5,6],[5,7],[6,6],[9,1],[9,2],[10,1],[10,2]], columns=['x','y'])
ex4 = pd.DataFrame([[1,2],[2,1],[5,6],[5,7],[6,6],[9,1],[9,2],[10,1],[10,2]], columns=['x','y'])

# 군집간의 거리 계산
def calculate_outer(list):
    distance01 = distance.euclidean(list[0], list[1])
    distance02 = distance.euclidean(list[0], list[2])
    distance03 = distance.euclidean(list[1], list[2])
    return (distance01 + distance02 + distance03) / 3

# 군집내부 거리 계산
def calculate_inner(list, df):
    total = []
    for i in range(len(list)):
        distance = 0
        tmp = df.index[df['clust'] == i].to_list()
        for j in tmp:
            distance += np.linalg.norm(np.array(list[i]) - np.array(df.iloc[j,0:2]))
        total.append(distance / len(tmp))
    return np.array(total).mean()

def experiment(data):
    # K-means
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=0)
    kmeans.fit(data)
    data['clust'] = kmeans.labels_
    # 중심점
    center = kmeans.cluster_centers_
    ## 군집이 섞여있다고 가정할때만
    # cl2 = np.array([[1, 2], [5, 6]]).mean()
    # cl0 = np.array([[2, 1], [5, 7], [6, 6]]).mean()
    # cl1 = np.array([[9,1],[9, 2], [10, 1], [10, 2]]).mean()
    # center = []
    # center.append(cl0)
    # center.append(cl1)
    # center.append(cl2)
    # data.iloc[1,2] = 0
    # data.iloc[2,2] = 2
    # 시각화
    target_list = np.unique(data['clust'])
    markers = ['o', 's', '^']
    for target in target_list:
        target_cluster = data[data['clust'] == target]
        plt.scatter(x=target_cluster['x'], y=target_cluster['y'], edgecolors='k', marker=markers[target])
        plt.xlabel('X', fontsize=20)
        plt.xlim([0, 15])
        plt.ylabel('Y', fontsize=20)
        plt.ylim([0, 15])
    plt.show()

    # 실루엣계수
    sil_score = silhouette_score(data[['x', 'y']], data['clust'], metric='euclidean')
    print('군집 간 거리:', calculate_outer(center), '군집 내 거리:', calculate_inner(center, data),
    '실루엣 계수:', sil_score)

experiment(ex4)


