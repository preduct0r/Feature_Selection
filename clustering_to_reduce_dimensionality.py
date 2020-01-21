import warnings
warnings.filterwarnings("ignore")

import sys, os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN


# достаем train, test данные
with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\vad_preprocessed.pkl", "rb") as f:                                                             # change path to base
    [x_train, x_test, y_train, y_test] = pickle.load(f)

vecs = []
for i in range(x_train.shape[1]):
    if x_train.iloc[:,i].isna().any()==True:
        print(i)
    vecs.append(x_train.iloc[:,i])
corr_matrix = pd.DataFrame(np.corrcoef(vecs))


def func(x):
    return 1-abs(x)

# почему-то 0 и 1512 фича считаются некорректно
corr_matrix = func(corr_matrix.drop(columns=[0,1512], index=[0,1512]))                # почему в одном месте 1512, в другом 1511?

# разбиваем фичи по кластерам
feature_indexes = list(range(x_train.shape[1]))
del feature_indexes[0]
# сначала удалили 0, поэтому 1511 вместо 1512
del feature_indexes[1511]


# Делаем кастомную affinity func
# def sim(x, y):
#     return corr_matrix.loc[int(x[0]),int(y[0])]
#
# def sim_affinity(X):
#     return pairwise_distances(X, metric=sim)
#

# # Обучаем Agglomerative Clustering с новым affinity
# cluster = AgglomerativeClustering(n_clusters=10, affinity=sim_affinity, linkage='average')
# cluster.fit(np.array(feature_indexes).reshape(-1, 1))

# Обучаем Agglomerative Clustering с "precomputed"
cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=0.1, affinity='precomputed', linkage='average')
cluster.fit(corr_matrix)                                                            # distance_threshold вместо n_cllusters

# # Обучаем DBSCAN с новым affinity
# cluster = DBSCAN(eps=0.03, min_samples=2, metric="precomputed")
# cluster.fit(corr_matrix)

# Делаю сводную таблицу соответствия фичей и лэйблов
clustering_labels = pd.DataFrame(columns=['feature', 'label'])
clustering_labels['feature'] = feature_indexes
clustering_labels['label'] = cluster.labels_


temp = pd.DataFrame(columns=['label', 'value_count'])
temp['value_count'] = clustering_labels.label.value_counts()
temp['label'] = clustering_labels.label.value_counts().index



# Находим самую репрезентативную фичу в каждом кластере, делаем сводную таблицу по кластерам
# и матрицу корреляции для отобранных фичей
best_indexes = []
grouped = clustering_labels.groupby('label')
df_clusters_info = pd.DataFrame(columns=['mean','max','std','label'])
df_clusters_info['label'] = clustering_labels.label.unique()
not_corr_idxs = []
for name,group in grouped:
   curr_features = group.feature.to_list()
   group_corr_matrix = corr_matrix[corr_matrix.index.isin(curr_features)].loc[:,curr_features]
   summ = group_corr_matrix.sum()
   maxx = group_corr_matrix.max().max()
   meann = group_corr_matrix.mean().mean()
   stdd = group_corr_matrix.std().mean()
   curr_idx = df_clusters_info[df_clusters_info.label == name].index[0]
   df_clusters_info.loc[curr_idx,:] = [round(meann,3), round(maxx,3), round(stdd,3), name]
   max_index = summ.idxmax()
   if name==-1:
       not_corr_idxs = list(group_corr_matrix.index)
   else:
       best_indexes.append(max_index)
best_indexes += not_corr_idxs
best_vecs = [vecs[x] for x in best_indexes]
zz = pd.DataFrame(data=abs(np.corrcoef(best_vecs)), index=best_indexes, columns=best_indexes)


df_clusters_info = df_clusters_info.merge(temp, how='left', on='label', left_index=True)
df_clusters_info = df_clusters_info.loc[:,['label','value_count','max','mean','std']]



# сохраняем сводную таблицу, матрицу корреляции и некоррелированные фичи
with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\best_indexes.pkl", "wb") as f:                                                                    # path_to_pickles
    pickle.dump(best_indexes, f)
zz.to_csv(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\best_features_corr_matrix.csv", index=True)
df_clusters_info.to_csv(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\df_clusters_info.csv", index=False)
# print(pd.read_csv(r"C:\Users\kotov-d\Documents\TASKS\task#7\best_features_corr_matrix.csv", index_col=0).iloc[:10,:10])