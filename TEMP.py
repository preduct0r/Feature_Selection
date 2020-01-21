import sys, os
import pickle
import numpy as np
import pandas as pd


with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\best_indexes.pkl", "rb") as f:                                                                    # path_to_pickles
    best_indexes = pickle.load(f)
zz = pd.read_csv(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\best_features_corr_matrix.csv", index_col=0)
df_clusters_info = pd.read_csv(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\df_clusters_info.csv")


print(df_clusters_info.shape)
print(len(best_indexes))

temp =zz[zz>0.9].count()>1
temp_index = temp[temp==True].index


ff = zz[zz>0.9].count()
print(ff[ff.loc[temp_index]].index.value_counts())


print(zz.iloc[:,14].sort_values(ascending=False))


