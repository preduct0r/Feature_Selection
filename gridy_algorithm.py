import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import pickle
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score


from gridy_alg_prepare_data import plus_L_minus_R, seq_forw_select



clf = LGBMClassifier(subsample=1.0, n_estimators=1200, min_split_gain=0, max_depth=8, learning_rate=0.01,
                     colsample_bytree=0.7, num_leaves=65)

with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\all_data.pkl", "rb") as f:
    [x_train, x_test, y_train, y_test] = pickle.load(f)


X = x_train.append(x_test, ignore_index=True)
y = y_train.append(y_test, ignore_index=True)
y = y.loc[:,'cur_label']



def criterion_func(features):
    # start_time = time.time()
    score = np.mean(cross_val_score(clf, X.loc[:,features], y, scoring='f1_macro', cv=2))
    # print(score)
    # print("time of execution :", round(time.time()-start_time, 3))

    return score



df = pd.read_csv(r'C:\Users\kotov-d\Documents\TASKS\feature_selection\forest_all_range.csv')
# df.drop(columns=['Unnamed: 0'], inplace=True)

dict_importance={}
for feature, importance in zip(df.feature, df.importance):
    dict_importance[feature] = importance


best_lgbm_features = []

for idx, w in enumerate(sorted(dict_importance, key=dict_importance.get, reverse=True)):
    if idx==500:
        break
    best_lgbm_features.append(w)


# best_features = seq_forw_select(best_lgbm_features, 300, criterion_func)

best_features = plus_L_minus_R(X.columns, 5, criterion_func, L=3, R=1)











