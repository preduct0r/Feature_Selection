import warnings
warnings.filterwarnings("ignore")

import os
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, classification_report, precision_score

from prepare_data_with_two_labels import load_iemocap, load_general, prep_all_data

from sklearn.ensemble import RandomForestClassifier

import pickle
import pandas as pd
import numpy as np
import time




# Sequential Forward Selection (SFS)
# код взят здесь
# https://nbviewer.jupyter.org/github/rasbt/pattern_classification/blob/master/dimensionality_reduction/feature_selection/sequential_selection_algorithms.ipynb
def seq_forw_select(features, max_k, criterion_func, print_steps=False):
    start_time = time.time()

    """
    Implementation of a Sequential Forward Selection algorithm.

    Keyword Arguments:
        features (list): The feature space as a list of features.
        max_k: Termination criterion; the size of the returned feature subset.
        criterion_func (function): Function that is used to evaluate the
            performance of the feature subset.
        print_steps (bool): Prints the algorithm procedure if True.

    Returns the selected feature subset, a list of features of length max_k.

    """

    # Initialization
    feat_sub = []
    k = 0
    d = len(features)
    if max_k > d:
        max_k = d

    loop = 0
    while True:


        # Inclusion step
        crit_func_max = criterion_func(feat_sub + [features[0]])
        best_feat = features[0]
        for x in features[1:]:
            crit_func_eval = criterion_func(feat_sub + [x])
            print()
            if crit_func_eval > crit_func_max:
                crit_func_max = crit_func_eval
                best_feat = x
        feat_sub.append(best_feat)
        if print_steps:
            print('include: {}'.format(best_feat))
        features.remove(best_feat)

        # Termination condition
        k = len(feat_sub)
        if k == max_k:
            break

        loop+=1

    print("time of execution :", round(time.time()-start_time, 3))
    return feat_sub


from lightgbm import LGBMClassifier



def criterion_func(features):
    # start_time = time.time()
    score = np.mean(cross_val_score(clf, X.loc[:,features], y, scoring='f1_macro', cv=3))
    # print(score)
    # print("time of execution :", round(time.time()-start_time, 3))

    return score




# for file in os.listdir(r'C:\Users\kotov-d\Documents\TASKS\feature_selection\small_features'):
#     print(file[2:-4])
#     with open(os.path.join(r'C:\Users\kotov-d\Documents\TASKS\feature_selection\small_features', file), "rb") as f:
#         [x_train, x_test, y_train, y_test] = pickle.load(f)
with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\all_data.pkl", "rb") as f:
    [x_train, x_test, y_train, y_test] = pickle.load(f)

X = x_train.append(x_test, ignore_index=True)
y = y_train.append(y_test)
#=======================================================================================
start_time = time.time()

clf = LGBMClassifier(subsample=1.0, n_estimators=10, min_split_gain=0, max_depth=8, learning_rate=0.01,
                     colsample_bytree=0.7, num_leaves=65)
clf.fit(X,y)
dict_importance = {}
for feature, importance in zip(X.columns, clf.feature_importances_):
    dict_importance[feature] = importance


df = pd.DataFrame(columns=['feature','importance'])
df['feature'], df['importance'] = X.columns, clf.feature_importances_
df.sort_values(by='importance', inplace=True)

df.iloc[:1000,:].to_csv(r'C:\Users\kotov-d\Documents\TASKS\feature_selection\xgb_all_range.csv')
print("time of execution :", round(time.time()-start_time, 3))


# =======================================================================================
start_time = time.time()

clf2 = RandomForestClassifier(n_estimators=10,
                              max_depth=8,
                              max_features='sqrt')
clf.fit(X, y)
dict_importance2 = {}
for feature, importance in zip(X.columns, clf.feature_importances_):
    dict_importance[feature] = importance


df = pd.DataFrame(columns=['feature', 'importance'])
df['feature'], df['importance'] = X.columns, clf.feature_importances_
df.sort_values(by='importance', inplace=True)

df.iloc[:1000, :].to_csv(r'C:\Users\kotov-d\Documents\TASKS\feature_selection\forest_all_range.csv')
print("time of execution :", round(time.time()-start_time, 3))


# =======================================================================================
best_lgbm_features = []

for idx, w in enumerate(sorted(dict_importance, key=dict_importance.get, reverse=True)):
    if idx==500:
        break
    best_lgbm_features.append(w)

X_best = X.loc[:,best_lgbm_features]


best_features = seq_forw_select(best_lgbm_features, 500, criterion_func, print_steps=True)

with open(os.path.join(r'C:\Users\kotov-d\Documents\TASKS\feature_selection\grid_alg_results', 'all_best_features'), "wb") as f:
    pickle.dump([x_train, x_test, y_train, y_test], f)










