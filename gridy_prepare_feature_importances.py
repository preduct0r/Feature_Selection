import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score, classification_report, precision_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel

with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\all_data.pkl", "rb") as f:
    [x_train, x_test, y_train, y_test] = pickle.load(f)


X = x_train.append(x_test, ignore_index=True)
y = y_train.append(y_test, ignore_index=True)


# =======================================================================================

# start_time = time.time()
#
# clf = LGBMClassifier(subsample=1.0, n_estimators=1500, min_split_gain=0, max_depth=8, learning_rate=0.01,
#                      colsample_bytree=0.7, num_leaves=65)
# clf.fit(X,y)
# dict_importance = {}
# for feature, importance in zip(X.columns, clf.feature_importances_):
#     dict_importance[feature] = importance
#
#
# df = pd.DataFrame(columns=['feature','importance'])
# df['feature'], df['importance'] = X.columns, clf.feature_importances_
# df.sort_values(by='importance', inplace=True)
#
# df.iloc[-1000:,:].to_csv(r'C:\Users\kotov-d\Documents\TASKS\feature_selection\xgb_all_range.csv')
# print("time of execution :", round(time.time()-start_time, 3))

#=======================================================================================
with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\not_correlated_X_columns.pkl","rb") as f:
    not_correlated_X = list(pickle.load(f))

X = X.loc[not_correlated_X]
# =======================================================================================

start_time = time.time()

clf = RandomForestClassifier(n_estimators=10000, criterion='entropy',
                              max_features='auto')


clf = clf.fit(X.loc[:,not_correlated_X], y)
model = SelectFromModel(clf, prefit=True)

df = pd.DataFrame(columns=['feature','importance','>mean'], index=range(len(not_correlated_X)))
df['feature'] = not_correlated_X
df['importance'] = clf.feature_importances_
df['>mean'] = model.get_support()

df.sort_values(by='importance', inplace=True)

df.iloc[-1000:,:].to_csv(r'C:\Users\kotov-d\Documents\TASKS\feature_selection\forest_all_range.csv', index=False)



