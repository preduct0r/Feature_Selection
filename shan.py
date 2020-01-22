import pickle
import numpy as np
import pandas as pd
import xgboost

import copy
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import f1_score, classification_report, precision_score
from prepare_data_with_two_labels import load_iemocap
import shap

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


# попытка использовать IEMOCAP
with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\iemocap_preprocessed.pkl", "rb") as f:
    [x_train, x_test, y_train, y_test] = pickle.load(f)


# # достаем train, test данные из telecom-vad и best_indexes
# with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\vad_preprocessed.pkl", "rb") as f:
#     [x_train, x_test, y_train, y_test] = pickle.load(f)



lgbm = LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.001, n_estimators=2000,
                             objective=None, min_split_gain=0, min_child_weight=3, min_child_samples=10, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=0, seed=17)
lgbm.fit(x_train, y_train)

# with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\lgbm.pkl", "wb") as f:
#     pickle.dump(lgbm, f)

# with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\lgbm.pkl", "rb") as f:
#     lgbm = pickle.load(f)

df = pd.DataFrame(columns=['feat_impo_lgbm', 'shan_tree', 'feat_impo_xgb'], index=x_test.columns)



for col in df.columns:
    if col=='shan_tree':
        zz = shap.TreeExplainer(lgbm, x_test)
        shap_values = zz.shap_values(x_test)

        temp_df = pd.DataFrame(columns=['feature', 'mean1', 'mean2', 'sum'], index=range(x_test.shape[1]))
        for idx, feature in enumerate(x_test.columns):
            mean1, mean2 = np.mean(abs(
                shap_values[:, idx])), 0  # np.mean(abs(shap_values[0][:, idx])), np.mean(abs(shap_values[1][:, idx]))
            temp_df.loc[idx, :] = [feature, mean1, mean2, mean1 + mean2]

        assert (list(temp_df.index) == list(temp_df.feature))

        temp_df.sort_values(by='mean1', ascending=False, inplace=True, axis=0)
        val = temp_df.feature.tolist()
        # with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\temp_df.pkl", "wb") as f:
        #     pickle.dump(temp_df, f)
        df[col] = val.copy()

    if col=='feat_impo_xgb':
        xgb = xgboost.XGBClassifier(learning_rate = 0.001, n_estimators = 2000, verbosity = 0, objective = 'binary:logistic',
                                    booster = 'gbtree', tree_method = 'auto', n_jobs = -1,
                                    gpu_id = 0, min_child_weight = 3, subsample = 0.8, colsample_bytree = 0.7)
        xgb.fit(x_train, y_train)

        dict_importance = {}
        for feature, importance in zip(x_train.columns, xgb.feature_importances_):
            dict_importance[feature] = importance

        best_xgb_features = []

        for idx, w in enumerate(sorted(dict_importance, key=dict_importance.get, reverse=True)):
            best_xgb_features.append(w)
        df[col] = best_xgb_features

    if col=='feat_impo_lgbm':
        dict_importance = {}
        for feature, importance in zip(x_train.columns, lgbm.feature_importances_):
            dict_importance[feature] = importance

        best_lgbm_features = []

        for idx, w in enumerate(sorted(dict_importance, key=dict_importance.get, reverse=True)):
            best_lgbm_features.append(w)
        df[col] = best_lgbm_features

df = (df+1)

df.to_csv(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\range_features(iemocap).csv", index=False)






