import pickle
import numpy as np
import pandas as pd
import xgboost

import copy
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import f1_score, classification_report, precision_score
from prepare_data_with_two_labels import load_iemocap, load_general
import shap
import os
import time

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')



lgbm = LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.001, n_estimators=2000,
                             objective=None, min_split_gain=0, min_child_weight=3, min_child_samples=10, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=0, seed=17)


path_to_features = r'C:\Users\kotov-d\Documents\TASKS\feature_selection\\features_to_calc'
path_to_calculated = r'\\Zstorage\!z\Shuranov\calculated_features'


for xXx in os.listdir(path_to_features):
    print(xXx[:-4])
    start = time.time()
    with open(os.path.join(path_to_features, xXx), "rb") as f:
        [x_train, x_test, y_train, y_test] = pickle.load(f)

    lgbm.fit(x_train, y_train)

    df = pd.DataFrame(columns=['feat_impo_lgbm', 'shan_tree', 'feat_impo_xgb'], index=x_test.columns)
    try:
        for col in df.columns:
            if col == 'shan_tree':
                try:
                    zz = shap.TreeExplainer(lgbm, x_test)
                    # shap_values = np.mean(abs(zz.shap_values(x_test)), axis=0)
                    shap_values = zz.shap_values(x_test)

                    temp_df = pd.DataFrame(columns=['col_idx', 'mean'], index=range(x_test.shape[1]))
                    temp_df['col_idx'] = x_test.columns
                    # df['mean'] = shap_values

                    means = []
                    for idx, row in df.iterrows():
                        means.append(np.mean(abs(shap_values[:, idx])))

                    temp_df['mean'] = means

                    temp_df.sort_values(by='mean', ascending=False, inplace=True, axis=0)

                    df[col] = temp_df.col_idx.tolist()

                except:
                    pass


            if col == 'feat_impo_xgb':
                xgb = xgboost.XGBClassifier(learning_rate=0.001, n_estimators=2000, verbosity=0,
                                            objective='binary:logistic',
                                            booster='gbtree', tree_method='auto', n_jobs=-1,
                                            gpu_id=0, min_child_weight=3, subsample=0.8, colsample_bytree=0.7)
                xgb.fit(x_train, y_train)

                dict_importance = {}
                for feature, importance in zip(x_train.columns, xgb.feature_importances_):
                    dict_importance[feature] = importance

                best_xgb_features = []

                for idx, w in enumerate(sorted(dict_importance, key=dict_importance.get, reverse=True)):
                    best_xgb_features.append(w)
                df[col] = best_xgb_features

            if col == 'feat_impo_lgbm':
                dict_importance = {}
                for feature, importance in zip(x_train.columns, lgbm.feature_importances_):
                    dict_importance[feature] = importance

                best_lgbm_features = []

                for idx, w in enumerate(sorted(dict_importance, key=dict_importance.get, reverse=True)):
                    best_lgbm_features.append(w)
                df[col] = best_lgbm_features

        print('время выполнения при {} фичах: {} сек'.format(x_train.shape[1], int(round(time.time() - start, 0))))

    except:
        pass

    df.iloc[:1000, :].to_csv(os.path.join(path_to_calculated, xXx[:-4] + '.csv'), index=False)













