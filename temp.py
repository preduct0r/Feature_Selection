import os
import pickle
from lightgbm import LGBMClassifier
from prepare_data_with_two_labels import load_iemocap, load_general,  prep_all_data
import time
import pandas as pd

# for i in os.listdir(r'C:\Users\kotov-d\Documents\BASES\undefined_base\feature'):
#     load_general(str(i))

prep_all_data()

# start_time = time.time()
#
# with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\all_data.pkl", "rb") as f:
#     [x_train, x_test, y_train, y_test] = pickle.load(f)
#
#
# X = x_train.append(x_test, ignore_index=True)
# y = y_train.append(y_test)
#
# print(X.shape)

#
# clf = LGBMClassifier(subsample=1.0, n_estimators=2000, min_split_gain=0, max_depth=8, learning_rate=0.01,
#                      colsample_bytree=0.7, num_leaves=65)
# clf.fit(X,y)
# dict_importance = {}
# for feature, importance in zip(X.columns, clf.feature_importances_):
#     dict_importance[feature] = importance


# df = pd.DataFrame(columns=['feature','importance'])
# df['feature'], df['importance'] = X.columns, clf.feature_importances_
# df.sort_values(by='importance', inplace=True)
#
# df.iloc[-1000:,:].to_csv(r'C:\Users\kotov-d\Documents\TASKS\feature_selection\xgb_all_range.csv')
# print("time of execution :", round(time.time()-start_time, 3))
