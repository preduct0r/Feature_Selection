import sys, os
import pickle
import numpy as np
import pandas as pd



base_dir = r'C:\Users\preductor\Google Диск\STC\базы'                                                   # path_to_bases
features_path = os.path.join(base_dir, 'telecom_vad', 'feature', 'opensmile')


with open(os.path.join(features_path, 'x_train.pkl'), 'rb') as f:
    x_train = pd.DataFrame(np.vstack(pickle.load(f)))
with open(os.path.join(features_path, 'x_test.pkl'), 'rb') as f:
    x_test = pd.DataFrame(np.vstack(pickle.load(f)))
with open(os.path.join(features_path, 'y_train.pkl'), 'rb') as f:
    y_train = pickle.load(f).loc[:, 'cur_label']
with open(os.path.join(features_path, 'y_test.pkl'), 'rb') as f:
    y_test = pickle.load(f).loc[:, 'cur_label']



x_train['target'] = y_train
x_test['target'] = y_test

x_train = x_train[x_train.target!='defective'][x_train.target!='sad'][x_train.target!='not_ informative']
y_train = x_train.target
x_train.drop(columns=['target'], inplace=True)

x_test = x_test[x_test.target!='defective'][x_test.target!='sad'][x_test.target!='not_ informative']
y_test = x_test.target
x_test.drop(columns=['target'], inplace=True)


with open("path_to_pickle", "wb") as f:                                                                    # path_to_pickles
    pickle.dump([x_train, x_test, y_train, y_test], f)