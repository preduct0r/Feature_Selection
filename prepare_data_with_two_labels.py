import sys, os
import pickle
import numpy as np
import pandas as pd


def load_vad():
    base_dir = r'C:\Users\kotov-d\Documents\BASES'                                                   # path_to_bases
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

    with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\vad_preprocessed.pkl", "wb") as f:
        pickle.dump([x_train, x_test, y_train, y_test], f)




def load_iemocap():
    base_dir = r'C:\Users\kotov-d\Documents\BASES'  # path_to_bases
    features_path = os.path.join(base_dir, 'IEMOCAP', 'iemocap','feature', 'opensmile')

    with open(os.path.join(features_path, 'x_train.pkl'), 'rb') as f:
        x_train = pd.DataFrame(pickle.load(f))
    with open(os.path.join(features_path, 'x_test.pkl'), 'rb') as f:
        x_test = pd.DataFrame(pickle.load(f))
    with open(os.path.join(features_path, 'y_train.pkl'), 'rb') as f:
        y_train = pickle.load(f).loc[:, 'cur_label']
    with open(os.path.join(features_path, 'y_test.pkl'), 'rb') as f:
        y_test = pickle.load(f).loc[:, 'cur_label']

    x_train['target'] = y_train
    x_test['target'] = y_test

    x_train = x_train[x_train.target!='dis'][x_train.target!='exc'][x_train.target!='fea'][x_train.target!='fru'] \
                [x_train.target != 'hap'][x_train.target != 'oth'][x_train.target != 'sad'][x_train.target != 'sur'][x_train.target != 'xxx']
    y_train = x_train.target
    x_train.drop(columns=['target'], inplace=True)

    x_test =  x_test[x_test.target!='dis'][x_test.target!='exc'][x_test.target!='fea'][x_test.target!='fru'] \
                [x_test.target != 'hap'][x_test.target != 'oth'][x_test.target != 'sad'][x_test.target != 'sur'][x_test.target != 'xxx']
    y_test = x_test.target
    x_test.drop(columns=['target'], inplace=True)

    with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\iemocap_preprocessed.pkl", "wb") as f:
        pickle.dump([x_train, x_test, y_train, y_test], f)

def load_general(xXx):
    feature_dir = r'C:\Users\kotov-d\Documents\BASES\telecom_vad\feature'
    features_path = os.path.join(feature_dir, xXx)

    with open(os.path.join(features_path, 'x_train.pkl'), 'rb') as f:
        x_train = pd.DataFrame(pickle.load(f))
    with open(os.path.join(features_path, 'x_test.pkl'), 'rb') as f:
        x_test = pd.DataFrame(pickle.load(f))
    with open(os.path.join(features_path, 'y_train.pkl'), 'rb') as f:
        y_train = pickle.load(f).loc[:, 'cur_label']
    with open(os.path.join(features_path, 'y_test.pkl'), 'rb') as f:
        y_test = pickle.load(f).loc[:, 'cur_label']

    x_train['target'] = y_train
    x_test['target'] = y_test

    x_train = x_train[x_train.target != 'defective'][x_train.target != 'sad'][x_train.target != 'not_ informative']
    y_train = x_train.target
    x_train.drop(columns=['target'], inplace=True)

    x_test = x_test[x_test.target != 'defective'][x_test.target != 'sad'][x_test.target != 'not_ informative']
    y_test = x_test.target
    x_test.drop(columns=['target'], inplace=True)

    path_to_save_dir = r"C:\Users\kotov-d\Documents\TASKS\feature_selection\features_to_calc"
    path_to_save_file = os.path.join(path_to_save_dir, xXx + ".pkl")
    with open(path_to_save_file, "wb") as f:
        pickle.dump([x_train, x_test, y_train, y_test], f)



