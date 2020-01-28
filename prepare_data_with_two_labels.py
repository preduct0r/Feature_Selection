import sys, os
import pickle
import numpy as np
import pandas as pd

def prep_all_data():
    main_path = r'C:\Users\kotov-d\Documents\TASKS\feature_selection\small_features'
    for idx,i in enumerate(os.listdir(main_path)):
        feature_path = os.path.join(main_path, i)
        if idx==0:
            with open(feature_path, "rb") as f:
                [x_train, x_test, y_train, y_test] = pickle.load(f)
                x_train.columns = [i[2:-4]+'_'+str(x) for x in x_train.columns]
                x_test.columns = [i[2:-4]+'_'+str(x) for x in x_test.columns]

        else:
            with open(feature_path, "rb") as f:
                [x_train_, x_test_, y_train_, y_test_] = pickle.load(f)
                x_train_.columns = [i[2:-4] + '_' + str(x) for x in x_train_.columns]
                x_test_.columns = [i[2:-4] + '_' + str(x) for x in x_test_.columns]
            x_train = pd.concat([x_train, x_train_], axis=1)
            x_test = pd.concat([x_test, x_test_], axis=1)
    with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\all_data.pkl", "wb") as f:
        pickle.dump([x_train, x_test, y_train, y_test], f)







def load_vad():
    base_dir = r'C:\Users\kotov-d\Documents\BASES'                                                   # path_to_bases
    features_path = os.path.join(base_dir, 'undefined_base', 'feature', 'opensmile')


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
    feature_dir = r'C:\Users\kotov-d\Documents\BASES\undefined_base\feature'
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

    path_to_save_dir = r"C:\Users\kotov-d\Documents\TASKS\feature_selection\small_features"
    path_to_save_file = os.path.join(path_to_save_dir, xXx + ".pkl")
    with open(path_to_save_file, "wb") as f:
        pickle.dump([x_train, x_test, y_train, y_test], f)



