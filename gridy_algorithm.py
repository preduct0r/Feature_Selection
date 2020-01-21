import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, classification_report, precision_score

from sklearn.svm import SVC
import pickle
import pandas as pd
import numpy as np
import time

# # загружаем telecom-vad
# with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\vad_preprocessed.pkl", "rb") as f:                                                             # change path to base
#     [x_train, x_test, y_train, y_test] = pickle.load(f)

# попытка использовать IEMOCAP
with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\iemocap_preprocessed.pkl", "rb") as f:
    [x_train, x_test, y_train, y_test] = pickle.load(f)


with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\logreg_features.pkl","rb") as f:
    logreg_features = pickle.load(f)


X = x_train.append(x_test, ignore_index=True)
y = y_train.append(y_test)



# Sequential Forward Selection (SFS)
# код взят здесь
# https://nbviewer.jupyter.org/github/rasbt/pattern_classification/blob/master/dimensionality_reduction/feature_selection/sequential_selection_algorithms.ipynb
def seq_forw_select(features, max_k, criterion_func, print_steps=False):
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
        time_per_loop = []
        scores_per_loop = []
        distrib_per_loop = []

        # Inclusion step
        if print_steps:
            print('\nInclusion from feature space', features)
        crit_func_max = criterion_func(feat_sub + [features[0]], [time_per_loop, scores_per_loop, distrib_per_loop])
        best_feat = features[0]
        for x in features[1:]:
            crit_func_eval = criterion_func(feat_sub + [x], [time_per_loop, scores_per_loop, distrib_per_loop])
            print()
            if crit_func_eval > crit_func_max:
                crit_func_max = crit_func_eval
                best_feat = x
        feat_sub.append(best_feat)
        if print_steps:
            print('include: {} -> feature subset: {}'.format(best_feat, feat_sub))
        features.remove(best_feat)

        # Termination condition
        k = len(feat_sub)
        if k == max_k:
            break

        with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\gridy_info\gridy_info_{}.pkl".format(loop), "wb") as f:
            pickle.dump([time_per_loop, scores_per_loop, distrib_per_loop], f)
        loop+=1



    return feat_sub


from lightgbm import LGBMClassifier
clf = LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.001, n_estimators=1000,
                             objective=None, min_split_gain=0, min_child_weight=3, min_child_samples=10, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=0, seed=17)


def criterion_func(features, check_info):
    time_per_loop, scores_per_loop, distrib_per_loop = check_info
    # clf = SVC(gamma=0.1, C=10)
    start_time = time.time()

    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)


    score = f1_score(pred,y_test, average='macro')
    exec_time = time.time()-start_time


    print(np.unique(pred, return_counts=True))
    print("score: ", round(score,3))
    print("time of execution :", round(exec_time,3))


    time_per_loop.append(exec_time)
    scores_per_loop.append(score)
    distrib_per_loop.append(np.unique(pred, return_counts=True))

    return score

seq_forw_select(logreg_features, 10, criterion_func, print_steps=True)
