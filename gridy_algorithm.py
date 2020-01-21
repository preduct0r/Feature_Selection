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


with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\vad_preprocessed.pkl", "rb") as f:                                                             # change path to base
    [x_train, x_test, y_train, y_test] = pickle.load(f)

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

    while True:

        # Inclusion step
        if print_steps:
            print('\nInclusion from feature space', features)
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
            print('include: {} -> feature subset: {}'.format(best_feat, feat_sub))
        features.remove(best_feat)

        # Termination condition
        k = len(feat_sub)
        if k == max_k:
            break

    return feat_sub

def criterion_func(features, x_train, x_test, y_train, y_test):
    clf = SVC(gamma=0.1, C=10)
    clf.fit(X.loc[:,features], y)

    pred = clf.predict(x_test)
    print(np.unique(pred, return_counts=True))

    clf.fit(x_train, y_train)
    score = f1_score(pred,y_test)
    print(score)

    return score

seq_forw_select(list(x_train.columns), 10, criterion_func, print_steps=True)