# last updated: 03/29/2014
# "Plus L take away R" (+L -R)

import time
from copy import deepcopy
from lightgbm import LGBMClassifier



clf = LGBMClassifier(subsample=1.0, n_estimators=1200, min_split_gain=0, max_depth=8, learning_rate=0.01,
                     colsample_bytree=0.7, num_leaves=65)



def plus_L_minus_R(features, max_k, criterion_func, L=5, R=1, print_steps=True):
    """
    Implementation of a "Plus l take away r" algorithm.

    Keyword Arguments:
        features (list): The feature space as a list of features.
        max_k: Termination criterion; the size of the returned feature subset.
        criterion_func (function): Function that is used to evaluate the
            performance of the feature subset.
        L (int): Number of features added per iteration.
        R (int): Number of features removed per iteration.
        print_steps (bool): Prints the algorithm procedure if True.

    Returns the selected feature subset, a list of features of length max_k.

    """
    with open(r'C:\Users\kotov-d\Documents\TASKS\feature_selection\gridy_alg_selected_features.txt','w') as f:
        f.write('')

    assert (L != R), 'L must be != R to avoid an infinite loop'

    ############################
    ### +L -R for case L > R ###
    ############################


    feat_sub = []
    k = 0

    # Initialization
    while True:

        for i in range(L):
            if len(features) > 0:
                crit_func_max = criterion_func(feat_sub + [features[0]])
                best_feat = features[0]
                if len(features) > 1:
                    for x in features[1:]:
                        crit_func_eval = criterion_func(feat_sub + [x])
                        if crit_func_eval > crit_func_max:
                            crit_func_max = crit_func_eval
                            best_feat = x
                features.remove(best_feat)
                feat_sub.append(best_feat)
                if print_steps:
                    print('include: {}'.format(best_feat))

                with open(r'C:\Users\kotov-d\Documents\TASKS\feature_selection\gridy_alg_selected_features.txt',
                          'a') as f:
                    f.write(str(k) + ' L ' + str(feat_sub) + '\n')

        # -R (Exclusion)
        if len(features) + len(feat_sub) > max_k:
            worst_feat = len(feat_sub) - 1
            worst_feat_val = feat_sub[worst_feat]
            crit_func_max = criterion_func(feat_sub[:-1])

            for j in reversed(range(0, len(feat_sub) - 1)):
                crit_func_eval = criterion_func(feat_sub[:j] + feat_sub[j + 1:])
                if crit_func_eval > crit_func_max:
                    worst_feat, crit_func_max = j, crit_func_eval
                    worst_feat_val = feat_sub[worst_feat]
            del feat_sub[worst_feat]
            features.append(worst_feat_val)
            if print_steps:
                print('exclude: {}'.format(worst_feat_val))

            with open(r'C:\Users\kotov-d\Documents\TASKS\feature_selection\gridy_alg_selected_features.txt',
                      'a') as f:
                f.write(str(k)+ ' R '+ str(feat_sub) + '\n')

        # Termination condition
        k = len(feat_sub)
        if k == max_k:
            break

    return feat_sub



# Sequential Forward Selection (SFS)
# код взят здесь
# https://nbviewer.jupyter.org/github/rasbt/pattern_classification/blob/master/dimensionality_reduction/feature_selection/sequential_selection_algorithms.ipynb
def seq_forw_select(features, max_k, criterion_func, print_steps=True):
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
    with open(r'C:\Users\kotov-d\Documents\TASKS\feature_selection\gridy_alg_selected_features.txt', 'w') as f:
        f.write('')

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
            if crit_func_eval > crit_func_max:
                crit_func_max = crit_func_eval
                best_feat = x
        feat_sub.append(best_feat)
        if print_steps:
            print('include: {}'.format(best_feat))

        with open(r'C:\Users\kotov-d\Documents\TASKS\feature_selection\gridy_alg_selected_features.txt', 'a') as f:
            f.write(str(loop)+' '+str(feat_sub)+'\n')

        features.remove(best_feat)

        # Termination condition
        k = len(feat_sub)
        if k == max_k:
            break

        loop+=1

    print("time of execution :", round(time.time()-start_time, 3))
    return feat_sub
