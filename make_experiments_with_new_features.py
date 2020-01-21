import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report, precision_score
from prepare_data_with_two_labels import load_iemocap

# # достаем train, test данные из telecom-vad и best_indexes
# with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\vad_preprocessed.pkl", "rb") as f:
#     [x_train, x_test, y_train, y_test] = pickle.load(f)

# попытка использовать IEMOCAP
with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\iemocap_preprocessed.pkl", "rb") as f:
    [x_train, x_test, y_train, y_test] = pickle.load(f)

with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\best_indexes.pkl", "rb") as f:
    best_indexes = pickle.load(f)


# проверка с помощью обучения lightgbm на разных фичах
cur_pred = 0
def select_features(x_train, y_train, x_test, y_test):
    clf = LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.001, n_estimators=1000,
                             objective=None, min_split_gain=0, min_child_weight=3, min_child_samples=10, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=0, seed=17)
    clf.fit(x_train, y_train)

    print(len(clf.feature_importances_))

    dict_importance = {}
    for feature, importance in zip(x_train.columns, clf.feature_importances_):
        dict_importance[feature] = importance

    best_lgbm_features = []

    for idx, w in enumerate(sorted(dict_importance, key=dict_importance.get, reverse=True)):
        if idx == 100:
            break
        best_lgbm_features.append(w)

    clf.fit(x_train.loc[:,best_lgbm_features], y_train)

    pred = clf.predict(x_test.loc[:,best_lgbm_features])

    print('ground truth values: ', np.unique(y_test, return_counts=True))
    print('recieved values: ', np.unique(pred, return_counts=True))
    print(round(f1_score(y_test, pred, average='macro'),3))

                                                        # у нас 2200 фичей и 600 примеров, конечно все пойдет в нейтраль!!
select_features(x_train, y_train, x_test, y_test)
select_features(x_train.iloc[:,best_indexes], y_train, x_test.iloc[:,best_indexes], y_test)




# # вывод предсказанных и реальных значений
# print(clf.classes_.tolist()+['pred','y'])
# print(pd.DataFrame(data=np.hstack((clf.predict_proba(x_test), clf.predict(x_test).reshape(-1,1),
#                                   y_test.values.reshape(-1,1))), columns=clf.classes_.tolist()+['pred','y']))



# ====================================================================================================================
# ====================================================================================================================
# # проверка полученных результатов с помощью усреднения значений ячеек таблицы корреляции
# best_matrix = corr_matrix[corr_matrix.index.isin(best_indexes)]
# print((best_matrix.sum().sum()-best_matrix.shape[0])/(best_matrix.shape[0]**2-best_matrix.shape[0]))
# print((corr_matrix.sum().sum()-corr_matrix.shape[0])/(corr_matrix.shape[0]**2-corr_matrix.shape[0]))



# # проверка с помощью обучения SVM на разных фичах
# from sklearn.svm import SVC
# clf=SVC(gamma=0.1, C=10)
# pred = clf.fit(x_train.iloc[:,best_indexes], y_train).predict(x_test.iloc[:,best_indexes])
# print("на фичах полученных при помощи корреляции f1 macro {}".format(round(f1_score(y_test, pred, average='macro'),3)))
#
# # pred = clf.fit(x_train[:,best_features], y_train).predict(x_test[:,best_features])
# # print("на фичах полученных при помощи feature_importances f1 macro {}".format(round(f1_score(y_test, pred, average='macro'),3)))
#
# pred = clf.fit(x_train, y_train).predict(x_test)
# print("на оригинальных фичах f1 macro {}".format(round(f1_score(y_test, pred, average='macro'),3)))



# возможно прегодится
# ====================================================================================================================
# ====================================================================================================================

# # смотрим сколько значений корреляции выше трешхолда
# threshold = 0.9
# print((corr_matrix[corr_matrix>threshold].count().sum()-corr_matrix.shape[0])//2, "корреляций")
# print(round((corr_matrix[corr_matrix>threshold].count().sum()-2269)/(2269**2)*100, 3), "%")



# # отрисовка дендограммы
# Z = linkage(corr_matrix, 'single')
# plt.figure(figsize=(25, 25))
# dn = dendrogram(Z)
# plt.show()
