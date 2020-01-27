from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
import pickle
from lightgbm import LGBMClassifier
import numpy as np

with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\vad_preprocessed.pkl", "rb") as f:                                                             # change path to base
    [x_train, x_test, y_train, y_test] = pickle.load(f)

X = x_train.append(x_test, ignore_index=True)
y = y_train.append(y_test)

params = dict(n_estimators=np.linspace(100, 1500, 10, dtype=int),
              learning_rate=np.logspace(-4, -2, num=10),
              colsample_bytree=np.linspace(0.5, 1.0, 5),
              subsample=np.linspace(0.5, 1.0, 3),
              max_depth = [2,4,6,8,10],
              min_split_gain=[0])

clf = RandomizedSearchCV(LGBMClassifier(), params, n_iter=300, scoring='f1_macro', verbose=2)
clf.fit(X, y)
print(clf.best_params_)