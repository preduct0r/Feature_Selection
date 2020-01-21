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

print(X.shape, y.shape)
print(np.unique(y, return_counts=True))

lasso = Lasso(normalize=True, random_state=0)
lasso.fit(x_train, y_train)

# lasso-это регрессионная модель, поэтому чтобы вывести коэффициенты надо интерпретировать ее как классификатор, что не есть хорошо
print(lasso.coef_)