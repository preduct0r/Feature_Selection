import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, classification_report, precision_score
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from sklearn.svm import SVC
import pickle
import pandas as pd
import numpy as np

with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\top_mi_features.pkl","rb") as f:
    top_mi_features = pickle.load(f)

with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\vad_preprocessed.pkl", "rb") as f:                                                             # change path to base
    [x_train, x_test, y_train, y_test] = pickle.load(f)

X = x_train.append(x_test, ignore_index=True)
y = y_train.append(y_test)

X = X.loc[:, top_mi_features]

print(X.shape, y.shape)
print(np.unique(y, return_counts=True))

clf = LogisticRegression(penalty='l1', random_state=0)
clf.fit(X, y)

# lasso-это регрессионная модель, поэтому чтобы вывести коэффициенты надо интерпретировать ее как классификатор, что не есть хорошо
# pd.Series(clf.coef_[0]).plot()
# # plt.show()

df = pd.DataFrame(columns=['feature','coef'], index = range(len(top_mi_features)))
df['feature'] = top_mi_features
df['coef'] = clf.coef_[0]


with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\logreg_features.pkl","wb") as f:
    pickle.dump(list(df[df.coef>0].feature), f)
