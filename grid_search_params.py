from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
import pickle

with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\vad_preprocessed.pkl", "rb") as f:                                                             # change path to base
    [x_train, x_test, y_train, y_test] = pickle.load(f)

X = x_train.append(x_test, ignore_index=True)
y = y_train.append(y_test)

params = dict(C = [0.001, 0.01, 0.1, 1, 10],
              gamma = [0.001, 0.01, 0.1, 1])

clf = RandomizedSearchCV(SVC(), params, random_state=0)
clf.fit(X, y)
print(clf.best_params_)