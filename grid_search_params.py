from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
import pickle

with open("path_to_pickle", "rb") as f:                                                             # change path to base
    [x_train, x_test, y_train, y_test] = pickle.load(f)


params = dict(Cs = [0.001, 0.01, 0.1, 1, 10],
              gammas = [0.001, 0.01, 0.1, 1])

clf = RandomizedSearchCV(SVC(), params, random_state=0)
search = clf.fit(x_train, y_train)
print(search.best_params_)