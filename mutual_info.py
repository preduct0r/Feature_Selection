import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import mutual_info_score
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\vad_preprocessed.pkl", "rb") as f:                                                             # change path to base
    [x_train, x_test, y_train, y_test] = pickle.load(f)

X = x_train.append(x_test, ignore_index=True)
y = y_train.append(y_test)

# считаем топ 500 фичей, зависимых с таргетом
mutual_info = pd.DataFrame(columns=['feature', 'mi'], index=list(x_train.columns))
mutual_info['feature'] = list(x_train.columns)
for idx, row in mutual_info.iterrows():
    mutual_info.loc[idx, 'mi'] = mutual_info_score(X.loc[:,int(row[0])], y)

mutual_info.to_csv(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\mutual_info.csv", index=False)

mutual_info = pd.read_csv(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\mutual_info.csv")

mutual_info = mutual_info['mi'].sort_values(ascending=False)
mutual_info.reset_index(drop=True).plot()

top_mi_features = list(mutual_info.index[:1600])
with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\top_mi_features.pkl","wb") as f:
    pickle.dump(top_mi_features, f)

plt.show()