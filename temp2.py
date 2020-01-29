import os
import pickle
from prepare_data_with_two_labels import load_iemocap, load_general, prep_all_data
import xgboost

# for i in os.listdir(r'C:\Users\kotov-d\Documents\BASES\undefined_base\feature'):
#     load_general(i)

with open(r"C:\Users\kotov-d\Documents\TASKS\feature_selection\all_data.pkl", "rb") as f:
    [x_train, x_test, y_train, y_test] = pickle.load(f)

print(x_train.shape, x_train.columns)
print(x_train.isna().any().any())


