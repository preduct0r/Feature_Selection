import os
import pickle
from prepare_data_with_two_labels import load_iemocap, load_general, prep_all_data
import xgboost

[x_train, x_test, y_train, y_test] =  prep_all_data()
