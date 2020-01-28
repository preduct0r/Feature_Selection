import os
import pickle
from prepare_data_with_two_labels import load_iemocap, load_general,  prep_all_data



# for i in os.listdir(r'C:\Users\kotov-d\Documents\BASES\undefined_base\feature'):
#     load_general(str(i))


prep_all_data()
