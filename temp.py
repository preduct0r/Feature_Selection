import os
import pickle
from prepare_data_with_two_labels import load_iemocap, load_general



for i in os.listdir(r'C:\Users\kotov-d\Documents\BASES\telecom_vad\feature'):
    load_general(str(i))