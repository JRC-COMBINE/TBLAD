#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training tsfresh
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
import datetime    
from tqdm import tqdm
import time
from params import *

## read data frame
df = pd.read_pickle(dataframe_path + "dataFrame_chunk"+str(chunk_size)+".pkl")
y  = pd.read_pickle(dataframe_path + "label_chunk"+str(chunk_size)+".pkl")
print(">>> Finished reading raw data")


extraction_settings = ComprehensiveFCParameters()

print(">>> start extracting relevant features")
t2=datetime.datetime.now()
X_filtered = extract_relevant_features(df,y, column_id ='id', column_sort = 'time', default_fc_parameters = extraction_settings)
print(X_filtered)
print(">>> Finish extracting filtered features, time cost: ",datetime.datetime.now()-t2)
print(X_filtered.info())
X_filtered.to_pickle(feature_path + "feature_chunk" + str(chunk_size) + ".pkl")
feature_list = list(X_filtered.columns)






























































