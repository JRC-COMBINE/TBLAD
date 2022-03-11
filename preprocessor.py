#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare for tsfresh analysis
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import datetime    # manipulating dates and times
from params import *


path_list = data_path
chunk_size = chunk_size
def get_data(path_list):
    data = {'id':[], 'time':[], 'value':[]}
    y = []
    ion = 0
    id = 0
    for path in path_list:
        ion = ion+1
        print("Start reading:", path)
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                print(">>> reading ",name)
                time = 0
                if name.endswith(".txt"):
                    with open(os.path.join(root,name),'r') as f:
                        # parse the ".txt" into python list
                        for line in f: 
                           
                            if time<chunk_size and time!=0:
                                data['id'].append(id)
                                data['time'].append(time)
                                data['value'].append(float(line.strip('\n')))
                                time+=1
                            elif time%chunk_size==0:
                                time = 0
                                id+=1
                                y.append(ion)
                                data['id'].append(id)
                                data['time'].append(time)
                                data['value'].append(float(line.strip('\n')))
                                time+=1
                                
                            
    df = pd.DataFrame(data)
    print(df.head())
    df.to_pickle(dataframe_path + "dataFrame_chunk"+str(chunk_size)+".pkl")
    return df, y
    
def get_label(path_list):
    idl = []
    y = []
    ion = 0
    id = 0
    for path in path_list:
        ion = ion+1
        print("Start reading:", path)
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                print(">>> reading ",name)
                time = 0
                if name.endswith(".txt"):
                    with open(os.path.join(root,name),'r') as f:
                        # parse the ".txt" into python list
                        for line in f: 
                            
                            if time<chunk_size and time!=0:
                                idl.append(id)
                                time+=1
                            elif time%chunk_size==0:
                                time = 0
                                id+=1
                                y.append(ion)
                                idl.append(id)
                                time+=1
                                
                            
    df2 = pd.Series(y, index=np.arange(1,np.max(idl)+1))
    df2.to_pickle(dataframe_path + "label_chunk"+str(chunk_size)+".pkl")
    return y


if os.path.exists(dataframe_path): 
	print("Dataframe path already exists")
else:
	try:
		os.mkdir(dataframe_path)
	except OSError:
		print ("Creation of the directory %s failed" % dataframe_path)
	else:
		print ("Successfully created the directory %s " % dataframe_path)
get_data(path_list)
get_label(path_list)
