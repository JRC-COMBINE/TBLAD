#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training Tree-based classifiers
"""
from params import *
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
from operator import add
import csv

# import the model we use
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

# define classification model "RF", "XGB" or "ET"
print(">>> Please select one of the following classifiers:")
print(">>> 1. Random Forest")
print(">>> 2. XGBoost")
print(">>> 3. Extra-Trees")
classifiers = ['RF', 'XGB', 'ET']
value = input(">>> Please enter an integer between 1 and 3:\n")
Model = int(value)


## read true labels
y  = pd.read_pickle(dataframe_path + "label_chunk"+str(chunk_size)+".pkl")
print(">>> Finished reading raw data")

## read trained tsfresh features
X_filtered = pd.read_pickle(feature_path + "feature_chunk" + str(chunk_size) + ".pkl")
feature_list = list(X_filtered.columns)
accuracy = []

## define number of forests
N = N_forests
## define accumulated list of importance
importance_acc = [0]*len(feature_list)
importance_matrix  = np.zeros((N,len(feature_list)))

## train m random forests
for m in tqdm(range(N)):
    ## Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(X_filtered, y, test_size = 0.2)
    if m==0:
        print('Training Features Shape:', train_features.shape)
        print('Training Labels Shape:', train_labels.shape)
        print('Testing Features Shape:', test_features.shape)
        print('Testing Labels Shape:', test_labels.shape)
        print(">>> Start training"+ classifiers[Model-1])

    ## Instantiate model with 20 decision trees
    ## Define classification model
    
    if Model == 1:
        rf = RandomForestClassifier(n_estimators = N_estimators)
    elif Model == 2:
        rf = XGBClassifier()
    elif Model == 3:
        rf = ExtraTreesClassifier(n_estimators = N_estimators)

    # Train the model on training data
    rf.fit(train_features, train_labels);
    
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)

    # Calculate accuracy = #Hit / (test set size) *100%
    # Convert test_labels from pandas Series to numpy
    test_labels = np.array(test_labels)
    hit = 0
    for i in np.arange(0,test_labels.size):
        if test_labels[i]==predictions[i]: hit+=1
    accuracy.append(hit/len(test_labels)*100)
    
    # Get numerical feature importances
    importance_matrix[m][:] = rf.feature_importances_[:]
    importances = list(rf.feature_importances_)
    importance_acc = list(map(add, importances, importance_acc))
    if m==N-1:

        # After the final forest, average the accumulated importance list by N
        importance_acc =  [x / N for x in importance_acc]
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 5)) for feature, importance in zip(feature_list, importance_acc)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        #[print('Variable: {:} Importance: {}'.format(*pair)) for pair in feature_importances if pair[1]>0.01]
    time.sleep(0.01)

# Save as csv file
csvfile = feature_path +"feature_importance_chunk" + str(chunk_size)+ "_" + classifiers[Model-1] + ".csv"
with open(csvfile, "w") as output:
  writer = csv.writer(output, lineterminator='\n')
  for line in feature_importances:
    feature = line[0]
    value = line[1]
    writer.writerow([feature, value])
print("The accuracy of prediction is: {}%".format(np.mean(accuracy)))
print("The accuracy std is: {}%".format(np.std(accuracy)))

## Create pie chart
reduced_importances = [x for x in feature_importances if x[1]>0.01]
print("reduced feature importances:", reduced_importances)
data = [x[1] for x in reduced_importances]
label = [x[0] for x in reduced_importances]
plt.rcdefaults()
fig, ax = plt.subplots()
y_pos = np.arange(len(label))
ax.barh(y_pos, data , align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(label)
ax.invert_yaxis()  
ax.set_xlabel('Importance Value')
plt.show()

## Correlation analysis for top 10 features
d = {}
for pair in feature_importances:
    if pair[1]>0.01:
    	name = pair[0]

    	# Find index of feature by "name"
    	index = feature_list.index(name)
    	feature_array = importance_matrix[:,index]

    	# Append feature to the dictionary d 
    	d[name] = feature_array
df = pd.DataFrame(data = d)
corr = df.corr()
matrix = np.triu(corr,k=1)
sns.heatmap(corr, mask=matrix, annot=True, cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True)

plt.show()

#'Visualization of one tree'
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')
