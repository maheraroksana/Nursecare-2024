#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 17:46:34 2024

@author: mahera
"""


import pandas as pd
import numpy as np

all_feature = pd.read_csv('/home/mahera/Documents/GenAI Challenge/Training/all_features.csv')
all_label = pd.read_csv('/home/mahera/Documents/GenAI Challenge/Training/all_labels.csv')

all_feature = all_feature.to_numpy()
print(all_feature.shape)
print(all_label.shape)
all_label = all_label.to_numpy()
all_label = all_label.T.flatten()
print(all_label.shape)

from sklearn.svm import SVC


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

#-------------------------------------------------------------------------


# Define Support Vector Machine
model = SVC(random_state=42)

# Create Pipeline
steps = list()
steps.append(('scaler', MinMaxScaler()))
steps.append(('classifier', model))
pipeline = Pipeline(steps=steps)

# Define Parameters
param1 = {}
param1['classifier__C'] = [10**-2, 10**-1, 10**0, 10**1, 10**2]
param1['classifier'] = [SVC(random_state=42)]

param2 = {}
param2['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
param2['classifier'] = [SVC(random_state=42)]

#----------------------------------------------------------------------------


#Grid search for ideal reg factor C:
print(f'Searching for the best value of parameter for {model}...')
grid = GridSearchCV(pipeline, param1, cv=5, n_jobs=-1, scoring='f1_macro').fit(all_feature, all_label)

# print the best results

print(f'The best value for parameter n_estimators of SVC is {grid.best_params_}')
print(f'The highest F1 score is {grid.best_score_}')
# Stats for each test - we have a total 125 tests
means = grid.cv_results_['mean_test_score']
params_summary = grid.cv_results_['params']

# Capture all data into a Data Frame

df = pd.DataFrame(list(zip(means, params_summary)), columns=['Mean Score', 'Parameters'])
df.to_csv(f'GridSearchCV5_n_estimators_SVC.csv', index=False)


#--------------------------------------------------------------------

# Create Pipeline
steps = list()
steps.append(('scaler', MinMaxScaler()))
steps.append(('classifier', model))
pipeline = Pipeline(steps=steps)

# GridSearch for best class_weights:
print(f'Searching for the best value of parameter max_depth...')
grid2 = GridSearchCV(pipeline, param2, cv=5, n_jobs=-1, scoring='f1_macro').fit(all_feature, all_label)

# print the best parameters...

print(f'The best value for parameter max_depth of SVC is {grid2.best_params_}')
# print best score for the best model (in our case roc_auc score)
print(f'The highest F1 score is {grid2.best_score_}')
# Stats for each test - we have a total 125 tests
means2 = grid2.cv_results_['mean_test_score']
params_summary2 = grid2.cv_results_['params']

#Capture all data into a Data Frame

df2 = pd.DataFrame(list(zip(means2, params_summary2)), columns=['Mean Score', 'Parameters'])
df2.to_csv(f'GridSearchCV5_class_weight_SVC.csv', index=False)


#-----------------------------------------------------------------------

# Best model parameters are C = 100, class_weights = {0:1,1:5}


model_final = SVC(kernel = 'linear',C=100,class_weight={0:1,1:5},random_state=11)


# Split train-test data
f_train,f_test,l_train,l_test = train_test_split(all_feature, all_label, test_size = 0.2, random_state=11, shuffle = True)

# Scale data
scaler = MinMaxScaler()
scaler.fit(f_train)
f_train = scaler.transform(f_train)
f_test = scaler.transform(f_test)


# Fit + Train data
model_final.fit(f_train,l_train)
prediction = model_final.predict(f_test)

# Evaluate
print(classification_report(l_test,prediction))



