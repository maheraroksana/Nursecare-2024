#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 17:51:10 2024

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



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

#------------------------------------------------------------------------

# Define Logistic regression
model = LogisticRegression(solver='sag', max_iter=200, random_state=42)

# Create Pipeline
steps = list()
steps.append(('scaler', MinMaxScaler()))
steps.append(('classifier', model))
pipeline = Pipeline(steps=steps)

# Define parameters for Logistic regression
param1 = {}
param1['classifier__C'] = [10**-2, 10**-1, 10**0, 10**1, 10**2]
param1['classifier'] = [LogisticRegression(random_state=42)]

# param2 = {}
# param2['classifier__penalty'] = ['l1', 'l2']
# param2['classifier'] = [LogisticRegression(random_state=42)]

param3 = {}
param3['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
param3['classifier'] = [LogisticRegression(random_state=42)]
#Total 40 parameters to test (5 * 2 * 4)


#--------------------------------------------------------------------------

#Grid search for ideal C
print(f'Searching for the best value of parameter for {model}...')
grid = GridSearchCV(pipeline, param1, cv=5, n_jobs=-1, scoring='f1_macro').fit(all_feature, all_label)

# print the best parameters...

print(f'The best value for parameter C of LogReg is {grid.best_params_}')
print(f'The highest F1 score is {grid.best_score_}')
# Stats for each test - we have a total 125 tests
means = grid.cv_results_['mean_test_score']
params_summary = grid.cv_results_['params']

#Capture all data into a Data Frame

df = pd.DataFrame(list(zip(means, params_summary)), columns=['Mean Score', 'Parameters'])
df.to_csv(f'GridSearchCV5_C_LogReg.csv', index=False)


#--------------------------------------------------------------------

# # GridSearch for best max depth:
# print(f'Searching for the best value of parameter penalty...')
# grid2 = GridSearchCV(pipeline, param2, cv=5, n_jobs=-1, scoring='f1_macro').fit(all_feature, all_label)

# # print the best parameters...

# print(f'The best value for parameter penalty of LogReg is {grid2.best_params_}')
# # print best score for the best model (in our case roc_auc score)
# print(f'The highest F1 score is {grid2.best_score_}')
# # Stats for each test - we have a total 125 tests
# means2 = grid2.cv_results_['mean_test_score']
# params_summary2 = grid2.cv_results_['params']

# #Capture all data into a Data Frame

# df2 = pd.DataFrame(list(zip(means2, params_summary2)), columns=['Mean Score', 'Parameters'])
# df2.to_csv(f'GridSearchCV5_penalty_LogReg.csv', index=False)


#-----------------------------------------------------------------------

# GridSearch for best class weight:
print(f'Searching for the best value of parameter class_weight...')
grid3 = GridSearchCV(pipeline, param3, cv=5, n_jobs=-1, scoring='f1_macro').fit(all_feature, all_label)

# print the best results:

print(f'The best value for parameter class_weight of LogReg is {grid.best_params_}')
# print best score for the best model (in our case roc_auc score)
print(f'The highest F1 score is {grid.best_score_}')
# Stats for each test - we have a total 125 tests
means3 = grid3.cv_results_['mean_test_score']
params_summary3 = grid3.cv_results_['params']

#Capture all data into a Data Frame

df3 = pd.DataFrame(list(zip(means3, params_summary3)), columns=['Mean Score', 'Parameters'])
df3.to_csv(f'GridSearchCV5_class_weight_LogReg.csv', index=False)

#----------------------------------------------------------------------------
# Best model parameters are C = 10, class_weight = None
# Best model does not converge!


model_final = LogisticRegression(C=10, class_weight=None,solver='saga', max_iter=500, random_state=42)


# Split train-test data
f_train,f_test,l_train,l_test = train_test_split(all_feature, all_label, test_size = 0.2, random_state=42, shuffle = True)

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
