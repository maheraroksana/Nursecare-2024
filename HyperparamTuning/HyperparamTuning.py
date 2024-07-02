#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 15:07:03 2024

@author: mahera
"""


import pandas as pd
import numpy as np

all_feature = pd.read_csv('/home/mahera/Documents/GenAI Challenge/Training/all_features.csv')
all_label = pd.read_csv('/home/mahera/Documents/GenAI Challenge/Training/all_labels.csv')



#Import all the models that you want to consider (include in the Grid search)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

from sklearn.model_selection import GridSearchCV

#Import other useful libraries
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler



# Define parameters for Random Forest 
param1 = {}
param1['classifier__n_estimators'] = [10, 50, 100, 250]
param1['classifier__max_depth'] = [5, 10, 20]
param1['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
param1['classifier'] = [RandomForestClassifier(random_state=42)]
#Total 48 parameters to test (4 * 3 * 4)

# Define parameters for support vector machine (SVC)
param2 = {}
param2['classifier__C'] = [10**-2, 10**-1, 10**0, 10**1, 10**2]
param2['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
param2['classifier'] = [SVC(random_state=42)]
#Total 20 parameters to test (5 * 4)

# Define parameters for Logistic regression
param3 = {}
param3['classifier__C'] = [10**-2, 10**-1, 10**0, 10**1, 10**2]
param3['classifier__penalty'] = ['l1', 'l2']
param3['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
param3['classifier'] = [LogisticRegression(random_state=42)]
#Total 40 parameters to test (5 * 2 * 4)

# Define parameters for K neighbors
param4 = {}
param4['classifier__n_neighbors'] = [2,5,10,25,50]
param4['classifier'] = [KNeighborsClassifier()]
#Total 5 parameters to test (5)

# Define parameters for Gradient boosting



param5 = {}
param5['classifier__learning_rate'] = [0.01, 0.1, 0.25] 
param5['classifier__max_features'] = [10, 50, 100]
param5['classifier__n_estimators'] = [10, 50, 100, 250]
param5['classifier__max_depth'] = [3, 5, 10]
param5['classifier'] = [xgb.XGBClassifier(random_state=42)]
#Total 12 parameters to test (4 * 3)

# define the pipeline to include scaling and the model. 
# Prepare the pipeline for the 1st model, others will be loaded appropriately
#during the Grid Search
#This pipeline will be the input to cross_val_score, instead of the model. 




#Capture all parameter dictionaries as a list
params = {'RFC' : param1, 
          'SVC' : param2, 
          'LogReg' : param3, 
          'KNN' : param4, 
          'XGBoost' : param5}
classifiers = {'RFC' : RandomForestClassifier(random_state=42),
               'SVC' : SVC(random_state=42),
               'LogReg' : LogisticRegression(random_state=42),
               'KNN' : KNeighborsClassifier(),
               'XGBoost' : xgb.XGBClassifier(random_state=42)
                   }


# print(params['RFC'].items())

for i in params['RFC'].items():
    print (f'bla {i}')



for model, model_inst in classifiers.items():
    steps = list()
    steps.append(('scaler', MinMaxScaler()))
    steps.append(('classifier', model_inst))
    pipeline = Pipeline(steps=steps)
    #Grid search - including cross validation
    param_set = params[model]
    print(f'Searching for the best model for {model}...')
    for i in param_set.items():
        grid = GridSearchCV(pipeline, i, cv=5, n_jobs=-1, scoring='f1_score').fit(all_feature, all_label)
    
        #Gridsearch object (in our case 'grid') stores all the information about
        #the best model and corresponding hyperparameters. 
        # print the best parameters...
        print(f'The best value for parameter {i[0]} of {model}  is {grid.best_params_}')
        # print best score for the best model (in our case roc_auc score)
        print(f'The highest F1 score is {grid.best_score_}')
        # Stats for each test - we have a total 125 tests
        means = grid.cv_results_['mean_test_score']
        params_summary = grid.cv_results_['params']
        
        #Capture all data into a Data Frame
    
        df = pd.DataFrame(list(zip(means, params_summary)), columns=['Mean Score', 'Parameters'])
        df.to_csv(f'GridSearch_{i[0]}_{model}.csv')


