#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 21:51:32 2024

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


from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


#------------------------------------------------------------------------
# Define XGBoost
model = XGBClassifier(objective='softmax',random_state=42)

# Create Pipeline
steps = list()
steps.append(('scaler', MinMaxScaler()))
steps.append(('classifier', model))
pipeline = Pipeline(steps=steps)

# Create Parameters
param1 = {}
param1['classifier__max_depth'] = [5, 10, 15]
# already tried 20, 50, 70 -> poor results
param1['classifier'] = [model]

param2 = {}
param2['classifier__min_child_weight'] = [5, 10, 30]
#already tried 50 100 -> poor results
param2['classifier'] = [model]

param3 = {}
param3['classifier__subsample'] = [0.6, 0.7]
#[0.8, 0.5]
param3['classifier'] = [model]

param4 = {}
param4['classifier__colsample_bytree'] = [0.8, 0.5]
param4['classifier'] = [model]

param5 = {}
param5['classifier__learning_rate'] = [0.1, 0.01, 0.001]
param5['classifier'] = [model]

param6 = {}
param6['classifier__n_estimators'] = [100, 350, 500, 650]
param6['classifier'] = [model]

### Note:
#    Want to sequentially tune the hyperparameters by:
#        GROUP 1: max_depth , min_child_weight
#        GROUP 2: subsample, colsample_bytree      
#        GROUP 3: learning_rate, num_boost_round

#       Updating the tuned parameters after every group
#       Reference: https://datascience.stackexchange.com/questions/108233/recommendations-for-tuning-xgboost-hyperparams

#--------------------------------------------------------------------------
# GridSearch for best max depth:
print(f'Searching for the best value of parameter max_depth...')
grid1 = GridSearchCV(pipeline, param1, cv=5, n_jobs=-1, scoring='f1_macro', verbose = 5).fit(all_feature, all_label)

# print the best parameters...

print(f'The best value for parameter max_depth of XGBoost is {grid1.best_params_}')
# print best score for the best model (in our case roc_auc score)
print(f'The highest F1 score is {grid1.best_score_}')
# Stats for each test - we have a total 125 tests
means1 = grid1.cv_results_['mean_test_score']
params_summary1 = grid1.cv_results_['params']

#Capture all data into a Data Frame

df1 = pd.DataFrame(list(zip(means1, params_summary1)), columns=['Mean Score', 'Parameters'])
df1.to_csv(f'GridSearchCV5_max_depth_XGBoost.csv', index=False)

#----------------------------------------------------------------------------

# GridSearch for best min_child_weight:
print(f'Searching for the best value of parameter min_child_weight...')
grid2 = GridSearchCV(pipeline, param2, cv=5, n_jobs=-1, scoring='f1_macro', verbose=4).fit(all_feature, all_label)

# print the best parameters...

print(f'The best value for parameter min_child_weight of XGBoost is {grid2.best_params_}')

print(f'The highest F1 score is {grid2.best_score_}')

means2 = grid2.cv_results_['mean_test_score']
params_summary2 = grid2.cv_results_['params']

#Capture all data into a Data Frame

df2 = pd.DataFrame(list(zip(means2, params_summary2)), columns=['Mean Score', 'Parameters'])
df2.to_csv(f'GridSearchCV5_min_child_weight_XGBoost.csv', index=False)

#---------------------------------------------------------------------------

### Sequentially tune model:
model_update = XGBClassifier(max_depth=15, min_child_weight=30, objective='softmax',random_state=42)
steps2 = list()
steps2.append(('scaler', MinMaxScaler()))
steps2.append(('classifier', model_update))
pipeline2 = Pipeline(steps=steps2)

print(pipeline2)


# GridSearch for best subsample fraction:
print(f'Searching for the best value of parameter subsample...')
grid3 = GridSearchCV(pipeline2, param3, cv=5, n_jobs=-1, scoring='f1_macro').fit(all_feature, all_label)

# print the best results

print(f'The best value for parameter subsample of XGBoost is {grid3.best_params_}')

print(f'The highest F1 score is {grid3.best_score_}')

means3 = grid3.cv_results_['mean_test_score']
params_summary3 = grid3.cv_results_['params']

#Capture all data into a Data Frame

df3 = pd.DataFrame(list(zip(means3, params_summary3)), columns=['Mean Score', 'Parameters'])
df3.to_csv(f'GridSearchCV5_subsample_XGBoost.csv', index=False)


#---------------------------------------------------------------------------

# GridSearch for best colsample_bytree:
print(f'Searching for the best value of parameter colsample_bytree...')
grid4 = GridSearchCV(pipeline2, param4, cv=5, n_jobs=-1, scoring='f1_macro').fit(all_feature, all_label)

# print the best parameters...

print(f'The best value for parameter colsample_bytree of XGBoost is {grid4.best_params_}')

print(f'The highest F1 score is {grid4.best_score_}')

means4 = grid4.cv_results_['mean_test_score']
params_summary4 = grid4.cv_results_['params']

#Capture all data into a Data Frame

df4 = pd.DataFrame(list(zip(means4, params_summary4)), columns=['Mean Score', 'Parameters'])
df4.to_csv(f'GridSearchCV5_colsample_bytree_XGBoost.csv', index=False)




#-----------------------------------------------------------------------
### Sequentially tune model:
model_update2 = XGBClassifier(subsample = 0.5, colsample_bytree = 0.5, max_depth = 15, min_child_weight = 30, objective='softmax',random_state=42)
steps3 = list()
steps3.append(('scaler', MinMaxScaler()))
steps3.append(('classifier', model_update2))
pipeline3 = Pipeline(steps=steps3)


# GridSearch for best learning rate:
print(f'Searching for the best value of parameter learning_rate...')
grid5 = GridSearchCV(pipeline3, param5, cv=5, n_jobs=-1, scoring='f1_macro').fit(all_feature, all_label)

# print the best parameters...

print(f'The best value for parameter learning_rate of XGBoost is {grid5.best_params_}')

print(f'The highest F1 score is {grid5.best_score_}')

means5 = grid5.cv_results_['mean_test_score']
params_summary5 = grid5.cv_results_['params']

#Capture all data into a Data Frame

df5 = pd.DataFrame(list(zip(means5, params_summary5)), columns=['Mean Score', 'Parameters'])
df5.to_csv(f'GridSearchCV5_learning_rate_XGBoost.csv', index=False)

#---------------------------------------------------------------------------

#Grid search for ideal num_boost_round
print(f'Searching for the best value of num_boost_round for XGBoost...')
grid6 = GridSearchCV(pipeline3, param6, cv=5, n_jobs=-1, scoring='f1_macro').fit(all_feature, all_label)

# print the best parameters...

print(f'The best value for num_boost_round of XGBoost is {grid6.best_params_}')
print(f'The highest F1 score is {grid6.best_score_}')
# Stats for each test - we have a total 125 tests
means6 = grid6.cv_results_['mean_test_score']
params_summary6 = grid6.cv_results_['params']

#Capture all data into a Data Frame

df6 = pd.DataFrame(list(zip(means6, params_summary6)), columns=['Mean Score', 'Parameters'])
df6.to_csv(f'GridSearchCV5_num_boost_round_XGBoost.csv', index=False)


#--------------------------------------------------------------------



# Best model parameters are


model_final = XGBClassifier(max_depth=15, min_child_weight=30, subsample=0.8, colsample_bytree=0.5, learning_rate=0.1, n_estimators =200, verbose=5 ,objective='softmax',random_state=42)


# Split train-test data
f_train,f_test,l_train,l_test = train_test_split(all_feature, all_label, test_size = 0.2, random_state=42, shuffle = True)

# Scaling data
scaler = MinMaxScaler()
scaler.fit(f_train)
f_train = scaler.transform(f_train)
f_test = scaler.transform(f_test)


# Fit + Train data
model_final.fit(f_train,l_train)
prediction = model_final.predict(f_test)

# Evaluate
print(classification_report(l_test,prediction))

 
