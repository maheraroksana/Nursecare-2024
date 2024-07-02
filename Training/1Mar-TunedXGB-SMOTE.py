#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 1 23:26:45 2024

@author: mahera
"""

import pandas as pd
import numpy as np

all_feature = pd.read_csv('/home/mahera/Documents/GenAI Challenge/Training/all_features.csv')
all_label = pd.read_csv('/home/mahera/Documents/GenAI Challenge/Training/all_labels.csv')

# all_feature = all_feature.to_numpy()
# print(all_feature.shape)
# print(all_label.shape)
# all_label = all_label.to_numpy()
# all_label = all_label.T.flatten()
# print(all_label.shape)

#--------------------------------------------------------------------------

# # Training

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

# Split train-test data
f_train,f_test,l_train,l_test = train_test_split(all_feature, all_label, test_size = 0.2, random_state=42, shuffle = True)

# Scale data
scaler = MinMaxScaler()
scaler.fit(f_train)
f_train = scaler.transform(f_train)
f_test = scaler.transform(f_test)

#--------------------------------------------------------------------------
# BASELINE MODEL

model_base = RandomForestClassifier(n_estimators=500,n_jobs=-1)
model_base.fit(f_train, l_train)

base_prediction = model_base.predict(f_test)

# Evaluate
print(classification_report(l_test,base_prediction))

print(set(l_test) - set(base_prediction))

#--------------------------------------------------------------------------
# BEST TUNED MODEL - XGBOOST
from xgboost import XGBClassifier

model_ml = XGBClassifier(subsample= 0.5, n_estimators= 900, min_child_weight=2, max_depth = 5, learning_rate= 0.1, colsample_bytree = 0.71)
model_ml.fit(f_train, l_train)

tuned_prediction = model_ml.predict(f_test)

# Evaluate
print(classification_report(l_test,tuned_prediction))

from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt
import seaborn as sns 

total_class = len(np.unique(l_test))

cm = confusion_matrix(l_test, tuned_prediction, labels=np.unique(l_test))
cm_norm = cm / np.sum(cm, axis=1, keepdims=True)
plt.figure(figsize=(10, 9))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', annot_kws={"fontsize":6}, yticklabels=np.arange(total_class), xticklabels=np.arange(total_class))
plt.show()



# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(model_ml, all_feature,all_label,cv=10,scoring='accuracy',n_jobs=-1)

# print(scores)
# all_label = pd.DataFrame(all_label)
# print(all_label.value_counts())
#--------------------------------------------------------------------------
# SMOTE
from imblearn.over_sampling import SMOTE

f_smote_tr, f_smote_te, l_smote_tr, l_smote_te = train_test_split(all_feature, all_label, test_size=0.2, stratify=all_label)


f_smote_tr, l_smote_tr = SMOTE().fit_resample(f_smote_tr, l_smote_tr)

print(all_label.value_counts())
print(l_smote_tr.value_counts())

# Scale data
scaler.fit(f_smote_tr)
f_smote_tr = scaler.transform(f_smote_tr)
f_smote_te = scaler.transform(f_smote_te)


# BEST TUNED MODEL on SMOTE DATA - XGBOOST

model_ml = XGBClassifier(subsample= 0.5, n_estimators= 900, min_child_weight=2, max_depth = 5, learning_rate= 0.1, colsample_bytree = 0.71)
model_ml.fit(f_smote_tr, l_smote_tr)

smote_prediction = model_ml.predict(f_smote_te)

# Evaluate
print(classification_report(l_smote_te,smote_prediction))

total_class = len(np.unique(l_smote_te))

cm_smote = confusion_matrix(l_smote_te, smote_prediction, labels=np.unique(l_smote_te))
cm_smote_norm = cm_smote / np.sum(cm_smote, axis=1, keepdims=True)
plt.figure(figsize=(10, 9))
sns.heatmap(cm_smote_norm, annot=True, fmt='.2f', cmap='Blues', annot_kws={"fontsize":6}, yticklabels=np.arange(total_class), xticklabels=np.arange(total_class))
plt.show()

import pickle

pickle.dump(model_ml,open('XGBoost-SMOTE.pkl','wb'))