#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 23:05:56 2024

@author: mahera
"""

import pandas as pd
import numpy as np

f_tr = pd.read_csv('/home/mahera/Documents/GenAI Challenge/Training/all_features.csv')
l_tr = pd.read_csv('/home/mahera/Documents/GenAI Challenge/Training/all_labels.csv')

from sklearn.preprocessing import MinMaxScaler

#--------------------------------------------------------------------------
# SMOTE
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

f_tr, l_tr = SMOTE().fit_resample(f_tr, l_tr)


# Scale data
scaler = MinMaxScaler()
scaler.fit(f_tr)
f_tr = scaler.transform(f_tr)


# BEST TUNED MODEL on SMOTE DATA - XGBOOST

model_xgbsmote = XGBClassifier(subsample= 0.5, n_estimators= 900, min_child_weight=2, max_depth = 5, learning_rate= 0.1, colsample_bytree = 0.71)
model_xgbsmote.fit(f_tr, l_tr)

import pickle


with open('/home/mahera/Documents/GenAI Challenge/Training/XGB_model.pkl','wb') as handle:
    pickle.dump(model_xgbsmote,handle)
