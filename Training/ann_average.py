#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 17:19:33 2024

@author: mahera
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DATA_FOLDER = '/home/mahera/Documents/GenAI Challenge/published_data/ann/'


TRAIN_ID = ["N01T1", "N01T2",
            "N02T1", "N02T2",
            "N04T1", "N04T2",
            "N06T1", "N06T2",
            "N07T1", "N07T2",
            "N11T1", "N11T2",
            "N12T1", "N12T2",
            "S01T1", "S01T2",
            "S02T1", "S02T2",
            "S03T1", "S03T2",
            "S05T1", "S05T2",
            "S07T1", "S07T2",
            "S08T1", "S08T2",
            "S09T1", "S09T2",
            "S10T1", "S10T2",
            "S11T1", "S11T2",]



all_time = np.zeros(12)
all_tag_counts = np.zeros(12)

for user_id in TRAIN_ID:
    path = DATA_FOLDER + user_id + '_ann.csv'
    ann_df = pd.read_csv(path)
    
    for i in range(len(ann_df)):
        start = ann_df['start_time'][i]
        stop = ann_df['stop_time'][i]
        diff = stop - start
        tag = int(ann_df['annotation'][i])
        
        all_time[tag] += diff
        all_tag_counts[tag] += 1
        
        if tag == 8:
            if i == 0:
                tag = 9
                all_time[tag] += diff
                all_tag_counts[tag] += 1
            elif i == len(ann_df)-1:
                tag = 10
                all_time[tag] += diff
                all_tag_counts[tag] += 1
            else:
                tag = 11
                all_time[tag] += diff
                all_tag_counts[tag] += 1
            
        
average = np.divide(all_time,all_tag_counts)
rounded_avg = np.round(average,0)
        
print(all_time[8]==np.sum(all_time[9:]))
print(all_tag_counts[8]==np.sum(all_tag_counts[9:]))



