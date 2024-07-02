#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 23:05:56 2024

@author: mahera
"""

import numpy as np
import pandas as pd
import warnings
import joblib
warnings.simplefilter('ignore')


DATA_FOLDER = "/home/mahera/Documents/GenAI Challenge/published_data"
FS = 30 # Sampling Rate
TOTAL_CLASSES = 9

#-----------------------------------------------------------------------------
# Remove redundant data -> ankles and knees and conf

def remove_redundant_kp(kp_df):
  kp_df = kp_df.loc[:, ~kp_df.columns.str.contains(
      'conf|left_knee|right_knee|left_ankle|right_ankle', regex=True)]
  return kp_df


# Data Smoothing

SMOOTH_LEN = 3 #seconds


def smooth_kp(kp_col):
  zero_idx = np.flatnonzero(kp_col == 0)
  split_idx = np.split(zero_idx, np.flatnonzero(np.diff(zero_idx) > 1) + 1)
  for each_split_idx in split_idx:
    if len(each_split_idx) == 0 or each_split_idx[0] == 0 or each_split_idx[-1] == (len(kp_col) - 1) or len(each_split_idx) > SMOOTH_LEN*FS:
      continue
    xp = [each_split_idx[0] - 1, each_split_idx[-1] + 1]
    fp = kp_col[xp]
    interp_kp = np.interp(each_split_idx, xp, fp)
    kp_col[each_split_idx] = interp_kp
  return kp_col


# Segmentation

def segment(data, max_time, sub_window_size, stride_size):
    sub_windows = np.arange(sub_window_size)[None, :] + np.arange(0, max_time, stride_size)[:, None]

    row, col = np.where(sub_windows >= max_time)
    uniq_row = len(np.unique(row))

    if uniq_row > 0 and row[0] > 0:
        sub_windows = sub_windows[:-uniq_row, :]

    return data[sub_windows]

import scipy

def extract_feature(data, fs):
    mean_ft = np.mean(data, axis=0)
    std_ft = np.std(data, axis=0)
    max_ft = np.max(data, axis=0)
    min_ft = np.min(data, axis=0)
    var_ft = np.var(data, axis=0)
    med_ft = np.median(data, axis=0)
    sum_ft = np.sum(data, axis=0)
    features = np.array([mean_ft, std_ft, max_ft, min_ft, var_ft, med_ft, sum_ft]).T.flatten()
    features = np.nan_to_num(features)
    return features

WINDOW_SIZE = 2 # seconds
OVERLAP_RATE = 0.5 * WINDOW_SIZE # overlap 50% of window size

# Joint Angles
def cal_angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle
def mid_point(a,b):
  a = (np.array(a))
  b = (np.array(b))
  mid = (np.subtract(a,b))/2
  return mid

def extract_joint_angles(kp_data, steps=2):
    # steps = 2 if kp_data is removed conf columns
    # steps = 3 if kp_data has conf columns
    left_elbow_shoulder_hip = np.asarray([cal_angle(kp_data[i, 7*steps:(7*steps+2)], kp_data[i, 5*steps:(5*steps+2)], kp_data[i, 11*steps:(11*steps+2)])
                                          for i in range(len(kp_data))])
    left_elbow_shoulder_hip = np.nan_to_num(left_elbow_shoulder_hip)
    right_elbow_shoulder_hip = np.asarray([cal_angle(kp_data[i, 8*steps:(8*steps+2)], kp_data[i, 6*steps:(6*steps+2)], kp_data[i, 12*steps:(12*steps+2)])
                                            for i in range(len(kp_data))])
    right_elbow_shoulder_hip = np.nan_to_num(right_elbow_shoulder_hip)
    left_wrist_elbow_shoulder = np.asarray([cal_angle(kp_data[i, 9*steps:(9*steps+2)], kp_data[i, 7*steps:(7*steps+2)], kp_data[i, 5*steps:(5*steps + 2)])
                                            for i in range(len(kp_data))])
    left_wrist_elbow_shoulder = np.nan_to_num(left_wrist_elbow_shoulder)
    right_wrist_elbow_shoulder = np.asarray([cal_angle(kp_data[i, 10*steps:(10*steps+2)], kp_data[i, 8*steps:(8*steps+2)], kp_data[i, 6*steps:(6*steps+2)])
                                              for i in range(len(kp_data))])
    right_wrist_elbow_shoulder = np.nan_to_num(right_wrist_elbow_shoulder)


    right_elbow_shoulder = np.asarray([cal_angle(kp_data[i, 8*steps:(8*steps+2)], kp_data[i, 6*steps:(6*steps+2)], kp_data[i, 5*steps:(5*steps+2)])
                                              for i in range(len(kp_data))])
    right_elbow_shoulder = np.nan_to_num(right_elbow_shoulder)
    left_elbow_shoulder = np.asarray([cal_angle(kp_data[i, 6*steps:(6*steps+2)], kp_data[i, 5*steps:(5*steps+2)], kp_data[i, 7*steps:(7*steps+2)])
                                              for i in range(len(kp_data))])
    left_elbow_shoulder = np.nan_to_num(left_elbow_shoulder)

    # < : right shoulder, nose, left shoulder
    right_shoulder_nose_left_shoulder = np.asarray([cal_angle(kp_data[i, 6*steps:(6*steps+2)], kp_data[i, 0*steps:(0*steps+2)], kp_data[i, 5*steps:(5*steps+2)])
                                              for i in range(len(kp_data))])
    right_shoulder_nose_left_shoulder = np.nan_to_num(right_shoulder_nose_left_shoulder)


    # Calculate center of hip
    h1 = [kp_data[i, 11*steps:(11*steps+2)] for i in range(len(kp_data))]
    h2 = [kp_data[i, 12*steps:(12*steps+2)] for i in range(len(kp_data))]
    hip_center = mid_point(h1,h2)

    # < : hip center, left shoulder, elbow
    hip_center_left_shoulder_elbow = np.asarray([cal_angle(hip_center[i] , kp_data[i, 5*steps:(5*steps+2)], kp_data[i, 7*steps:(7*steps+2)])
                                              for i in range(len(kp_data))])
    hip_center_left_shoulder_elbow = np.nan_to_num(hip_center_left_shoulder_elbow)
    # < : hip center, right shoulder, elbow
    hip_center_right_shoulder_elbow = np.asarray([cal_angle(mid_point(kp_data[i, 11*steps:(11*steps+2)],kp_data[i, 12*steps:(12*steps+2)]) , kp_data[i, 6*steps:(6*steps+2)], kp_data[i, 8*steps:(8*steps+2)])
                                              for i in range(len(kp_data))])
    hip_center_right_shoulder_elbow = np.nan_to_num(hip_center_right_shoulder_elbow)



    joint_angles = np.array([left_elbow_shoulder_hip, right_elbow_shoulder_hip, left_wrist_elbow_shoulder, right_wrist_elbow_shoulder, right_elbow_shoulder, left_elbow_shoulder, right_shoulder_nose_left_shoulder, hip_center_left_shoulder_elbow,hip_center_right_shoulder_elbow]).T

    return joint_angles

# Velocity, acc, jerk
def extract_velocity(kp_data):
    velocity = np.diff(kp_data, axis=0)
    return velocity
def extract_acceleration(kp_data):
    velocity = extract_velocity(kp_data)
    acc = np.diff(velocity,axis=0)
    return acc
def extract_jerk(kp_data):
    acc = extract_acceleration(kp_data)
    jerk = np.diff(acc,axis=0)
    return jerk

def cal_distance(a,b):
  distance = np.sqrt(np.sum((b-a)**2))
  return distance

def extract_joint_distances(kp_data, steps=2):
    # steps = 2 if kp_data is removed conf columns
    # steps = 3 if kp_data has conf columns

    # -- : right shoulder, wrist
    d_right_shoulder_wrist = np.asarray([cal_distance(kp_data[i, 6*steps:(6*steps+2)], kp_data[i, 10*steps:(10*steps+2)])
                                          for i in range(len(kp_data))])
    d_right_shoulder_wrist = np.nan_to_num(d_right_shoulder_wrist)
    # -- : left shoulder, wrist
    d_left_shoulder_wrist = np.asarray([cal_distance(kp_data[i, 5*steps:(5*steps+2)], kp_data[i, 9*steps:(9*steps+2)])
                                          for i in range(len(kp_data))])
    d_left_shoulder_wrist = np.nan_to_num(d_left_shoulder_wrist)

    #Calculate hip center
    h1 = [kp_data[i, 11*steps:(11*steps+2)] for i in range(len(kp_data))]
    h2 = [kp_data[i, 12*steps:(12*steps+2)] for i in range(len(kp_data))]
    hip_center = mid_point(h1,h2)

    # -- : hip center, right elbow
    d_hipc_right_elbow = np.asarray([cal_distance(hip_center[i], kp_data[i, 8*steps:(8*steps+2)])
                                          for i in range(len(kp_data))])
    d_hipc_right_elbow = np.nan_to_num(d_hipc_right_elbow)
    # -- : hip center, left elbow
    d_hipc_left_elbow = np.asarray([cal_distance(hip_center[i], kp_data[i, 7*steps:(7*steps+2)])
                                          for i in range(len(kp_data))])
    d_hipc_left_elbow = np.nan_to_num(d_hipc_left_elbow)
    # -- : hip center, right wrist
    d_hipc_right_wrist = np.asarray([cal_distance(hip_center[i], kp_data[i, 10*steps:(10*steps+2)])
                                          for i in range(len(kp_data))])
    d_hipc_right_wrist = np.nan_to_num(d_hipc_right_wrist)
    # -- : hip center, left wrist
    d_hipc_left_wrist = np.asarray([cal_distance(hip_center[i], kp_data[i, 9*steps:(9*steps+2)])
                                          for i in range(len(kp_data))])
    d_hipc_left_wrist = np.nan_to_num(d_hipc_left_wrist)
    # -- : right wrist, left wrist
    d_right_wrist_left_wrist = np.asarray([cal_distance(kp_data[i, 10*steps:(10*steps+2)], kp_data[i, 9*steps:(9*steps+2)])
                                          for i in range(len(kp_data))])
    d_right_wrist_left_wrist = np.nan_to_num(d_right_wrist_left_wrist)
    # -- : right hip, left hip
    d_right_hip_left_hip = np.asarray([cal_distance(kp_data[i, 12*steps:(12*steps+2)], kp_data[i, 11*steps:(11*steps+2)])
                                          for i in range(len(kp_data))])
    d_right_hip_left_hip = np.nan_to_num(d_right_hip_left_hip)
    # -- : right hip, right wrist
    d_right_hip_right_wrist = np.asarray([cal_distance(kp_data[i, 12*steps:(12*steps+2)], kp_data[i, 10*steps:(10*steps+2)])
                                          for i in range(len(kp_data))])
    d_right_hip_right_wrist =  np.nan_to_num(d_right_hip_right_wrist)
    # -- : left hip, left wrist
    d_left_hip_left_wrist = np.asarray([cal_distance(kp_data[i, 11*steps:(11*steps+2)], kp_data[i, 9*steps:(9*steps+2)])
                                          for i in range(len(kp_data))])
    d_left_hip_left_wrist = np.nan_to_num(d_left_hip_left_wrist)


    # Calculate Spine joint
    s1 = [kp_data[i, 5*steps:(5*steps+2)] for i in range(len(kp_data))]
    s2 = [kp_data[i, 6*steps:(6*steps+2)] for i in range(len(kp_data))]
    spine = mid_point(s1,s2)


    # Calculate distance between hip center to spine joint -> relatively invariant when performing activities
    norm_distance = np.asarray([cal_distance(hip_center[i], spine[i])
                                          for i in range(len(kp_data))])
    norm_distance = (np.nan_to_num(norm_distance)).reshape(-1)

    joint_distances = np.array([d_right_shoulder_wrist, d_left_shoulder_wrist, d_hipc_right_elbow, d_hipc_left_elbow, d_hipc_right_wrist, d_hipc_left_wrist, d_right_wrist_left_wrist, d_right_hip_left_hip, d_right_hip_right_wrist, d_left_hip_left_wrist]).T

    # Normalize the joint distances
    normalized_joint_distances = joint_distances/norm_distance[:,None]

    return joint_distances

#-----------------------------------------------------------------------------
# MODEL XGBOOST + SMOTE
import pickle
model_ml = pickle.load(open('/home/mahera/Documents/GenAI Challenge/Training/final_XGB_model.pkl','rb'))

#-----------------------------------------------------------------------------
# TESTING

TEST_ID = ["N03T1", "N03T2",
           "N05T1", "N05T2",
           "N09T1", "N09T2",
           "S06T1", "S06T2",
           "S04T1", "S04T2",
           "S12T1", "S12T2",

]
SUBMISSION_CSV = "/home/mahera/Documents/GenAI Challenge/Testing/submission_final.csv"

def generate_timestamp(len_file):
    time_intervals = []
    for i in range(len_file):
        minutes, seconds = divmod(i, 60)
        start_time = f"{int(minutes):02d}:{int(seconds):02d}"
        minutes, seconds = divmod(i+1, 60)
        end_time = f"{int(minutes):02d}:{int(seconds):02d}"
        time_interval = f"{start_time} - {end_time}"
        time_intervals.append(time_interval)
    return time_intervals


#----------------------
submission = pd.DataFrame(columns=['subjectID', 'timestamp', 'activityID'])
segs =[14, 43, 4, 14, 5, 19,5, 9, 41, 43, 4, 14, 5, 19,5, 9, 41, 0   ]
      #[8  0   1  2   3  4  5  6  7   0   1  2   3  4  5  6  7  <_>]

for user_id in TEST_ID:
    keypoint_csv = DATA_FOLDER + "/keypoints/{}_keypoint.csv".format(user_id)
    kp_df = pd.read_csv(keypoint_csv)
    kp_df = kp_df.loc[:, ~kp_df.columns.str.contains('^Unnamed')]
    kp_df = remove_redundant_kp(kp_df)
    for i, kp_name in enumerate(kp_df.columns):
          kp_df.iloc[:, i] = smooth_kp(np.array(kp_df.iloc[:, i]))
          
    time = 0
    for i in range(len(segs)):
        if i == len(segs) - 1:
            seg = kp_df.loc[time*FS :]
        else:
            seg = kp_df.loc[time*FS : (time+segs[i])*FS]
            time = time + segs[i]
            
        if len(seg) > 0 and (len(seg) >= WINDOW_SIZE * FS):
    
            # Calculate joint angles from keypoint data
            joint_angles = extract_joint_angles(np.array(seg))
            
            # Calculate joint distances from keypoint data
            joint_distances = extract_joint_distances(np.array(seg))

            # Segment keypoint data and joint angles by WINDOW_SIZE without OVERLAP_RATE
            ws_seg = segment(np.array(seg), max_time=len(seg), sub_window_size=WINDOW_SIZE * FS, stride_size=int(WINDOW_SIZE * FS))
            joint_angles_seg = segment(joint_angles, max_time=len(seg), sub_window_size=WINDOW_SIZE * FS,
                                              stride_size=int(WINDOW_SIZE * FS))
            joint_distances_seg = segment(joint_distances, max_time=len(seg), sub_window_size=WINDOW_SIZE * FS,
                                              stride_size=int((WINDOW_SIZE* FS)))
            # Calculate velocity, acceleration, jerk from each segment of keypoint data
            velocity_seg = [extract_velocity(ws_seg[i]) for i in range(len(ws_seg))]
            acc_seg = [extract_acceleration(ws_seg[i]) for i in range(len(ws_seg))]
            jerk_seg = [extract_jerk(ws_seg[i]) for i in range(len(ws_seg))]
            
            # Calculate features from each segment of keypoint data, joint angles and velocity
            feature_seg = [extract_feature(ws_seg[i], FS) for i in range(len(ws_seg))]
            feature_joint_angles_seg = [extract_feature(joint_angles_seg[i], FS) for i in
                                        range(len(joint_angles_seg))]
            feature_joint_distances_seg = [extract_feature(joint_distances_seg[i], FS) for i in
                                        range(len(joint_distances_seg))]
        
            feature_velocity_seg = [extract_feature(extract_velocity(ws_seg[i]), FS) for i in range(len(ws_seg))]
            feature_acc_seg = [extract_feature(extract_acceleration(ws_seg[i]), FS) for i in range(len(ws_seg))]
            feature_jerk_seg = [extract_feature(extract_jerk(ws_seg[i]), FS) for i in range(len(ws_seg))]
        
            # Concatenate all features
            feature_seg = np.concatenate([feature_seg, feature_joint_angles_seg, feature_joint_distances_seg, 
                                          feature_velocity_seg, feature_acc_seg, feature_jerk_seg], axis=1)

            # Scale
            scaler = joblib.load('/home/mahera/Documents/GenAI Challenge/Training/scaler.gz')
            feature_seg = scaler.transform(feature_seg)
            
            # Predict
            y_predict = model_ml.predict(feature_seg)

            # Get prediction in each second
            activityID = np.repeat(y_predict, WINDOW_SIZE)
        
            # Add result in submission dataframe
            subjectID = [user_id]*len(activityID)
            timestamp = generate_timestamp(len(activityID))
        
            new_data = {'subjectID': subjectID, 'timestamp': timestamp, 'activityID': activityID}
            new_df = pd.DataFrame(new_data)
        
            submission = pd.concat([submission, new_df])

# Export submission dataframe into csv
submission.to_csv(SUBMISSION_CSV, header = False, index=False)



