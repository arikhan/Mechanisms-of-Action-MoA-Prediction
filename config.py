#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 20:57:33 2020

@author: D.Gezgen
@author: A.Arik

"""

import pandas as pd
import numpy as np

#Range values taken from: 
#https://medium.com/@vincentteyssier/optimizing-the-size-of-a-pandas-dataframe-for-low-memory-environment-5f07db3d72e

FLOAT16_MIN_RANGE = -32768

FLOAT16_MAX_RANGE = 32767

FLOAT32_MIN_RANGE = -2147483648

FLOAT32_MAX_RANGE = 2147483647

INT8_MIN_RANGE = -128

INT8_MAX_RANGE = 127

RARE_PERCENTAGE = 5.0

STURGE_VALUE = int(1 + np.log2(23814))

KFOLD_NUMBER = STURGE_VALUE

TRAIN_DATASET_PATH = "lish-moa/train_features.csv"

TEST_DATASET_PATH = "lish-moa/test_features.csv"

TRAIN_FOLDS_DATASET_PATH = "lish-moa/train_folds.csv"

TARGET_DATASET_PATH = "lish-moa/train_targets_scored.csv"

NONTARGET_DATASET_PATH = "lish-moa/train_targets_nonscored.csv"

SAMPLE_SUBMISSION_PATH = "lish-moa/sample_submission.csv"

SUBMISSION_PATH = "Submissions/submission.csv"

XGBOOST_FEATURE_PATH = "lish-moa/xgboost_features.csv"

HYPER_PARAMETERS = {'eta': [0.01, 0.015, 0.025, 0.05, 0.1], 
                    'gamma': [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                    'max_depth': [3, 5, 7, 9, 12, 15, 17, 25],
                    'min_child_weight': [ 1, 3, 5, 7],
                    'subsample': [ 0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'lambda': [0.01, 0.1, 1.0 ],
                    'alpha': [0, 0.1, 0.5, 1.0]}

CATEGORICALS = ['cp_type', 'cp_time', 'cp_dose']

def load_data(path):
    
    return pd.read_csv(path, sep = ",")

def write_csv(dataframe, path):
    
    return dataframe.to_csv(path, sep = ",")