#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 11:02:41 2020

@author: D.Gezgen
@author: A.Arik
"""

import config

import cross_validation as CV

from dataset_tricks import memory_reduction_df as memoryReduce
from dataset_tricks import f_score
from dataset_tricks import xgboost_feature_selection
from dataset_tricks import cat_to_num
from dataset_tricks import feature_variances
from dataset_tricks import data_binning
from dataset_tricks import reduce_variances
from data_cleaning import countNaNs
from Model import train, validate, predict
from datetime import datetime

beginning = datetime.now()

print("Loading Data...")

#Load the data
train_df = config.load_data(config.TRAIN_DATASET_PATH)
target_df = config.load_data(config.TARGET_DATASET_PATH)
test_df = config.load_data(config.TEST_DATASET_PATH)
# xgboost_features = config.load_data(config.XGBOOST_FEATURE_PATH)

print("Reducing Memory Allocation...")

#Save some memory
train_df = memoryReduce(train_df)
target_df = memoryReduce(target_df)
test_df = memoryReduce(test_df)

# uncomment to check the data types and memory allocations
# train_df.info()
# test_df.info()
# target_df.info()

#go for "custom" cross validation
train_df = CV.cross_validation(train_df, target_df)

#Check for null values now and we will see that we have no NaN values.
#Therefore, no need to check for imputation methods or for models to fill missing values.
countNaNs(train_df)

#Since the metric is log-based, make a square root transformation to observe the effect.
traindf = train_df.drop(['sig_id', 'kfold' ], axis = 1)
testdf = test_df.drop(['sig_id'], axis = 1)
target_df = target_df.drop(['sig_id'], axis = 1)
 
print("Feature Engineering...")

traindf, testdf = cat_to_num(traindf, testdf)

train_binning = traindf.drop(['cp_type', 'cp_time', 'cp_dose'], axis = 1)

test_binning = testdf.drop(['cp_type', 'cp_time', 'cp_dose'], axis = 1)

train_binning, test_binning = data_binning(train_binning, test_binning, 'std', 50)

train_binning, test_binning = reduce_variances(train_binning, test_binning)

traindf[train_binning.columns] = train_binning.values

testdf[test_binning.columns] = test_binning.values

train_variances, test_variances = feature_variances(traindf, testdf)

linear_features = f_score(traindf, target_df)

#Xgboost feature selection is taking too much time, load them from csv file instead.
# xboost_features = xgboost_feature_selection(traindf, target_df)

linear_features = linear_features.sort_values(['mean'], ascending = False)

# xgboost_features = xgboost_features.sort_values(['mean'], ascending = False)

# config.write_csv(xgboost_features, config.XGBOOST_FEATURE_PATH)

columns_fvalue = linear_features.features.head(10).values.copy()

# columns_xgboost = xgboost_features.features.head(25).values.copy()

# traindf = square_root_transformation(traindf)
# testdf = square_root_transformation(testdf)

train_set = traindf[columns_fvalue].copy()

test_set = testdf[columns_fvalue].copy()

train_set['kfold'] = train_df['kfold'].copy()

target_df['kfold'] = train_df['kfold'].copy()

# Check for null values now and we will see that we have no NaN values.
# countNaNs(train_df_log)
# countNaNs(test_df_log)

#Modelling

log_loss = 1.0
precall = 0
best_model = ''
best_kfold = -1

print("Modelling Phase...")

for kfold in range(config.KFOLD_NUMBER):
    
    print("kfold value: " + str(kfold)) 
    x_train = train_set.loc[train_set['kfold'] != kfold] 
    y_train = target_df.loc[target_df['kfold'] != kfold] 
    x_train.drop('kfold', inplace = True, axis = 1)
    y_train.drop('kfold', inplace = True, axis = 1)
    
    x_val = train_set.loc[train_set['kfold'] == kfold] 
    y_val = target_df.loc[target_df['kfold'] == kfold] 
    x_val.drop('kfold', inplace = True, axis = 1)
    y_val.drop('kfold', inplace = True, axis = 1)
    
    model = train(x_train, y_train)
    print(str(datetime.now() - beginning))
    
    log_loss_score, precision_score, recall_score = validate(model, x_val, y_val)
    
    if log_loss_score < log_loss and (precision_score + recall_score) / 2 > precall:
        
        print("log_loss: " + str(log_loss_score) )
        print("precision: " + str(precision_score) )
        print("recall: " + str(recall_score) )
        
        predict(model, test_set, test_df['sig_id'])
        
        log_loss = log_loss_score
        best_model = model
        best_kfold = kfold
        precall = (precision_score + recall_score) / 2