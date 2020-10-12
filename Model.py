#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 12:23:15 2020

@author: D.Gezgen
@author: A.Arik
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

import config
import log_loss 

def hyper_parameter_optimization_gridSearch(train_X, train_Y):
    
    model = XGBClassifier()
    scorer = make_scorer(log_loss.metric, beta=2)
    
    clf = MultiOutputClassifier(model, config.HYPER_PARAMETERS, 
                       scoring = scorer, return_train_score=True)
    
    return clf.fit(train_X, train_Y)    
    
def gridSearch_predict(model, test_X):
    
    y_hat = model.predict(test_X)
    
    sample = pd.read_csv(config.SAMPLE_SUBMISSION_PATH, sep = ",")
    
    submission = sample.drop('sig_id', axis = 1)
    
    sample[submission.columns] = y_hat
        
    sample.to_csv(config.PREDICTION_PATH, sep = ";", index = False)
    
    config.SUBMISSION_NUMBER += 1
    
def train(x_train, y_train):
    
    params = {'eta': 0.025, 
              'gamma': 0.05,
              'max_depth': 3,
              'min_child_weight': 1,
              'subsample': 0.7,
              'colsample_bytree': 0.6,
              'lambda': 0.01,
              'alpha': 0.5}

    clf = MultiOutputClassifier(XGBClassifier())
    model = clf.fit(x_train, y_train)
    return model 

def validate(model, x_val, y_val):
    
    y_val_hat = pd.DataFrame(columns = y_val.columns, data = model.predict(x_val))
    log_loss_score = log_loss.metric(y_val, y_val_hat)
    precision_score = log_loss.macro_average_precision(y_val, y_val_hat)    
    recall_score = log_loss.macro_average_recall(y_val, y_val_hat)    
    
    return log_loss_score, precision_score, recall_score

def predict(model, x_test, sig_id):
    
    sample = pd.read_csv(config.SAMPLE_SUBMISSION_PATH, sep = ",")
    columns = sample.drop('sig_id', axis = 1).columns
    public_id = list(sample['sig_id'].values)

    df_test = pd.read_csv(config.TEST_DATASET_PATH, sep = ",")
    test_id = list(df_test['sig_id'].values)
    private_id = list(set(test_id)-set(public_id))

    # baseline : set cp_type = {'trt_cp':0.1, 'ctl_vehicle':0 }
    submission = pd.DataFrame(index = public_id+private_id, columns = columns)
    submission.index.name = 'sig_id'
    submission[:] = 0.1
    
    y_hat = model.predict_proba(x_test)
        
    index = 0
    for value in y_hat:
        submission[columns[index]] = list(np.maximum( np.minimum(pd.DataFrame(value)[1], 1-10**-15), 10**-15))
        index = index + 1

    #--------------------------------------------------------------------------
    submission.loc[df_test[df_test.cp_type=='ctl_vehicle'].sig_id]=0
    
    submission.to_csv(config.SUBMISSION_PATH, index = True)
