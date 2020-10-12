#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 21:45:04 2020

@author: D.Gezgen
@author: A.Arik
"""

import pandas as pd
import numpy as np
import config
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#log_loss metric based on Evaluation formula on Kaggle for more info:
#https://www.kaggle.com/c/lish-moa/overview/evaluation
def metric(y_true, y_pred):
    
    column_results = []
    
    for column in y_true:
        
        #Apply log-loss for a single MoA feature
        column_results.append(pd.Series(-1.0 * ( y_true[column] * np.log(y_pred[column] + 1e-15) + (1 - y_true[column]) * np.log(1- y_pred[column] + 1e-15) )).sum() / len(y_pred[column]))
    
    #return the result by dividing to the #MoAs to make it multi-class.
    return sum(column_results) / len(column_results)


#Test code snippet:
#################################################
#labels = pd.read_csv(config.TRAIN_TARGET_PATH)

#labels.drop('sig_id',inplace = True, axis = 1)

#pred = labels.copy()

#Value comes as 0.0 as expected.
#value = metric(pred, labels)
#################################################

def macro_average_precision(y_true, y_pred):
    precisions = 0
    for column in y_true.columns:
        precisions += precision_score(y_true[column], y_pred[column], average = 'macro')
        
    return precisions / len(y_true.columns)

def macro_average_recall(y_true, y_pred):
    recalls = 0
    for column in y_true.columns:
        recalls += recall_score(y_true[column], y_pred[column], average = 'macro')
        
    return recalls / len(y_true.columns)