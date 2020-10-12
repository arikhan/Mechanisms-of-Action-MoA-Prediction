"""
Created on Sun Sep 13 20:05:52 2020

@author: D.Gezgen
@author: A.Arik
"""

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn import preprocessing
from skmultilearn.model_selection import IterativeStratification

import config

from sklearn import model_selection

# We could either vectorize the labelset based on instances and give a representative label for stratified kfold,
# or we will use another method to make it done...

def cross_validation(train_df, targets_df):
    
    #train_df = target_vectorization(train_df, targets_df)
    train_df = kfold(train_df, targets_df)
    #For checkpoint purposes
    #train_df.to_csv( config.TRAIN_FOLDS_DATASET_PATH, index = False, sep = ';')
    return train_df

# kfold is kind of a try-out method in here for cross validation. We are particularly looking for
# imbalanced distribution of the "combinational" equivalent of multi labeled target classes.
# This may or may not make sense but intuitively, it could work since thinking based on combinations 
# on drug components may make sense on drug production after all.
def kfold(train_df, targets_df):
    
    train_df['kfold'] = -1
    
    train_df = train_df.sample(frac=1).reset_index(drop = True)
    
    k_fold = IterativeStratification(n_splits=config.KFOLD_NUMBER, order=1)
    for f, (t_,v_) in enumerate(k_fold.split(X = train_df, y = targets_df.drop('sig_id', axis = 1))):
        train_df.loc[v_, 'kfold'] = f

    return train_df

    # y = train_df.label_enc.values
    
    # kf = model_selection.StratifiedKFold(n_splits = )
    
    # for f, (t_,v_) in enumerate(kf.split(X=train_df, y=y)):
    
    #     train_df.loc[v_, 'kfold'] = f
        
    # return train_df
    
# Convert multi-label classes in to single label classes by vectorizing and using label encoding.
# The idea is to make cross validation based on target combinations and keep the percentages of
# combinations leveled for every fold.
def target_vectorization(train, targets):
    
    solo_targets = targets.drop('sig_id', axis = 1)

    target_concat = []
    for index, row in solo_targets.iterrows():
        target_concat.append(pd.Series(row).astype('str').str.cat(sep=''))
    
    le = preprocessing.LabelEncoder()
    
    train['label_enc'] = le.fit_transform(target_concat)
    
    return train