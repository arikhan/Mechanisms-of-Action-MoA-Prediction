#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 20:10:45 2020

@author: D.Gezgen
@author: A.Arik
"""

import pandas as pd
import numpy as np
import seaborn as sns
from Model import train, validate

from sklearn.preprocessing import FunctionTransformer
from sklearn import preprocessing
from sklearn.feature_selection import f_classif, SelectKBest
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from scipy import stats

import config

#This function reduces the memory allocation by changing the data types 
def memory_reduction_df(df):
    
    for column in df.columns:
        
        if 'float' in str(df[column].dtype):
        
            #if all of the values are within float16 range
            if df[column].max() <= config.FLOAT16_MAX_RANGE and df[column].min() >= config.FLOAT16_MIN_RANGE:
                df[column] = df[column].astype('float16')
                
            #if all of the values are within float32 range
            elif df[column].max() <= config.FLOAT32_MAX_RANGE and df[column].min() >= config.FLOAT32_MIN_RANGE:
                df[column] = df[column].astype('float32')
            
        elif 'int' in str(df[column].dtype):
    
            #if all of the values are within float16 range
            if df[column].max() <= config.INT8_MAX_RANGE and df[column].min() >= config.INT8_MIN_RANGE:
                df[column] = df[column].astype('int8')
                
            #if all of the values are within float16 range since ranges are same with int16
            elif df[column].max() <= config.FLOAT16_MAX_RANGE and df[column].min() >= config.FLOAT16_MIN_RANGE:
                df[column] = df[column].astype('int16')
                
            #if all of the values are within float32 range since ranges are same with int32
            elif df[column].max() <= config.FLOAT32_MAX_RANGE and df[column].min() >= config.FLOAT32_MIN_RANGE:
                df[column] = df[column].astype('int32')
         
    return df

def square_root_transformation(data):
    
    for column in data.columns:
        
        data[column] = np.sqrt(data[column])
        
    return data
    
#We may need a log transformation to fit the data better for the metric by reducing the variance.
def log_transformation(data):
    
    transformer = FunctionTransformer(np.log1p, validate=True)
    
    for column in data.columns:
        
        temp = data[column].values.reshape(-1,1)
        
        data[column] = transformer.transform(temp + 1)
        
    return data

#We may need a exponential transformation to make a re-transformation back to the original values.
def exponential_transformation(data):
    
    transformer = FunctionTransformer(np.exp1p, validate=True)
    
    for column in data.columns:
        
        if column not in config.CATEGORICALS:
        
            data[column] = transformer.transform(data[column])
        
    return data

def cat_to_num(train, test):
    
    train['isTrain'] = 1
    test['isTrain'] = 0
    
    data = pd.concat([train, test])
    le = preprocessing.LabelEncoder()
    
    for column in config.CATEGORICALS:
        data[column] = le.fit_transform(data[column])
        
    train = data.loc[data['isTrain'] == 1]
    test = data.loc[data['isTrain'] == 0]
    
    train.drop('isTrain', inplace = True, axis = 1)
    test.drop('isTrain', inplace = True, axis = 1)

    return train, test
    
def f_score(train_df, targets):
    
    selected_features = [] 
    for label in targets.columns:
        selector = SelectKBest(f_classif, k='all')
        selector.fit(train_df, targets[label])
        selected_features.append(list(selector.scores_))
    
    # MeanCS 
    selected_features_mean = np.mean(selected_features, axis=0) 
    # MaxCS
    selected_features_max = np.max(selected_features, axis=0)
    
    features = pd.DataFrame()
    
    features['features'] = train_df.columns
    
    features['mean'] = selected_features_mean
    
    features['max'] = selected_features_max

    return features

def xgboost_feature_selection(train_df,targets):
    
    selected_features = [] 

    model = train(train_df, targets)
    
    for clf in model.estimators_:
        selected_features.append(clf.feature_importances_)

    # MeanCS 
    selected_features_mean = np.mean(selected_features, axis=0) 
    # MaxCS
    selected_features_max = np.max(selected_features, axis=0)
    
    features = pd.DataFrame()
    
    features['features'] = train_df.columns
    
    features['mean'] = selected_features_mean
    
    features['max'] = selected_features_max

    return features    

def feature_variances(train, test):
    
    train_variances = pd.DataFrame()
    test_variances = pd.DataFrame()
    
    train_values = []
    train_percentages = []
    train_min_values = []
    train_max_values = []
    train_mean_values = []
    train_deviations = []

    test_values = []
    test_percentages = []
    test_min_values = []
    test_max_values = []
    test_mean_values = []
    test_deviations = []

    train_variances['columns'] = train.columns
    test_variances['columns'] = test.columns
    
    for feature in train.columns:
        
        train_values.append(len(train[feature].unique()))
        train_percentages.append(len(train[feature].unique()) * 100.0 / len(train))
        train_min_values.append(train[feature].min())
        train_max_values.append(train[feature].max())
        train_mean_values.append(train[feature].mean())
        train_deviations.append(train[feature].std())
        
        test_values.append(len(test[feature].unique()))
        test_percentages.append(len(test[feature].unique()) * 100.0 / len(test))
        test_min_values.append(test[feature].min())
        test_max_values.append(test[feature].max())
        test_mean_values.append(test[feature].mean())
        test_deviations.append(test[feature].std())

    train_variances['distincts'] = train_values
    train_variances['percentages'] = train_percentages
    train_variances['min_values'] = train_min_values
    train_variances['max_values'] = train_max_values
    train_variances['mean_values'] = train_mean_values
    train_variances['std_values'] = train_deviations
    
    test_variances['distincts'] = test_values
    test_variances['percentages'] = test_percentages
    test_variances['min_values'] = test_min_values
    test_variances['max_values'] = test_max_values
    test_variances['mean_values'] = test_mean_values
    test_variances['std_values'] = test_deviations

    return train_variances, test_variances

def data_binning(train, test, statistic, bin_number):
    
    for feature in train.columns:
        
        feature_max = max(train[feature].max(), test[feature].max())
        feature_min = min(train[feature].min(), test[feature].min())
        
        train[feature] = stats.binned_statistic(train[feature], np.arange(len(train)), statistic = statistic, bins=bin_number, range = [feature_min, feature_max])[2]
        test[feature] = stats.binned_statistic(test[feature], np.arange(len(test)), statistic = statistic, bins=bin_number, range = [feature_min, feature_max])[2]
    
    return train, test

def reduce_variances(train, test):
    
    train['train'] = 1
    test['train'] = 0
    
    dataframe = pd.concat([train, test]).reset_index(drop = True)
    
    labels = dataframe['train']
    dataframe.drop('train', inplace = True, axis = 1)
    
    for feature in dataframe.columns:
        
        max_value = dataframe[feature].max()
        
        listy = dataframe[feature].value_counts() * 100 / len(dataframe[feature])
        
        indexes =  list(listy.loc[listy < config.RARE_PERCENTAGE].index)
        
        for value in indexes:
            
            dataframe.loc[dataframe[feature] == value, feature] = max_value + 1
        
    dataframe['train'] = labels
    
    train = dataframe.loc[dataframe['train'] == 1]
    test = dataframe.loc[dataframe['train'] == 0]
    
    train.drop('train', axis = 1, inplace = True)
    test.drop('train', axis = 1, inplace = True)
    
    return train, test