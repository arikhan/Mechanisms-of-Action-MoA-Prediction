#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:44:48 2020

@author: D.Gezgen
@author: A.Arik
"""


#We do need to handle NaN values, customize this function for your 
#problem and change the way you deal with NaN values based on requirement.
def countNaNs (data):   
    
    for column in data.columns:
        if data[column].isna().sum() != 0:
            print( "Null value count for " + str(column) + ": "  + str(data[column].isna().sum()))
        