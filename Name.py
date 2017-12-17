#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 21:19:03 2017

@author: mingrenshen
"""
# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 

# 
# !! Must have gender.py in the file

from gender import getGenders

# import data
#data_all = pd.read_csv("./data/louis_users_all_features_label_1207_updated.csv", encoding='ISO-8859-1')

getGenders(["Brian","Apple","Jessica","Zaeem","NotAName"])