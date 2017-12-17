#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 21:19:03 2017

@author: mingrenshen
"""
# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 

# time
import time 
# !! Must have gender.py in the file

from gender import getGenders

# import data
data_all = pd.read_csv("./data/louis_users_all_features_label_1207_updated.csv", encoding='ISO-8859-1')

#getGenders(["Brian","Apple","Jessica","Zaeem","NotAName"])

data_all["user_predicted_fname_genderM"] = 0
data_all["user_predicted_fname_genderF"] = 0
data_all["user_predicted_fname_genderO"] = 0
data_all["user_fname_count"] = 0
#namelist = ["Brian","Apple","Jessica","Zaeem","NotAName"]
'''
for i in namelist:
    out = getGenders(i)
    print(out)
'''
# Groupby user_id then apply the needed functions
for row_index,row in data_all.iterrows():
    usr_fname = row["user_fname"]
    time.sleep(1)
    #result = getGenders(usr_fname)
    out = getGenders(usr_fname)
    if len(out[0]) != 3:
        print(row_index)
        print(out)
    else:
        print(out)
        if (out[0][0] == 'male'):
            data_all[row_index,"user_predicted_fname_genderM"] = out[0][1]
            data_all[row_index,"user_fname_count"] = out[0][2]
        
        if(out[0][0] == 'female'):
            data_all[row_index,"user_predicted_fname_genderF"] = out[0][1]
            data_all[row_index,"user_fname_count"] = out[0][2]
            
        if(out[0][0] == 'None'):
            data_all[row_index,"user_predicted_fname_genderO"] = out[0][1]
            data_all[row_index,"user_fname_count"] = out[0][2]
    
        
    #(u'male', u'1.00', 483)
    
