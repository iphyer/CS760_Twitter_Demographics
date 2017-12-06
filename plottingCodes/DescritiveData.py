#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 20:44:28 2017

@author: mingrenshen
"""
# import libarary needed

import pandas as pd # data processing
import matplotlib.pyplot as plt


######################################################
# read in data
######################################################

## user data

allUsrFeatureData = pd.read_csv("../data/louis_users_all_features_label_1205_updated.csv")

# plotting Data

#grouped = allUsrFeatureData['freqWeekDay'].groupby('gender')
print allUsrFeatureData['gender'].value_counts()

# Font for figure
font_axis_publish = {
        'color':  'black',
        'weight': 'normal',
        'size': 15,
        }

#ax = allUsrFeatureData.boxplot(column='freqWeekDay',by='gender')

#plt.ylabel('RMSF ($\AA$)', fontdict=font_axis_publish)

#plt.xlim(0,1000)
#plt.set_title("")


col_list = list(allUsrFeatureData.columns.values)

starting_index = col_list.index("gender")

for i in range(len(col_list)):
    if i > starting_index:
        curr_feature = col_list[i]
        allUsrFeatureData.boxplot(column=curr_feature,by='gender')
        plt.title(curr_feature, fontdict=font_axis_publish)
        plt.suptitle("")
        plt.xlabel('gender', fontdict=font_axis_publish)
        #plt.show()
        str_tmp = curr_feature + '.png'
        plt.savefig(str_tmp)
        plt.close()