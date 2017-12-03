#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 14:15:07 2017

@author: mingrenshen
"""

# import libarary needed

import numpy as np # linear algebra
import pandas as pd # data processing
import datetime as dt



######################################################
# read in data
######################################################
## user id 

userFeature = pd.read_csv("louis_users_features_label_1129.csv")
## user data

userData = pd.read_csv("louis_tweets_1127_labeled.csv", 
                       encoding='ISO-8859-1')
userDataTime = pd.read_csv("date_time_1203.csv" )
userDataTime.columns = ["tweet_date","tweet_time"]

userDataAll = pd.concat([userData,userDataTime], axis=1)

# only leaving userDataAll for further reference
del userData
del userDataTime

######################################################
# Data Preprosessing
######################################################
userDataAll['tweet_date'] =  pd.to_datetime(userDataAll['tweet_date'])
userDataAll['tweet_time'] =  pd.to_datetime(userDataAll['tweet_time']).dt.strftime('%H:%M:%S')

# extract user_id to make things simple
#user_id_list = userFeature["user_id"].tolist()

# Adding two empty column for recording

# WeekDay

userFeature["freqWeekDay"] = 0
userFeature["freqWeekend"] = 0

# Day time
userFeature["freqMoring"] = 0      # 6 ~ 12
userFeature["freqNoon"] = 0        # 11 ~ 14
userFeature["freqAfternoon"] = 0   # 12 ~ 18
userFeature["freqNight"] = 0       # 17 ~ 23
userFeature["freqLateNight"] = 0   # 22 ~ 6

# Monthes
userFeature["freqJan"] = 0
userFeature["freqFeb"] = 0  
userFeature["freqMar"] = 0
userFeature["freqApr"] = 0
userFeature["freqMay"] = 0
userFeature["freqJun"] = 0
userFeature["freqJul"] = 0
userFeature["freqAug"] = 0
userFeature["freqSep"] = 0
userFeature["freqOct"] = 0
userFeature["freqNov"] = 0
userFeature["freqDec"] = 0

######################################################
# Define Functions for statistics
######################################################
# Weekday

# Groupby user_id then apply the needed functions
for row_index,row in userFeature.iterrows():
   uid = row["user_id"]
   tempDF = userDataAll.loc[userDataAll.user_id == uid]
   totNum , _ = tempDF.shape
   # Counting Month Frequency
   monSeries = tempDF.tweet_date.dt.strftime('%b').value_counts()
   for mon in monSeries.index:
       temp_str = 'freq' + mon
       row[temp_str] = monSeries[mon] * 1.0 / totNum
   # Counting WeekDay Frequency
   # weekday() 0 ~ 4 : Mon ~ Fri, 5 ~ 6 : Sat ~ Sun
   tempDF['weekday'] = tempDF['tweet_date'].apply(lambda i: i.weekday() >= 4 )
   weekdaySeries = tempDF['weekday'].value_counts()
   row["freqWeekend"] = 1.0 * weekdaySeries[True] / totNum 
   row["freqWeekDay"] = 1.0 * weekdaySeries[False] / totNum
   # Counting DayTime
   