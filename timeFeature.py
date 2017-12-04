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


# Adding 19 empty columns for recording

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
        userFeature.loc[row_index,temp_str] = monSeries[mon] * 1.0 / totNum
   # Counting WeekDay Frequency
   # weekday() 0 ~ 4 : Mon ~ Fri, 5 ~ 6 : Sat ~ Sun
    tempDF['weekday'] = tempDF['tweet_date'].apply(lambda i: i.weekday() >= 4 )
    weekdaySeries = tempDF['weekday'].value_counts()
    for item in weekdaySeries:
        if item == True:
            userFeature.loc[row_index,"freqWeekend"] = 1.0 * weekdaySeries[True] / totNum
        if item == False:
            userFeature.loc[row_index,"freqWeekDay"] = 1.0 * weekdaySeries[False] / totNum

   # Counting DayTime
    numMoring = 0
    numNoon = 0
    numAfternoon = 0
    numNight = 0
    numLateNight = 0
   
    for temp_row_index, tempDFrow in tempDF.iterrows():
        temp_time = dt.datetime.strptime(tempDFrow["tweet_time"], '%H:%M:%S').time()
        if (temp_time >= dt.time(6,0,0)) and (temp_time < dt.time(12,0,0)):
            numMoring += 1
        if (temp_time >= dt.time(11,0,0)) and (temp_time < dt.time(14,0,0)):
            numNoon += 1
        if (temp_time >= dt.time(12,0,0)) and (temp_time < dt.time(18,0,0)):
            numAfternoon += 1
        if (temp_time >= dt.time(17,0,0)) and (temp_time < dt.time(23,0,0)):
            numNight += 1
        if (temp_time >= dt.time(23,0,0)) or (temp_time < dt.time(6,0,0)):
            numLateNight += 1
    userFeature.loc[row_index,"freqMoring"] = 1.0 * numMoring / totNum      # 6 ~ 12
    userFeature.loc[row_index,"freqNoon"] = 1.0 * numNoon / totNum         # 11 ~ 14
    userFeature.loc[row_index,"freqAfternoon"] = 1.0 * numAfternoon / totNum    # 12 ~ 18
    userFeature.loc[row_index,"freqNight"] = 1.0 * numNight / totNum        # 17 ~ 23
    userFeature.loc[row_index,"freqLateNight"] = 1.0 * numLateNight / totNum    # 22 ~ 6
    print row_index

userFeature.to_csv('louis_users_features_label_1203.csv', header=True)