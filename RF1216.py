#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 06:36:51 2017

@author: mingrenshen
"""

# import libarary needed

# load Pandas
import pandas as pd 

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
# Voting Ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# Load numpy

import numpy as np

# In[1]:
######################################################
# read in data
######################################################

# Set random seed
np.random.seed(0)

data_all = pd.read_csv("./data/louis_users_all_features_label_1213_addtext.csv", encoding='ISO-8859-1')

#instance_data = data_all[ ['freqWeekDay' ,'freqWeekend' ,'freqMoring' ,'freqNoon'
#      ,'freqAfternoon' ,'freqNight' ,'freqLateNight' ,'freqJan' ,'freqFeb' ,'freqMar' 
#      ,'freqApr' ,'freqMay','freqJun' ,'freqJul' ,'freqAug' ,'freqSep' ,'freqOct' 
#      ,'freqNov' ,'freqDec' ,'frequency_1','frequency_2' ,'frequency_3' ,'frequency_4' 
#      ,'frequency_5' ,'frequency_6' ,'tralength_1_2' ,'tralength_1_3' ,'tralength_1_4'
#      ,'tralength_1_5' ,'tralength_1_6' ,'tralength_2_3' ,'tralength_2_4' ,'tralength_2_5'
#      ,'tralength_2_6' ,'tralength_3_4','tralength_3_5' ,'tralength_3_6' ,
#      'tralength_4_5' ,'tralength_4_6' ,'tralength_5_6']]

instance_data = data_all[ ['freqWeekDay','freqWeekend','freqMoring','freqNoon'
      ,'freqNight','freqJan' ,'freqFeb' ,'freqMar' 
      ,'freqMay','freqJun' ,'freqAug' ,'freqSep' ,'freqOct' 
      ,'frequency_1','frequency_2' ,'frequency_4' 
      ,'frequency_5' ,'frequency_6' ,'tralength_1_2' ,'tralength_1_3' ,'tralength_1_4'
      ,'tralength_1_5' ,'tralength_1_6' ,'tralength_2_3' ,'tralength_2_4' ,'tralength_2_5'
      ,'tralength_2_6' ,'tralength_3_4','tralength_3_5' ,'tralength_3_6','tralength_4_5'
      ,'tralength_4_6' ,'tralength_5_6','wtralength_1_2','wtralength_1_3','wtralength_1_4'
      ,'wtralength_1_5','wtralength_1_6','wtralength_2_3','wtralength_2_4','wtralength_2_5'
      ,'wtralength_2_6','wtralength_3_4','wtralength_3_5','wtralength_3_6','wtralength_4_5'
      ,'wtralength_4_6','wtralength_5_6','O','M','F'] ]

# deal with NA
instance_data = instance_data.fillna(0)

# adding the gender as the label 
X = instance_data
Y = data_all['gender']

# In[2]:
######################################################
# Bagged Decision Trees for Classification
######################################################

seed = 7
kfold = model_selection.KFold(n_splits=5, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 140
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

# In[3]:
######################################################
# Random Forest Classification
######################################################
seed = 0
num_trees = 2000
max_features = 50
kfold = model_selection.KFold(n_splits=5, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

# In[4]:
######################################################
# Random Forest Classification
######################################################

seed = 0
num_trees = 100
max_features = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

# In[5]:
######################################################
# AdaBoost Classification
######################################################

seed = 0
num_trees = 100
max_features = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

# In[6]:
######################################################
# Stochastic Gradient Boosting Classification
######################################################

seed = 7
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

# In[7]:
######################################################
# Voting Ensemble for Classification
######################################################

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())