# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 16:36:55 2017

@author: sheen
"""


# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# we'll want this for plotting
import matplotlib.pyplot as plt

# we'll want this for text manipulation
import re

# for quick and dirty counting
from collections import defaultdict

# the Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
# function to split the data for cross-validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
# function for cross validation
from sklearn.model_selection import cross_val_score
# function for transforming documents into counts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
# function for encoding categories
from sklearn.preprocessing import LabelEncoder


# In[3]:


# have to use latin1 even though it results in a lot of dead characters
#import os
# os.getcwd()


# In[4]:


twigen = pd.read_csv("louis_users_all_features_label_1207_updated.csv", encoding='ANSI')
# names = ["user_key", "user_id", "user_account", "user_name", "user_location", "user_profile", "user_start_time", "user_fname", "user_lname", "user_mname", "gender", "user_follower", "user_tweet_count", "freqWeekDay"]

twigen.head()


# In[5]:


twigen['gender'].head()
#twigen['description_norm'] = [normalize_text(s) for s in twigen['description']]
twigen2 = [twigen['freqWeekDay'], twigen['freqWeekend'], twigen['freqMoring'], twigen['freqNoon'], 
     twigen['freqAfternoon'], twigen['freqNight'], twigen['freqLateNight'], twigen['freqJan'], twigen['freqFeb'], twigen['freqMar'], twigen['freqApr'], twigen['freqMay'], 
     twigen['freqJun'], twigen['freqJul'], twigen['freqAug'], twigen['freqSep'], twigen['freqOct'], twigen['freqNov'], twigen['freqDec'], twigen['frequency_1'], 
     twigen['frequency_2'], twigen['frequency_3'], twigen['frequency_4'], twigen['frequency_5'], twigen['frequency_6'], twigen['tralength_1_2'], twigen['tralength_1_3'], twigen['tralength_1_4'], 
     twigen['tralength_1_5'], twigen['tralength_1_6'], twigen['tralength_2_3'], twigen['tralength_2_4'], twigen['tralength_2_5'], twigen['tralength_2_6'], twigen['tralength_3_4'],
     twigen['tralength_3_5'], twigen['tralength_3_6'], twigen['tralength_4_5'], twigen['tralength_4_6'], twigen['tralength_5_6']]
df = pd.DataFrame(twigen2)
df = df.T
X = list()

for index, row in df.iterrows():
# for x in df[:]:
    X.append(row.tolist())
    
encoder = LabelEncoder()
#print(twigen['gender'])
twigen['gender']=twigen['gender'].fillna("")
y = encoder.fit_transform(twigen['gender'])    

# In[6]:

# pull the data into vectors
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[7]:
# =============================================================================

clf = MLPClassifier(solver='sgd', alpha=0.0001, hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(x_train, y_train)                         
MLPClassifier(activation='logistic', alpha=1e-05, batch_size=10,
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

clf.predict(x_train)

