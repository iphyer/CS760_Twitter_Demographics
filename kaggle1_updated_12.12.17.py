#!/usr/bin/python

#coding: utf-8

# In[1]:

import random
import datetime
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
# function for encoding categories
from sklearn.preprocessing import LabelEncoder


# In[3]:


# have to use latin1 even though it results in a lot of dead characters
#import os
# os.getcwd()


# In[4]:


twigen = pd.read_csv("louis_tweets_1127_labeled.csv", encoding='ISO-8859-1')
twigen.head()


# In[5]:


def normalize_text(s):
    # just in case
    s = str(s)
    s = s.lower()
    s = re.sub('http://.*',"", s)
    s = re.sub('\@.*?\ ', "", s)
    s = re.sub('\#', "", s)
    
    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W\s',' ',s)
    
    # make sure we didn't introduce any double spaces
    s = re.sub('\s+',' ',s)
    s = re.sub('http://.*',"", s)
    s = re.sub('\@.*?\ ', "", s)
    return s


# In[6]:


twigen['text_norm'] = [normalize_text(s) for s in twigen['content']]
twigen['gender'].replace('F','female',inplace=True)
twigen['gender'].replace('M','male',inplace=True)
twigen['gender'].replace('O','brand',inplace=True)
twigen['gender']=twigen['gender'].fillna("")
#twigen['description_norm'] = [normalize_text(s) for s in twigen['description']]
new1 = []
new2 = []
s = ""
id = twigen['user_id'].get(0)
gender = twigen['gender'].get(0)
for i in range(len(twigen['user_id'])):
    if (twigen['user_id'][i] == id):
        s += twigen['text_norm'][i]
    else:
        new1.append(s)
        new2.append(gender)
        s = ""
        id = twigen['user_id'][i]
        gender = twigen['gender'][i]
new1.append(s)
new2.append(gender)
newtwigen = pd.Series(new1)
newgender = pd.Series(new2)
#print(twigen['text_norm'])
# In[7]:


# twigen['text_norm'].head()


# In[ ]:


# pull the data into vectors
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(newtwigen)
encoder = LabelEncoder()
#print(newgender)
y = encoder.fit_transform(newgender)
# =============================================================================
# # split into train and test sets
#
# # take a look at the shape of each of these
#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)
# =============================================================================
weigh = []
learning_rate = 0.1
l1 = len(x.toarray()[0])
for i in range(l1):
    random.seed(datetime.datetime.now())
    initial = random.random()
    weigh.append(initial)
x = x.astype(float)
# =============================================================================
# # split into train and test sets
x_res, x_1, y_res, y_1 = train_test_split(x, y, test_size = 0.2)
x_train = x_res
y_train = y_res
x_res, x_2, y_res, y_2 = train_test_split(x_res, y_res, test_size = 0.25)
x_res, x_3, y_res, y_3 = train_test_split(x_res, y_res, test_size = 1/3)
x_5, x_4, y_5, y_4 = train_test_split(x_res, y_res, test_size = 0.5)
xlist=[x_1, x_2, x_3, x_4, x_5]
ylist=[y_1, y_2, y_3, y_4, y_5]

epochnum = 5
for n in range(epochnum):
    count = 0
    for m in range(epochnum): 
        if m != n:
            for k in range(len(xlist[m].toarray())):
                x_train[count] = xlist[m][k]
                y_train[count] = ylist[m][k]
                count += 1
    x_test = xlist[n]
    y_test = ylist[n]
    l2 = len(x_test.toarray())
    nb = MultinomialNB()
    nb.fit(x_train, y_train)
# =============================================================================
#update weigh
    weighed_x = x_test
    for i in range(l2):
        order = y_test[i]
        for j in range(l1):
            if x[i, j] != 0:
                weighed_x[i,j] = weigh[j]*x[i,j]
        P = nb.predict_proba(weighed_x[i])[0]
        O = P[order]
        T = 1
        error = T - O
        for j in range(3):
            if (j != order):
                error += -1*P[j]
        error = error/3
        for j in range(l1):
            weigh[j] += learning_rate*x[i,j]*error
        print(i)
        print(P)
        print(error)
weighed_x = x
for i in range(len(new1)):
    for j in range(l1):
            if x[i, j] != 0:
                weighed_x[i,j] = weigh[j]*x[i,j]
    P = nb.predict_proba(weighed_x[i])[0]
    print(i)
    print(P)
#print(nb.score(x_test, y_test))
#cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
#cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
#scores = cross_val_score(nb, weighed_x, y, cv=cv)#, scoring='f1_macro')
#print(scores)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# =============================================================================
#twigen['all_features'] = twigen['text_norm'].str.cat(twigen['user_fname'], sep=' ')
#vectorizer = CountVectorizer()
#x = vectorizer.fit_transform(twigen['text_norm'])
#
#encoder = LabelEncoder()
#y = encoder.fit_transform(twigen['gender'])
#
## split into train and test sets
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#
#nb = MultinomialNB()
#nb.fit(x_train, y_train)
#
#print(nb.score(x_test, y_test))
# =============================================================================
#output to NN and obtain T - O from NN
#for i in range(l1):
#    sum = 0
#    for j in range(l2):
#        sum = sum + x[j][i]
#    aver = sum/l2
#    weigh[i] += learning_rate*aver#*(T - O)
