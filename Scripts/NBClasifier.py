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
from scipy.sparse.csr import csr_matrix
from scipy.sparse import coo_matrix, vstack


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
#print(nb.score(x_test, y_test))
#cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
nb=MultinomialNB();
cv = ShuffleSplit(n_splits=2, test_size=0.5, random_state=0)
scores = cross_val_score(nb, x, y, cv=cv)#, scoring='f1_macro')
nb.fit(x,y)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#for i in range(len(new1)):
#    P = nb.predict_proba(x[i])[0]
#    print(i)
#    print(P)
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