
# coding: utf-8

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
# function for encoding categories
from sklearn.preprocessing import LabelEncoder


# In[3]:


# have to use latin1 even though it results in a lot of dead characters
#import os
# os.getcwd()


# In[4]:


twigen = pd.read_csv("louis_tweets_1125_labeled.csv", encoding='ISO-8859-1')
twigen.head()


# In[5]:


def normalize_text(s):
    # just in case
    s = str(s)
    s = s.lower()
    
    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W\s',' ',s)
    
    # make sure we didn't introduce any double spaces
    s = re.sub('\s+',' ',s)
    
    return s


# In[6]:


twigen['text_norm'] = [normalize_text(s) for s in twigen['content']]
#twigen['description_norm'] = [normalize_text(s) for s in twigen['description']]


# In[7]:


# twigen['text_norm'].head()


# In[ ]:


# pull the data into vectors
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(twigen['text_norm'])

encoder = LabelEncoder()
twigen['gender'].replace('F','female',inplace=True)
twigen['gender'].replace('M','male',inplace=True)
twigen['gender'].replace('O','brand',inplace=True)
#print(twigen['gender'])
twigen['gender']=twigen['gender'].fillna("")
y = encoder.fit_transform(twigen['gender'])

# =============================================================================
# # split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# 
# # take a look at the shape of each of these
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# =============================================================================
nb = MultinomialNB()
#nb.fit(x_train, y_train)

#print(nb.score(x_test, y_test))
#cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
scores = cross_val_score(nb, x, y, cv=cv, scoring='f1_macro')
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
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
