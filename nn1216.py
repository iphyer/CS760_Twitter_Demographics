# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 16:36:55 2017

@author: Myron
"""
# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 
import numpy as np
# function to split the data for cross-validation
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report,confusion_matrix


# In[3]:


# have to use latin1 even though it results in a lot of dead characters
#import os
# os.getcwd()


# In[4]:


data_all = pd.read_csv("./data/louis_users_all_features_label_1213_addtext.csv", encoding='ISO-8859-1')

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

X = instance_data
y = data_all['gender']

# In[5]:

#data_all['OO'] = data_all['O'].astype('float64')
#data_all['MM'] = data_all['M'].astype('float64')
#data_all['FF'] = data_all['F'].astype('float64')
#nb_data = [data_all['OO'],data_all['MM'],data_all['FF']]
nb_data = data_all[['O','M','F']]
#nb_data = np.asfarray(nb_data0,float)
nb_data['max'] = nb_data.idxmax(axis=1)
print(confusion_matrix(y,nb_data['max']))
print(classification_report(y,nb_data['max']))

# In[6]:
# split training and testing dateset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# In[7]:
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# In[8]:
mlp = MLPClassifier(activation='relu', alpha=1e-3, batch_size=1, hidden_layer_sizes=(35,10,5,5,5),
                    learning_rate_init=0.002, max_iter=1200, shuffle=True, solver='lbfgs')
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
#
#for index, row in df.iterrows():
## for x in df[:]:
#    X.append(row.tolist())

#clf = MLPClassifier(solver='sgd', alpha=0.0001, hidden_layer_sizes=(5, 2), random_state=1)

#clf.fit(x_train, y_train)                         
#MLPClassifier(activation='logistic', alpha=1e-05, batch_size=10,
#       beta_1=0.9, beta_2=0.999, early_stopping=False,
#       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
#       learning_rate_init=0.001, max_iter=200, momentum=0.9,
#       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
#       solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=False,
#       warm_start=False)

#clf.predict(x_train)

