# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 16:36:55 2017

@author: Myron
"""
# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 
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


data_all = pd.read_csv("./data/louis_users_all_features_label_1207_updated.csv", encoding='ISO-8859-1')

instance_data = data_all[ ['freqWeekDay' ,'freqWeekend' ,'freqMoring' ,'freqNoon'
      ,'freqAfternoon' ,'freqNight' ,'freqLateNight' ,'freqJan' ,'freqFeb' ,'freqMar' 
      ,'freqApr' ,'freqMay','freqJun' ,'freqJul' ,'freqAug' ,'freqSep' ,'freqOct' 
      ,'freqNov' ,'freqDec' ,'frequency_1','frequency_2' ,'frequency_3' ,'frequency_4' 
      ,'frequency_5' ,'frequency_6' ,'tralength_1_2' ,'tralength_1_3' ,'tralength_1_4'
      ,'tralength_1_5' ,'tralength_1_6' ,'tralength_2_3' ,'tralength_2_4' ,'tralength_2_5'
      ,'tralength_2_6' ,'tralength_3_4','tralength_3_5' ,'tralength_3_6' ,
      'tralength_4_5' ,'tralength_4_6' ,'tralength_5_6'] ]

# deal with NA

instance_data = instance_data.fillna(0)

X = instance_data
y = data_all['gender']


# split training and testing dateset
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(learning_rate='adaptive',activation='logistic', alpha=1e-1,hidden_layer_sizes=(100,200,300,200,100))
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
#       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='adaptive',
#       learning_rate_init=0.001, max_iter=200, momentum=0.9,
#       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
#       solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=False,
#       warm_start=False)

#clf.predict(x_train)

