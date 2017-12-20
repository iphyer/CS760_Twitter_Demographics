# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 16:36:55 2017

@author: Myron
"""
import os as os
#import pydotplus
# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from IPython.display import Image 

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

data_train = pd.read_csv("./data/trainset.csv", encoding='ISO-8859-1')
data_test = pd.read_csv("./data/testset.csv", encoding='ISO-8859-1')
data_all = pd.read_csv("./data/louis_users_all_features_label_1217_addtext.csv", encoding='ISO-8859-1')
#data_all['OO'] = data_all['O'].astype(float)
#data_all['MM'] = data_all['M'].astype(float)
#data_all['FF'] = data_all['F'].astype(float)

# In[4]:

#instance_data = data_all[ ['user_key','freqWeekDay','freqWeekend','freqMoring','freqNoon'
#      ,'freqNight','freqJan' ,'freqFeb' ,'freqMar' 
#      ,'freqMay','freqJun' ,'freqAug' ,'freqSep' ,'freqOct' 
#      ,'frequency_1','frequency_2' ,'frequency_4' 
#      ,'frequency_5' ,'frequency_6' ,'tralength_1_2' ,'tralength_1_3' ,'tralength_1_4'
#      ,'tralength_1_5' ,'tralength_1_6' ,'tralength_2_3' ,'tralength_2_4' ,'tralength_2_5'
#      ,'tralength_2_6' ,'tralength_3_4','tralength_3_5' ,'tralength_3_6','tralength_4_5'
#      ,'tralength_4_6' ,'tralength_5_6','wtralength_1_2','wtralength_1_3','wtralength_1_4'
#      ,'wtralength_1_5','wtralength_1_6','wtralength_2_3','wtralength_2_4','wtralength_2_5'
#      ,'wtralength_2_6','wtralength_3_4','wtralength_3_5','wtralength_3_6','wtralength_4_5'
#      ,'wtralength_4_6','wtralength_5_6','O','F','M'] ]

instance_data = data_all[ ['freqWeekDay' ,'freqWeekend' ,'freqMoring' ,'freqNoon'
      ,'freqAfternoon' ,'freqNight' ,'freqLateNight' ,'freqJan' ,'freqFeb' ,'freqMar' 
      ,'freqApr' ,'freqMay','freqJun' ,'freqJul' ,'freqAug' ,'freqSep' ,'freqOct' 
      ,'freqNov' ,'freqDec' ,'frequency_1','frequency_2' ,'frequency_3' ,'frequency_4' 
      ,'frequency_5' ,'frequency_6' ,'tralength_1_2' ,'tralength_1_3' ,'tralength_1_4'
      ,'tralength_1_5' ,'tralength_1_6' ,'tralength_2_3' ,'tralength_2_4' ,'tralength_2_5'
      ,'tralength_2_6' ,'tralength_3_4','tralength_3_5' ,'tralength_3_6'
      ,'tralength_4_5' ,'tralength_4_6' ,'tralength_5_6','wtralength_1_2','wtralength_1_3'
      ,'wtralength_1_4','wtralength_1_5','wtralength_1_6','wtralength_2_3','wtralength_2_4'
      ,'wtralength_2_5','wtralength_2_6','wtralength_3_4','wtralength_3_5','wtralength_3_6'
      ,'wtralength_4_5','wtralength_4_6','wtralength_5_6','O','F','M']]

# deal with NA
instance_data = instance_data.fillna(0)

X = instance_data
Y = data_all[['user_key','gender']]

# In[5]:
train_index = data_train['index']
test_index = data_test['index']
X_train = instance_data[instance_data['user_key'].isin(train_index)]
X_test = instance_data[instance_data['user_key'].isin(test_index)]
Y_train = Y[Y['user_key'].isin(train_index)]
Y_test = Y[Y['user_key'].isin(test_index)]
y_train = Y_train['gender']
y_test = Y_test['gender']

# In[5]:
seed = 0
num_trees = 40
max_features = 30
#kfold = model_selection.KFold(n_splits=15, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_depth=20, random_state=seed, max_features=max_features)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

i_tree = 0
for tree_in_forest in model.estimators_:
    with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
        dot_data = tree.export_graphviz(tree_in_forest, out_file = my_file,
                                        filled=True, rounded=True, special_characters=True)
    i_tree = i_tree + 1

# In[ ]:
os.system('dot -Tpng tree_3.dot -o tree_3.png')
os.system('dot -Tpng tree_4.dot -o tree_4.png')
os.system('dot -Tpng tree_5.dot -o tree_5.png')  

# In[ ]:
seed = 7
#kfold = model_selection.KFold(n_splits=5, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 140
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# In[ ]:
seed = 0
num_trees = 100
max_features = 7
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# In[ ]:
seed = 0
num_trees = 100
max_features = 7
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# In[ ]:
seed = 0
num_trees = 100
max_features = 7
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# In[ ]:

mlp = MLPClassifier(activation='relu', alpha=1e-3, batch_size=7, hidden_layer_sizes=(30,10,5,5),
                    learning_rate_init=0.002, max_iter=1200, shuffle=True, solver='lbfgs')
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# In[ ]:

#nb_data = [data_all['OO'],data_all['MM'],data_all['FF']]
nb_data = data_all[['OO','MM','FF']]
name_test = nb_data[nb_data['user_key'].isin(test_index)]
#nb_data = np.asfarray(nb_data0,float)
name_test['max'] = name_test.idxmax(axis=1)
print(confusion_matrix(y_test,name_test['max']))
print(classification_report(y_test,name_test['max']))

# In[6]:
# split training and testing dateset
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# In[7]:
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# In[8]:
mlp = MLPClassifier(activation='relu', alpha=1e-3, batch_size=1, hidden_layer_sizes=(35,20,20,5),
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

