#!/bin/python

import pandas as pd

# Importing Data
dataFileName = "louis_tweets_1119.csv"
data = pd.read_csv(dataFileName,encoding = "ISO-8859-1")

######################################
# Data Cleaning 
######################################

#print data.columns.values

# Only keep all the user whose twitter number is larger than 10

data_reduce = data
userTwitterNum = data["user_id"].value_counts().to_dict()

for Twitter_id,Twitter_num in userTwitterNum.items():
    if (Twitter_num < 10):
        data_reduce = data_reduce[data_reduce.user_id != Twitter_id]


# Output the number of valid users
userTwitterNumGT10 = data_reduce["user_id"].value_counts().to_dict()

with open('user_idGT10_2522.csv', 'w') as f:
    [f.write('{0},{1}\n'.format(key, value)) for key, value in userTwitterNumGT10.items()]

data_reduce.to_csv('louis_tweets_1119_reduce.csv', header=True, encoding = "ISO-8859-1")