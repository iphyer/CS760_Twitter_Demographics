#!/bin/python

import pandas as pd
import numpy as np
from textblob import TextBlob
#import requests
#import time
#import nltk
#from nltk.corpus import treebank
#import sentiment_mod as s
data = pd.read_csv("louis_tweets_cleaned_new_1031.csv",encoding = "ISO-8859-1")
#data = pd.read_csv("testdata.csv",encoding = "ISO-8859-1")

#print data["content"]

#t = treebank.parsed_sents('wsj_0001.mrg')[0]
#t.draw()
#a = nltk.sentiment.util.demo_tweets(trainer, n_instances=None, output=None)

#print(data["content"])

'''
url     = 'http://text-processing.com/api/sentiment/'
txtTwitter = [('text', 'I am great!'),]
#headers = {}
res = requests.post(url,data = txtTwitter)
#print(res.headers['content-type'])
a = res.json()
#print(type(a))
print(a["label"])

'''
emotions = []
#url     = 'http://text-processing.com/api/sentiment/'

for item in data["content"].iteritems():
	txtTwitter = item[1]
	#time.sleep(3)
	polarity = round(TextBlob(txtTwitter).sentiment.polarity,4)
	#res = requests.post(url,data = txtTwitter)
	emotions.append(polarity)
	print(item[0])

arr = np.asarray(emotions)
data["emotions"] = arr
np.savetxt("foo.csv", arr, delimiter=",")

#print(emotions)

#data.to_csv("louis_tweets_cleaned_new_1031_emotions.csv",encoding = "ISO-8859-1")

#data2 = pd.read_csv("louis_tweets_cleaned_new_1031.csv",encoding = "ISO-8859-1")