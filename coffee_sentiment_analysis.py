#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 16:31:03 2018

@author: yusheng
"""
import json

#with open('tweet_stream_coffee_1307.json','r') as f1:
    #t1 = json.load(f1)

#with open('tweet_stream_coffee_2222.json','r') as f2:
    #t2 = json.load(f2)

#with open('tweet_stream_coffee_2668.json','r') as f3:
    #t3 = json.load(f3)
    
#with open('tweet_stream_coffee_4663.json','r') as f4:
    #t4 = json.load(f4)
    
#with open('tweet_stream_coffee_9200.json','r') as f5:
    #t5 = json.load(f5)

#with open('tweet_stream_coffee_5.json','r') as f6:
    #t6 = json.load(f6)
    
#T = t1 + t2 + t3 + t4 + t5 + t6

#with open('tweet_stream_coffee_20065.json','w') as F:
    #json.dump(T,F,indent = 4)

#store all texts of tweets
with open('tweet_stream_coffee_20065.json','r') as f:
    list_of_t = json.load(f)

str_of_all_t = ''
for t in list_of_t:
    str_of_all_t += '  {}'.format(t['text'])


#remove punctuation and digits
import string
p = string.punctuation
d = string.digits
#string concatenation
p_d = p + d

p_d_table = str.maketrans(p_d,len(p_d)*' ')

without_p_d = str_of_all_t.translate(p_d_table)

#stemming and lemmatization
import nltk
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

after_l = wnl.lemmatize(without_p_d)

#tokenize text file with nltk
after_token = nltk.word_tokenize(after_l.lower())
#tokenized string becomes a list of words

with open('coffee_tweets_20065.txt','w') as infile:
    infile.write(after_l)
    
#remove stopwords
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(['rt','https','coffee','co','bts','twt'])
without_stop = [w for w in after_token if w not in stopwords and len(w) > 1]
#for w in after_token:
    #if w not in stopwords and len(w) > 1:
        #without_stop.append(w)

#get frequency of words
freq = nltk.FreqDist(without_stop)
freq.plot(30)

#wordcloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
#without_stop is a list, so need to remove stopwords again in the text file
from os import path
from PIL import Image
import numpy as np

d = path.dirname(__file__)

text = open(path.join(d,'coffee_tweets_20065.txt')).read()

coffee_mask = np.array(Image.open(path.join(d,'coffee_cup.jpg')))

stopwords2 = set(STOPWORDS)
stopwords2.update(['rt','https','coffee','co','bts','twt'])

wc = WordCloud(background_color='white',max_words=2000,mask=coffee_mask,stopwords=stopwords2)
wc.generate(text)

wc.to_file(path.join(d,'coffee.jpg'))

plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.figure()
plt.imshow(coffee_mask,cmap=plt.cm.gray,interpolation='bilinear')
plt.axis('off')
plt.show()

#sentiment analysis
from textblob import TextBlob
with open('coffee_tweets_20065.txt','r') as infile:
    content = infile.read()
    sentences = content.split('\n')
    
sub_list = []
pol_list = []
for s in sentences:
    tb = TextBlob(s)
    sub_list.append(tb.sentiment.subjectivity)
    pol_list.append(tb.sentiment.polarity)

plt.hist(sub_list,bins=10)
plt.xlabel('subjectivity score')
plt.ylabel('sentence count')
plt.grid(True)
plt.savefig('subjectivity.pdf')
plt.show()

plt.hist(pol_list,bins=10)
plt.xlabel('polarity score')
plt.ylabel('sentence count')
plt.grid(True)
plt.savefig('polarity.pdf')
plt.show()
