# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 01:47:13 2020

@author: PC
"""

import os

os.chdir('C:/Users/PC/Desktop/Project/Spam-NLP')

import pandas as pd

spam=pd.read_csv('SMSSpamCollection',sep='\t',names=['labels','message'])

import nltk
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

stemmer=PorterStemmer()

corpus=[]
for i in range(0,len(spam)):
    review=re.sub('[^a-zA-Z]',' ', spam['message'][i])
    review=review.lower()
    review=review.split()
    review=[stemmer.stem(word) for word in review if not word in set(stopwords.words('english')) ]
    review=' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
X_boW=cv.fit_transform(corpus).toarray()

y=pd.get_dummies(spam['labels'])

y=y.iloc[:,1].values


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X_boW,y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import MultinomialNB

spam_detactor_model=MultinomialNB().fit(X_train,y_train)

y_pred=spam_detactor_model.predict(X_test)

from sklearn.metrics import confusion_matrix

conf_mt=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score

score=accuracy_score(y_test,y_pred)

print(score)

######################################################################

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()