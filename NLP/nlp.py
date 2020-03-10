# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:36:00 2019

@author: SATWIKRAM.K
"""
# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datsets
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter ='\t',quoting = 3)

#cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', '  ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review =[ps.stem(word) for  word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#craeting the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

#importing train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = (0.20), random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

test_data = pd.read_excel('review.xlsx',delimiter ='\t',quoting = 3)

#cleaning the texts
"""import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer"""
corpus1 = []
for i in range(0,2):
    test_data = re.sub('[^a-zA-Z]', '  ', dataset['test_data'][i])
    test_data = test_data.lower()
    test_data = test_data.split()
    ps = PorterStemmer()
    test_data =[ps.stem(word) for  word in test_data if not word in set(stopwords.words('english'))]
    test_data = ' '.join(test_data)
    corpus1.append(test_data)
    
x1 = cv.fit_transform(corpus1).toarray()

    



# Predicting the Test set results
y_pred = classifier.predict(test_data)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)















