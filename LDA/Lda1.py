# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 20:12:29 2019

@author: SATWIKRAMK.K

"""
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataets
dataset = pd.read_csv('Wine.csv')
x = dataset.iloc[:,:13].values
y = dataset.iloc[:,13].values

#splitting the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#applying lDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
x_train = lda.fit_transform(x_train,y_train)
x_test = lda.transform(x_test)

#applying logistic algorithm
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(random_state = 0)
regressor.fit(x_train,y_train)

#predicting the y 
y_pred = regressor.predict(x_test)
