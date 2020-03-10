# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 19:27:59 2019

@author: SATWIKRAM.K
"""
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#importing datasets
dataset = pd.read_csv('Wine.csv')
print(dataset.head())

dataset.describe()
dataset.info()

x = dataset.iloc[:,:13 ].values
y = dataset.iloc[:,13].values

#splitting the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
explained_varience = pca.explained_variance_ratio_

#applying logistic regression
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(random_state = 0)
regressor.fit(x_train, y_train)


#predicting the result
y_pred = regressor.predict(x_test)

#calculating accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)