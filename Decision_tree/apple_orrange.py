# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 19:32:04 2019

@author: SATWIKRAM.K
"""
# imorting libraies
import numpy as np
import pandas as pd

# defining the datasets

features = [[100, 1], [140, 1], [150,0],[170,0]] # 1 = apple, 0 = orange
labels = [1,1,0,0];

#importing tree
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(features,labels)

result =  classifier.predict([[190,0]])

if result == 1:
    print("apple")
    
else:
        print("orange")
        
