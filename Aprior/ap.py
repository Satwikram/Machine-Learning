# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 21:26:23 2019

@author: SATWIKRAM.K
"""
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing datasets
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

#forming the list
transactions = []

for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)] ) 
    
#importing apriori model
from apyori import apriori
results = apriori(transactions, min_support, min_confidence, min_lift, min_lenght = 2)

    
    


