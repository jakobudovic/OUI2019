#!/usr/bin/env python
# coding: utf-8

# In[30]:


# 7. naloga 
# Source:
# Disc: https://discordapp.com/channels/@me/651110088190459973/651110266813153303
# doesn't work properly


# In[31]:


import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer

file = pandas.read_csv("data/reg/78.csv")
file
ucna = file[:40]
testna = file[40:]


# In[32]:


ucna_Y = ucna.iloc[:,8].values
ucna_X = ucna.iloc[:,:-1].values
testna_Y = testna.iloc[:,8].values
testna_X = testna.iloc[:,:-1].values


# In[33]:


opt_k = 20
min = 100000

for i in range(1, 35):
    scores = cross_val_score(KNeighborsRegressor(n_neighbors = i),ucna_X,ucna_Y,         scoring=make_scorer(mean_squared_error), cv=10)
    zdajsnji = scores.mean()
    if (zdajsnji < min) :
        opt_k = i
        min = zdajsnji

    print("{:.3f}\n".format(scores.mean()))
    prejsnji = scores.mean()

reg = KNeighborsRegressor(n_neighbors = opt_k).fit(ucna_X,ucna_Y)
Y_pred = reg.predict(testna_X)

print(mean_squared_error(testna_Y,Y_pred))


# In[ ]:




