#!/usr/bin/env python
# coding: utf-8

# In[159]:


# 6. naloga 
# Source:
# https://medium.com/@svanillasun/how-to-deal-with-cross-validation-based-on-knn-algorithm-compute-auc-based-on-naive-bayes-ff4b8284cff4


# In[160]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataAll = pd.read_csv("data/reg/75.csv") 

data = dataAll.head(30)
data
data.shape


# In[165]:


# X = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
# X = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']]
X = data[['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6']]
# X = data[['X1', 'X2', 'X3', 'X4', 'X5']]
y = data['Y']


# In[162]:


# import k-folder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
# from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
k_range = range(1, 29, 2)
k_scores = []

for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    loss = abs(cross_val_score(knn, X, y, cv=10, scoring='neg_mean_squared_error'))
    k_scores.append(loss.mean())
    k
    loss.mean()

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated MSE')
plt.show()


# In[163]:


# training the algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  
regressor.fit(X, y)


# In[ ]:




