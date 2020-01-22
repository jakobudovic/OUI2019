#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 3. naloga
# Source:
# https://stackabuse.com/linear-regression-in-python-with-scikit-learn/


# In[37]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data/reg/91.csv") 

data


# In[44]:


X = data[['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6']]
y = data['Y']


# In[45]:


# divide data in training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[46]:


# fit data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)


# In[48]:


# print coefficients
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
coeff_df

