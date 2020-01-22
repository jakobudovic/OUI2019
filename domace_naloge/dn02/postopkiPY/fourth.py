#!/usr/bin/env python
# coding: utf-8

# In[12]:


# 4. naloga TO - DO ?
# Source:
# https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f


# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataAll = pd.read_csv("data/reg/181.csv") 

data = dataAll.head(55)
data


# In[30]:


X = data[['X1', 'X2', 'X3', 'X4', 'X5']]
y = data['Y']

X
y


# In[31]:


# training the algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  
regressor.fit(X, y)


# In[32]:


# just for the feeling
#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)


# In[33]:


# predict
y_pred = regressor.predict(X)


# In[34]:


from sklearn import metrics
print('Mean Squared Error aka MSE:', metrics.mean_squared_error(y, y_pred))  


# In[ ]:




