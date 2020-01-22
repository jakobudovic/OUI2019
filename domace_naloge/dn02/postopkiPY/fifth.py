#!/usr/bin/env python
# coding: utf-8

# In[3]:


# 5. naloga MULTIPLE LINEAR REGRESSION
# Source:
# https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f


# In[167]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataAll = pd.read_csv("data/reg/22.csv") 

data = dataAll[:53]
test = dataAll[53:]


# In[168]:


x_train = data[['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']]
y_train = data['Y']
x_test = test[['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']]
y_test = test['Y']


# In[169]:


# training the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  
regressor.fit(x_train, y_train)


# In[170]:


# calculating coefficients
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
coeff_df


# In[171]:


# predict y and compare results with original/actual y values
y_pred = regressor.predict(x_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# In[178]:


# different types of errors:
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:




