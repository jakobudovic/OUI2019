#!/usr/bin/env python
# coding: utf-8

# In[183]:


# 2. naloga
# Source:
# https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9
# https://stackabuse.com/linear-regression-in-python-with-scikit-learn/


# In[203]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data = pd.read_csv("data/clu/41.csv") 
data = pd.read_csv("data/reg/116.csv") 

data.describe()
data


# In[204]:


# show data
data.plot(x='X1', y='Y', style='o')
plt.title('graf 2')
plt.xlabel('x os')
plt.ylabel('y os')
plt.show()


# In[206]:


# prepare data
# X = data.iloc[:, :-1].values
# y = data.iloc[:, 1].values

X = data[['X0', 'X1', 'X2', 'X3', 'X4', 'X5']]
y = data['Y']
print(X)
print("--------------------------------")
print(y)


# In[207]:


# train data
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X,y)


# In[208]:


print(reg.coef_)


# In[209]:


# Visualisation
coeff_df = pd.DataFrame(reg.coef_, X.columns, columns=['Coefficient'])
coeff_df

