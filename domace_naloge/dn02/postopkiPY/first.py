#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1. naloga
# Source:
# 
# ? https://towardsdatascience.com/why-feature-correlation-matters-a-lot-847e8ba439c4 


# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data/reg/150.csv") 
# data = pd.read_csv("data/reg/135.csv")

data


# In[41]:


data[['X2', 'Y']].corr()

