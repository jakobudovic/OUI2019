#!/usr/bin/env python
# coding: utf-8

# In[50]:


# 6. naloga 
# Source:
# https://towardsdatascience.com/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a
# https://medium.com/@svanillasun/how-to-deal-with-cross-validation-based-on-knn-algorithm-compute-auc-based-on-naive-bayes-ff4b8284cff4


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataAll = pd.read_csv("data/reg/181.csv") 

data = dataAll.head(30)
data
data.shape


# In[41]:


"""
# we separate X and y values in 2 tables
X = data.drop(columns=['Y'])
y = y = data['Y']
X
y
"""
X = data[['X1', 'X2', 'X3', 'X4', 'X5']]
y = data['Y']


# In[1]:


from sklearn.model_selection import cross_val_score
import numpy as np
#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=3)
#train model with cv of 5 
cv_scores = cross_val_score(knn_cv, X, y, cv=5)
#print each cv score (accuracy) and average them
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))

