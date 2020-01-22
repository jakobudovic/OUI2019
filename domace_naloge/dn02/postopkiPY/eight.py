#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 8. naloga
# Source:
# https://towardsdatascience.com/machine-learning-algorithms-part-12-hierarchical-agglomerative-clustering-example-in-python-1e18e0075019


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data/clu/155.csv") 

data
X = data.values


# In[3]:


import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='average'))

plt.title('Dendrogram')
plt.ylabel('Euclidean Distance')
plt.show()


# In[6]:


from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='average')
# model.fit(X)
model.fit_predict(X)

labels = model.labels_
labels


# In[7]:


plt.scatter(X[labels==0, 0], X[labels==0, 1], s=50, marker='o', color='red')
plt.scatter(X[labels==1, 0], X[labels==1, 1], s=50, marker='o', color='blue')
"""
# plt.scatter(X[labels==, 0], X[labels==, 1], s=50, marker='o', color='')
plt.scatter(X[labels==2, 0], X[labels==2, 1], s=50, marker='o', color='green')
plt.scatter(X[labels==3, 0], X[labels==3, 1], s=50, marker='o', color='purple')
plt.scatter(X[labels==4, 0], X[labels==4, 1], s=50, marker='o', color='orange')
plt.scatter(X[labels==5, 0], X[labels==5, 1], s=50, marker='o', color='yellow')
plt.scatter(X[labels==6, 0], X[labels==6, 1], s=50, marker='o', color='black')
plt.scatter(X[labels==7, 0], X[labels==7, 1], s=50, marker='o', color='cyan')
plt.scatter(X[labels==8, 0], X[labels==8, 1], s=50, marker='o', color='brown')
plt.scatter(X[labels==9, 0], X[labels==9, 1], s=50, marker='o', color='brown')
plt.scatter(X[labels==10, 0], X[labels==10, 1], s=50, marker='o', color='brown')
"""
plt.show()


# In[8]:


import collections
collections.Counter(labels)

