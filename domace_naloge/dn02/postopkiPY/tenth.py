#!/usr/bin/env python
# coding: utf-8

# In[20]:


# 10. naloga
# Source:
# 9. naloga
# https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py


# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv("data/clu/41.csv") 

data
X = data.values


# In[22]:


centroids = np.array([[-52,-54],[-67,95]])
centroids


# In[23]:


# začetno stanje
plt.scatter(X[ : , 0], X[ : , 1], s =50, c='b')
plt.scatter(centroids[0][0], centroids[0][1], s=200, c='g', marker='s')
plt.scatter(centroids[1][0], centroids[1][1], s=200, c='r', marker='s')
plt.show()


# In[24]:


Kmean = KMeans(n_clusters=2, max_iter=100, init=centroids, n_init=1)
Kmean.fit(data)
Kmean.cluster_centers_


# In[25]:


first = Kmean.cluster_centers_[0]
second = Kmean.cluster_centers_[1]
Kmean.n_iter_
first
second


# In[26]:


# koncno stanje
plt.scatter(X[ : , 0], X[ : , 1], s =50, c='b')
plt.scatter(first[0], first[1], s=200, c='g', marker='s')
plt.scatter(second[0], second[1], s=200, c='r', marker='s')
plt.show()


# In[ ]:




