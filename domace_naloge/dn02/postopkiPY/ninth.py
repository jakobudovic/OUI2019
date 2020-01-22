#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 9. naloga
# Source:
# https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv("data/clu/124.csv") 

data
X = data.values


# In[56]:


centroids = np.array([[44,5],[79,-38]])
centroids


# In[57]:


plt.scatter(X[ : , 0], X[ : , 1], c='b')
plt.scatter(centroids[0][0], centroids[0][1], s=200, c='g', marker='s')
plt.scatter(centroids[1][0], centroids[1][1], s=200, c='r', marker='s')
plt.show()


# In[50]:


Kmean = KMeans(n_clusters=2, max_iter=100, init=centroids, n_init=1)
Kmean.fit(data)
Kmean.cluster_centers_


# In[48]:


first = Kmean.cluster_centers_[0]
second = Kmean.cluster_centers_[1]
Kmean.n_iter_
first
second


# In[49]:


plt.scatter(X[ : , 0], X[ : , 1], s =50, c='b')
plt.scatter(first[0], first[1], s=200, c='g', marker='s')
plt.scatter(second[0], second[1], s=200, c='r', marker='s')
plt.show()


# In[51]:


Kmean.labels_


# In[53]:


import collections
collections.Counter(Kmean.labels_)


# In[55]:


# primer napovedovanja/klasifikacije točke
sample_test=np.array([-3.0,-3.0])
second_test=sample_test.reshape(1, -1)
Kmean.predict(second_test)

