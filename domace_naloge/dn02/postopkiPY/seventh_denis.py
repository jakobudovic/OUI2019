#!/usr/bin/env python
# coding: utf-8

# In[5]:


# @denisdebenis

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

data = pd.read_csv("data/reg/78.csv")
data_attr = data.drop("Y", axis=1)
L = data[:40]
T = data[40:]
L_attr = data_attr[:40]
T_attr = data_attr[40:]

data_labels = data["Y"].copy()
L_labels = data_labels[:40]
T_labels = data_labels[40:]


# In[12]:


attribs = list(data_attr)
ct = ColumnTransformer([("num", "passthrough", attribs)])
L_prepared = ct.fit_transform(L_attr)
T_prepared = ct.transform(T_attr)


# In[17]:



knns=[]
for i in range(1,37):
    knns.append(KNeighborsRegressor(n_neighbors=i))
for knn in knns:
    knn.fit(L_prepared, L_labels)

mse_scores = []
for knn in knns:
    scores = cross_val_score(knn, L_prepared, L_labels, scoring="neg_mean_squared_error", cv=10)
    print(scores)
    mse_scores.append(-scores)

for j in range(0, len(knns)):
    mse_scores[j]=mse_scores[j].mean()


# In[18]:


knn = knns[mse_scores.index(min(mse_scores))]
print(knn.n_neighbors)
T_prediction = knn.predict(T_prepared)
T_mse = mean_squared_error(T_labels, T_prediction)
print("MSE:{:f}".format(T_mse))


# In[ ]:




