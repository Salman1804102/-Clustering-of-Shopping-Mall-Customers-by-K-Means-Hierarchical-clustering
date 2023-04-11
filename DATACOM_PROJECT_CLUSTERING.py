#!/usr/bin/env python
# coding: utf-8

# In[54]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
plt.style.use('Solarize_Light2')


# In[55]:


# Get dataframe from CSV file
df = pd.read_csv('Mall_Customers.csv')
df.head()


# In[56]:


df.shape


# In[57]:


df.describe()


# In[58]:


plt.figure(1, figsize=(16,4))
n = 0 
for i in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
    sns.histplot(df[i] , bins = 32)
    plt.title(f'Histogram of {i}')
plt.show()


# In[59]:


import warnings
warnings.filterwarnings("ignore")


# In[60]:


# Assignment Stage

X1 = df.loc[:, ['Age', 'Spending Score (1-100)']].values
inertia = []
for n in range(1 , 11):
    model = KMeans(n_clusters = n,
               init='k-means++',
               max_iter=500,
               random_state=42)
    model.fit(X1)
    inertia.append(model.inertia_)


# In[61]:


plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()


# In[62]:


model = KMeans(n_clusters = 4,
            init='k-means++',
            max_iter=500,
            random_state=42)
model.fit(X1)
labels = model.labels_
centroids = model.cluster_centers_
y_kmeans = model.fit_predict(X1) 

plt.figure(figsize=(20,10))
plt.scatter(X1[y_kmeans == 0, 0], X1[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X1[y_kmeans == 1, 0], X1[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X1[y_kmeans == 2, 0], X1[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X1[y_kmeans == 3, 0], X1[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X1[y_kmeans == 4, 0], X1[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of Customers - Age X Spending Score')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.legend()
plt.show()


# In[63]:


# Assignment Stage

X2 = df.loc[:, ['Annual Income (k$)', 'Spending Score (1-100)']].values
inertia = []
for n in range(1 , 11):
    model = KMeans(n_clusters = n,
               init='k-means++',
               max_iter=500,
               random_state=42)
    model.fit(X2)
    inertia.append(model.inertia_)

plt.figure(1 , figsize = (20, 10))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()


# In[64]:


model = KMeans(n_clusters = 5,
            init='k-means++',
            max_iter=500,
            random_state=42)
model.fit(X2)
labels = model.labels_
centroids = model.cluster_centers_
y_kmeans = model.fit_predict(X2) 

plt.figure(figsize=(20,10))
plt.scatter(X2[y_kmeans == 0, 0], X2[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X2[y_kmeans == 1, 0], X2[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X2[y_kmeans == 2, 0], X2[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X2[y_kmeans == 3, 0], X2[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X2[y_kmeans == 4, 0], X2[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of Customers - Annual Income (k$) X Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend()
plt.show()


# In[65]:


# Assignment Stage

from sklearn.cluster import KMeans

X3 = df.loc[:, ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
inertia = []
for n in range(1 , 11):
    model = KMeans(n_clusters = n,
               init='k-means++',
               max_iter=500,
               random_state=42)
    model.fit(X3)
    inertia.append(model.inertia_)

plt.figure(1 , figsize = (20, 10))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()


# In[66]:


model = KMeans(n_clusters = 6,
            init='k-means++',
            max_iter=500,
            random_state=42)
model.fit(X3)
labels = model.labels_
#centroids = model.cluster_centers_

df['cluster'] =  labels
df


# In[67]:


fig = px.scatter_3d(df,
                    x="Age",
                    y="Annual Income (k$)",
                    z="Spending Score (1-100)",
                    color='cluster',
                    hover_data=["Age",
                                "Annual Income (k$)",
                                "Spending Score (1-100)"],
                    category_orders = {"cluster": range(0, 5)},
                    )

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()


# In[68]:


import scipy.cluster.hierarchy as sch


# In[69]:


# Visualising the dendrogram
fig = plt.figure(figsize=(25, 10))
dendrogram=sch.dendrogram(sch.linkage(X2,method='ward'))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Eucledian distance")
plt.show()


# In[70]:


from sklearn.cluster import AgglomerativeClustering


# In[71]:


hc=AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")


# In[72]:


y_hc=hc.fit_predict(X2)


# In[73]:


y_hc


# In[74]:


y_hc.astype


# In[75]:


# Visualising the clusters
fig = plt.figure(figsize=(25, 10))
plt.scatter(X2[y_hc == 0, 0], X2[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X2[y_hc == 1, 0], X2[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X2[y_hc == 2, 0], X2[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X2[y_hc == 3, 0], X2[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X2[y_hc == 4, 0], X2[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




