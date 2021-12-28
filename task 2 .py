#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


dataset = pd.read_csv(r"C:\Users\Sameh Zaher\Desktop\New folder\Wuzzuf_Jobs.csv")
dataset.head()


# In[5]:


dataset["YearsExp"].value_counts()


# In[7]:


dataset["Factorize-YearsExp"] =pd.factorize(dataset["YearsExp"])[0]
dataset.head()


# In[8]:


dataset["Factorize-Title"] =pd.factorize(dataset["Title"])[0]
dataset["Factorize-Company"] =pd.factorize(dataset["Company"])[0]
dataset.head()


# In[9]:


data = dataset.iloc[: ,[-2,-1]]
from sklearn.cluster import KMeans
wcss=[]
for i in  range (1 ,11):
    km = KMeans(n_clusters = i ,init ='k-means++' ,random_state = 42 ,max_iter = 500)
    km.fit(data)
    wcss.append(km.inertia_)
plt.plot(range(1,11) , wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


# In[10]:


km =KMeans(n_clusters =2 , init ='k-means++' ,random_state = 42)
y_mean = km.fit_predict(data)
y_mean


# In[11]:


data['target'] = y_mean 
data.head() 


# In[13]:


plt.scatter(data.iloc[:,0] , data.iloc[:,1] , color = 'yellow')
plt.show()


# In[14]:


#filter rows of original data
filtered_label0 = data.loc[data.target == 0]
 
filtered_label1 = data.loc[data.target == 1]
 
#Plotting the results
plt.figure(figsize =(20 ,8))
plt.scatter(filtered_label0.iloc[:,0] , filtered_label0.iloc[:,1] , color = 'red' , s=50 , label ='cluster 1')
plt.scatter(filtered_label1.iloc[:,0] , filtered_label1.iloc[:,1] , color = 'black' , s=50 ,label = 'cluster 2')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s =500, c = 'yellow', label = 'Centroids')
plt.xlabel("Jobs Title")
plt.ylabel("Company")
plt.title("Clusters of employees")
plt.legend()
plt.show()


# In[ ]:




