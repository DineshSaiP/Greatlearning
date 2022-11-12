#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings 
warnings.filterwarnings('ignore')

## Visualizing charts
plt.rcParams['figure.figsize']=[15,6]


# ## Train Test Split

# In[4]:


sales=pd.read_csv('~/Downloads/train_XnW6LSF.csv')
sales


# In[5]:


# Library to import Train and test split

from sklearn.model_selection import train_test_split


# In[9]:


# split the data in X and y
X=sales.drop('Item_Outlet_Sales',axis=1)
y=sales.Item_Outlet_Sales


# In[11]:


# Split the X and Y in train and test
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.20,random_state=0)


# In[12]:


print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(xtest.shape)


# ### Transformations 
# 
# * Log Transformation => It does not work on 0 or -ve values
# * Square root Transformation => It cannot work on -ve numbers
# * Power Transformation (1/x) => It cannot work on 0's
# * Boxcox Transformation => Only +ve numbers
# * Yeo Johnson Transformation => Can work on any number
# 
# Remember that the purpose of the transformation is to reduce the skewness & trying to make the data normal
# 
# You cannot make the data perfectly normal, however, you can reduce the skewness a little bit

# In[16]:


# Apply the Transformation and check the skewness post transformation

# Log Transformation 

import scipy.stats as stats

print('Before Transformation:',sales.Item_Outlet_Sales.skew())
print('After Transformation:',np.log(sales.Item_Outlet_Sales.skew()))


# In[17]:


# Square Root Transformation 

print('Before Transformation:',sales.Item_Outlet_Sales.skew())
print('After Transformation:',np.sqrt(sales.Item_Outlet_Sales.skew()))


# In[20]:


# Power Transformation 

print('Before Transformation:',sales.Item_Outlet_Sales.skew())
print('After Transformation:',(1/sales.Item_Outlet_Sales).skew())

# So if the data is -vely skewed, we can try Power Transformation to reduce the skewness in the -ve direction


# In[28]:


# Boxcox Transformation 

print('Before Transformation:',sales.Item_Outlet_Sales.skew())
print('After Transformation:',pd.Series(stats.boxcox(sales.Item_Outlet_Sales)[0]).skew())


# In[29]:


# Yeo Johnson Transformation 

print('Before Transformation:',sales.Item_Outlet_Sales.skew())
print('After Transformation:',pd.Series(stats.yeojohnson(sales.Item_Outlet_Sales)[0]).skew())


# In[25]:


stats.boxcox(np.array([23,78,0,-500]))


# In[26]:


np.log(np.array([23,78,0,-500]))


# ### Summary
# 
# * Log Transformation,Box Cox and Yeo Johnsom returned -ve skewness
# * Where as Square Root reduced the Skewness for the Target variable
# * In Power Transformation,it increases the skewness and seems like it is a good fit for -vely skewed data
# 
# Note : The purpose of the Transformation is to reduce the skewness. So, if the data is quite close to normal,then we skip this step and directly build the model

# In[32]:


sns.distplot(sales.Item_Outlet_Sales,kde=True,hist=True,color='magenta')
plt.title('Before Transformation')


# In[34]:


sns.distplot(np.sqrt(sales.Item_Outlet_Sales),kde=True,hist=True,color='green')
plt.title('SQRT Transformation')


# In[ ]:




