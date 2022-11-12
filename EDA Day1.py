#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Predict the sales of the retail store

### EDA Framework

## 1. Univariate analysis (single variable analysis)
# 1.1. for numerical variable we use - distplot(prefered),histogram,boxplot(used to identify the outliners of the data)
# 1.2.for categorical variable we use - countplot

## 2. Bivariate/Multivariate analysis - it is done wrt target variable
# 2.1. Num vs Num : Scatterplot
# 2.2. Cat vs Num : boxplot/violinplot and barplot
# 2.3. Cat vs Cat : pd.crosstab() to generate the frequency table and then countplot

## 3. Dealing with the missing values

## 4. Outliner analysis and removal
# (hint: we cannot remove outliners in financial data i.e, for sales,revenue,costs)

## 5. ** Feature Engineering**

## 6. Scaling and Transformation

## 7. Categorical encoding

# NOTE : Once the above mentioned steps are done, we split the data in train and test
(hint : target variable never has missing values)


# In[2]:


# Import the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## Visualizing charts
plt.rcParams['figure.figsize']=[15,6]


# In[10]:


# Import the data

sales=pd.read_csv('~/Downloads/train_XnW6LSF.csv')
sales


# In[12]:


# Check the shape of the data :: Rows and Columns

sales.shape


# In[13]:


# Check the type of the data :

sales.info()


# ## Univariate analysis
# 
# ### numerical variable : distplot

# In[16]:


# Item weight

import warnings 
warnings.filterwarnings('ignore')

sns.distplot(sales.Item_Weight)
plt.title('Item_Weight')
plt.show()

# since the spread of the data is high, we can assume that the data is uniformly distributed.


# In[17]:


# target variable
sns.distplot(sales.Item_Outlet_Sales)
plt.title('Target')

# its right-skewed


# In[18]:


# Statistical Summary of the data
sales.Item_Outlet_Sales.describe()


# In[19]:


# Skewness
print('Skewness:',sales.Item_Outlet_Sales.skew())


# In[23]:


# Mode
print('Mode:',sales.Item_Outlet_Sales.mode()[0])


# ### Skewness
# 
# * The skewness refers to the distortion in the shape of the data due to presence of outliners.
# 
# * If the data is +vely skewed,we will see the skewness value as +ve and vice-versa for -vely skewed data
# 
# * If the skewness is btw **0 to 0.5**, we say that the data is normal.
# 
# * However, If the range of skewness is btw **0.5 to 1**, we say that the data is moderately skewed.
# 
# * If skewness is **1 or more than 1**, it is perfectly skewed.

# In[25]:


## Item MRP
sns.distplot(sales.Item_MRP)

# Item MRP shows 4 different frequencies and by looking at it ,It is clear that it is Multi-Modal data


# In[26]:


## Item Visibility
sns.distplot(sales.Item_Visibility)


# In[30]:


# for numerical variables
sales.select_dtypes(include=np.number)


# ## Univariate analysis
# ### Categorical variable : countplot

# In[27]:


# Find out the list of categories....

sales.select_dtypes(include='object')


# In[31]:


sales.select_dtypes(include='object').columns


# In[43]:


cols=['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier',
       'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
rows = 2
columns = 3
iterator = 1

for i in cols:
    plt.subplot(rows,columns,iterator)
    sns.countplot(sales.loc[:, i])
    plt.title(i)
    iterator+=1
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# ### Inferences
# 
# 1. Most of the products that are sold are **Low Fat** items.
# 2. The top 5 item types are **Fruits and veggies, Dairy, Snack food, Household etc **.
# 3. Most of the outlets in the business are **Medium** size outlets.
# 4. The business has opened most of their outlets in **Tier 3 cities**.
# 5. The type of the outlet is **S1**. 

# In[ ]:




