#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Agenda
* Missing values
* 
* 
* Coefficient of Variation
* Dealing with cat variables


# In[2]:


# Import the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings 
warnings.filterwarnings('ignore')

## Visualizing charts
plt.rcParams['figure.figsize']=[15,6]


# In[3]:


sales=pd.read_csv('~/Downloads/train_XnW6LSF.csv')
sales


# In[4]:


## Missing values
sales.isnull().sum()[sales.isnull().sum()!=0]


# In[5]:


sales.loc[sales.Item_Identifier=='FDS36', 'Item_Weight']


# In[6]:


# dict_=sales.groupby('Item_Identifier')['Item_Weight'].median().to_dict
# sales['Item_Weight']=sales.groupby('Item_Identifier')['Item_Weight'].fillna(dict_)

# sales.groupby('Item_Identifier')['Item_Weight'].map(dict_)
# (method 1)


# In[7]:


sales['Item_Weight']=sales.groupby('Item_Identifier')['Item_Weight'].transform(lambda x:x.fillna(x.median()))
sales.isnull().sum()
# (method 2)


# In[8]:


sales.loc[sales.Item_Weight.isnull()]


# In[9]:


# where item type == frozen food,Item weight.median()
print(sales.loc[sales.Item_Type=='Frozen Foods','Item_Weight'].median())
sales.loc[927,'Item_Weight']=12.85
sales.loc[sales.Item_Weight.isnull()]


# In[10]:


a=print(sales.loc[sales.Item_Type=='Snack Foods','Item_Weight'].median())
sales.loc[1922,'Item_Weight']=13.15
sales.loc[sales.Item_Weight.isnull()]


# In[11]:


print(sales.loc[sales.Item_Type=='Dairy','Item_Weight'].median())
sales.loc[4187,'Item_Weight']=13.35
sales.loc[sales.Item_Weight.isnull()]


# In[12]:


print(sales.loc[sales.Item_Type=='Baking Goods','Item_Weight'].median())
sales.loc[5022,'Item_Weight']=11.65
sales.loc[sales.Item_Weight.isnull()]

## or
# sales.loc[(sales.Item_Type='Baking Foods') & (sales.Item_Weight.isnull()),'Item_Weight']=11.65


# In[15]:


sales.isnull().sum()


# In[13]:


sales['Outlet_Size']=sales.groupby('Item_Type')['Outlet_Size'].transform


# In[14]:


# Calculate the standard deviation of target and predictor variables
sales.std()


# In[49]:


# Calculate the Coefficient of Variation in the data
sales.std()/sales.mean()


# ### Interpretation of CV
# 
# * Coefficient of Variation suggests which of the categories have the lowest variaition wrt Mean.
# * It is calculated as **std/mean** which means what %age of the data is deviaiting the mean.
# * Higher the CV, less reliable that variable in the data is.
# * Lower the CV, we can count on that feature more.....
# 
# Note : Low CV is always preferred when comparing two or more products/employees/sports players/stocks etc.
# 
# 
# ### Kurtosis
# 
# * Tells you the shape of the data.
# * It tells us by looking ata the **Peakedness of the data**. 
# * Measure of Tailedness of the data
# * Here Peakedness of the data represents **Heavy tails** or **Light tails**.
# * The Normal Distribution Curve can be divided in 3 parts -
#     **Mesokurtic Curve , Platykurtic and Leptokurtic Curve**
# 
# Note : Normal Distribution is **Meskurtic Curve** with a Kurtosis of **3**.
# 
# Note : If any curve has a Kurtosis of **<3,Platykurtic curve** and if it is **>3,it is Leptokurtic**
# 
# Note : In Platykurtic, Std is higher and in Leptokurtic, Std is lower
# 
# **NOTE : The Standard Relation is Skewness = 0 and Kurtosis = 3 for the data to be Normal**
# 

# In[50]:


sales.kurt()


# ## Scaling
# 1. Standard scaler
# 2. Min Max scaler

# In[51]:


df=sales.select_dtypes(include=np.number)


# In[53]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[54]:


## Initiate the machine

sc=StandardScaler()
mmax=MinMaxScaler()


# In[58]:


# applying the function - Fit transform
# Standard Scaler Transformation
sc.fit_transform(df) # returns array
scaled=pd.DataFrame(sc.fit_transform(df),columns=df.columns)
scaled


# In[59]:


scaled.describe()
# here mean=0(its 10^-16 so its literally 0) and std=1


# In[61]:


# MinMax Scaler Transformation
scaled_mmax=pd.DataFrame(mmax.fit_transform(df),columns=df.columns)
scaled_mmax


# In[62]:


scaled_mmax.describe()
# here min=0 and max=1


# ### About Normal Distribution
# 
# * It is a Bell-shaped curve
# * The total area under the curve is **1**
# * The Probability under the area under the curve(AUC) is btw **0 and 1**
# * The Normal distribution can be converted into Standard normal distribution where the **mean of the data will be 0** & **std will be 1**
# 
# 
# ### Emperical Rule
# * **68% data** lies within **1 standard deviation**
# * **95% data** lies within **2 standard deviation**
# * **99.7% data** lies within **3 standard deviation**
# 
# Note: Any value that lies above 3 sigma or below 3 sigma is an outlier

# ### Categorical Variables
# 
# * One Hot Encoding
# * Label Encoding
# * Frequency Encoding
# * Target Encoding
# 
# **Note on One Hot Encoding and Label Encoding**
# 
# * One Hot Encoding encodes the data in 0 and 1.
# * It means each and every column is 0 and 1 at the same time.
# * It is popularly used for those categorical variables which has no order.
# * In case the categorical variables have order,then it makes sense to use Label Encoder.
# * One Hot Encoder converts all the categories in 0 and 1 and therefore,it generates as many new columns as the count of categories.
# * In case of Label encoding,it is used when the categories are ordinal in nature.It means that the categories have a certain order and we can say that 3>2>1.
# * The key aspect of LE is that it encodes the categories alphabetically.
# 

# In[63]:


sales.select_dtypes(include='object').columns


# In[65]:


sales.Item_Type.unique()


# In[66]:


pd.get_dummies(sales)


# In[67]:


sales.Outlet_Size.dropna().unique()


# In[68]:


mapped = {'Small':1,'Medium':2,'High':3}


# In[69]:


sales.Outlet_Size.dropna().map(mapped)  # label encoding


# In[ ]:




