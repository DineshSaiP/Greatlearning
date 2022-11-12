#!/usr/bin/env python
# coding: utf-8

# ### Frequency Encoding

# In[79]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings 
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize']=[15,6]


# In[18]:


sales=pd.read_csv('~/Downloads/train_XnW6LSF.csv')
sales


# In[19]:


mapped=sales.Item_Type.value_counts(normalize=True).to_dict()


# In[20]:


# Map the item type on the basis of frequency %age
sales['Item_Type_Percent']=sales.Item_Type.map(mapped)
sales


# In[21]:


sales


# In[ ]:





# ### Target Encoding on Outlet Id

# In[22]:


mapped=sales.groupby('Outlet_Identifier')['Item_Outlet_Sales'].median().to_dict()


# In[23]:


sales['Outlet_ID']=sales.Outlet_Identifier.map(mapped)


# In[24]:


sales


# In[25]:


sales.loc[:,['Outlet_ID','Item_Outlet_Sales']].corr()


# ### Binning the Cat variables

# In[38]:


sales.Item_Type.unique()


# In[39]:


# Perishables and Non-Persishables
perish=['Dairy','Meat', 'Fruits and Vegetables','Breakfast','Breads', 'Starchy Foods', 'Seafood']


# In[40]:


def perish_ables(x):
    if x in perish:
        return('Perishables')
    else:
        return('Non_Perishables')


# In[44]:


sales['Item_Type_Cat']=sales.Item_Type.apply(perish_ables)


# In[45]:


sales


# In[46]:


## Drop the Item Type,Outlet Est year and Outlet ID

newsales=sales.drop(['Outlet_Establishment_Year','Outlet_Identifier'],axis=1)


# In[47]:


newsales


# ## Feature Engineering

# In[48]:


# Extracting the first 02 letters from the Item ID

sales.Item_Identifier[0][:2]


# In[49]:


ids=[]
for i in newsales.Item_Identifier:
    ids.append(i[:2])


# In[51]:


newsales['Item_Identifier']=pd.Series(ids)
newsales


# In[52]:


newsales.Item_Identifier.unique()


# In[84]:


# Boxplot between Item_id and sales
sns.boxplot(x='Item_Identifier',y='Item_Outlet_Sales',data=newsales)


# In[54]:


# Item_Fat_Content
newsales.Item_Fat_Content.unique()


# In[55]:


newsales.Item_Fat_Content.replace(to_replace=['LF','low fat','reg'],value=['Low Fat','Low Fat','Regular'],inplace=True)


# In[57]:


newsales.Item_Fat_Content.unique()


# In[83]:


sns.boxplot(data=newsales,x='Item_Fat_Content',y='Item_Outlet_Sales')


# In[82]:


sns.violinplot(data=newsales,x='Item_Fat_Content',y='Item_Outlet_Sales')
plt.axhline(7500,color='r')


# In[81]:


sns.distplot(newsales['Item_Outlet_Sales'],kde=True)


# In[68]:


newsales.groupby('Item_Fat_Content')['Item_Outlet_Sales'].describe()


# In[74]:


# Replacing the Item Fat Content with Non Edible where the Item ID is NC
newsales.loc[newsales.Item_Identifier=='NC','Item_Fat_Content']="Non Edible"
newsales


# In[76]:


# Replacing 0s in Item Visibility
newsales.groupby('Item_Identifier')['Item_Visibility'].transform(lambda x: x.replace(to_replace = 0,value=x.median()))


# In[80]:


sns.scatterplot(x='Item_Visibility',y='Item_Outlet_Sales',data=newsales,color='magenta')


# In[85]:


# Impute the Outlet size with mode
newsales.Outlet_Size.value_counts()


# In[89]:


pd.DataFrame(newsales.groupby(['Outlet_Type','Outlet_Location_Type'])['Outlet_Size'].count()).T


# In[90]:


pd.DataFrame(newsales.groupby(['Outlet_Size','Outlet_Location_Type'])['Outlet_Type'].count()).T


# In[99]:


# Mode Imputation
value=newsales.Outlet_Size.mode()[0]
newsales.Outlet_Size.fillna(value,inplace=True)
newsales.isna().sum()


# ## Outlier Removal
# 
# * Using Boxplot :: **Any value > Q3+1.5*IQR** or **Any value < Q1-1.5*IQR is known** as **Outlier**.
# * Using Z Scores :: If **Z Score is > 3** or if **Z Score < -3** is termed as **Outlier**.

# In[101]:


newsales.plot(kind='box')


# In[103]:


# Calculate the IQR and other Quantiles

q1=newsales.quantile(0.25)
q3=newsales.quantile(0.75)
iqr = q3 - q1

upper_lim = q3+1.5*iqr
lower_lim = q1-1.5*iqr


# In[113]:


# Condition

# Data without outliers

# ~ Not operator meeting the reverse criterion
# .any(axis=1) : returns the any row where the conditions are met
# | : Either or that means if the conditions are met either in lower lim or in upper limit
newsales.loc[~((newsales<lower_lim) | (newsales>upper_lim)).any(axis=1)]


# ## Removal of Outliers using Z Score
# 
# * Step1 - Convert the data into Z score values using StandardScaler/Fn
# * Step2 - Find the Outliers and Eliminate them

# In[114]:


from sklearn.preprocessing import StandardScaler

sc=StandardScaler()


# In[115]:


## Fetch the list of numerical variable only

newsales.select_dtypes(include = np.number).columns


# In[117]:


cols=['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Outlet_Sales','Outlet_ID']
for i in cols:
    newsales.loc[:,i]=sc.fit_transform(pd.DataFrame(newsales.loc[:,i]))
    


# In[118]:


newsales


# In[122]:


localdf = newsales.loc[:,cols]
localdf.loc[~((localdf<-3) | (localdf>3)).any(axis=1)]


# In[124]:


newsales.drop(['Item_Type'],axis=1,inplace=True)


# In[126]:


finaldata=pd.get_dummies(newsales,drop_first=True)


# In[127]:


finaldata.loc[~((finaldata<-3) | (finaldata>3)).any(axis=1)]


# In[ ]:




