#!/usr/bin/env python
# coding: utf-8

# In[245]:


import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[246]:


titan = pd.read_csv("D:\python\data set/titanic_lr.csv")
titan.head()


# In[247]:


titan.info()


# In[248]:


import seaborn as sns


# In[249]:


sns.countplot(x="Survived",data=titan)


# In[250]:


sns.countplot(x="Survived", hue = "Sex",data=titan)


# In[251]:


sns.countplot(x = "Pclass" , hue = "Survived" , data = titan)


# In[252]:


x = titan["Age"]
x.head()


# In[253]:


x.isnull().value_counts()


# In[254]:


x1 = x.fillna(0)
x1.head()


# In[255]:


x1.isnull().value_counts()


# In[256]:


sns.distplot(x1 , bins = 10 ,kde=True, rug=False );


# Kde ----> Whether to plot a gaussian kernel density estimate.
# Rug ----> Whether to draw a rugplot on the support axis.


# In[257]:


titan["Age"].plot.hist()


# In[258]:


titan["Fare"].plot.hist(bins=20,figsize=(10,5))


# Fare is not making a big diffrence


# In[259]:


sns.countplot(x="SibSp",data=titan)


# In[260]:


sns.countplot(x="Parch",data=titan)  # no of parents and children


#means who are single servived the most


# In[261]:


titan.isnull().sum()


# In[262]:


sns.heatmap(titan.isnull(),yticklabels = False)


# In[263]:


titan_filter = titan.dropna(subset = ["Age" , "Cabin" , "Embarked"])


# In[264]:


sns.heatmap(titan_filter.isnull(),yticklabels = False)


# In[265]:


titan_filter.isnull().sum() 


#now our data is clean


# In[266]:


titan_filter['gender_factor'] = pd.factorize(titan_filter.Sex)[0]
titan_filter.head()


# In[267]:


pd.get_dummies(titan_filter["gender_factor"]).head()


# here 0 column denotes Female and 1 column denotes male


# In[268]:


# drop 1 column as we need only one because if 1 is true other is false

gender = pd.get_dummies(titan_filter["gender_factor"],drop_first=True)
gender.head()


# In[269]:


embark = pd.get_dummies(titan_filter["Embarked"])
embark.head()


# In[270]:


embark = pd.get_dummies(titan_filter["Embarked"] , drop_first = True)
embark.head()


# In[271]:


pcl = pd.get_dummies(titan_filter["Pclass"],drop_first=True)
pcl.head()


# In[272]:


titan_filter.drop(['Sex','Embarked','PassengerId','Pclass','Name','Ticket','gender_factor', 'Cabin'],axis=1,inplace=True)


# In[273]:


titan_filter.head()


# In[274]:


titan_fil =pd.concat([titan_filter,gender,embark,pcl],axis=1)


# In[275]:


titan_fil.head()


# In[276]:


# data is clean and clear , so we can train our data

x =titan_fil.drop("Survived",axis=1)
y = titan_fil["Survived"]


# In[277]:


from sklearn.cross_validation import train_test_split


# In[278]:


train_test_split


# In[279]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

X_train, X_test, y_train, y_test


# In[280]:


from sklearn.linear_model import LogisticRegression


# In[281]:


logmodel=LogisticRegression()


# In[282]:


logmodel.fit(X_train, y_train)


# In[ ]:





# In[ ]:




