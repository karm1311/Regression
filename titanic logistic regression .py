
# coding: utf-8

# In[146]:


import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[147]:


tit_data = pd.read_csv('D:\DATA SCIENCE\data set\python/titanic_lr.csv')


# In[148]:


tit_data.head()


# In[149]:


tit_data.info()


# In[150]:


#analyzing data
sns.countplot(x="Survived",data=tit_data)


# In[151]:


sns.countplot(x="Survived", hue = "Sex",data=tit_data)


# In[152]:


sns.countplot(x="Survived", hue = "Pclass",data=tit_data)


# In[153]:


tit_data["Age"].plot.hist()


# In[154]:


tit_data["Fare"].plot.hist(bins=20,figsize=(10,5))


# In[155]:


sns.countplot(x="SibSp",data=tit_data)


# In[156]:


sns.countplot(x="Parch",data=tit_data) #no of parents and children


# In[157]:


#cleaning the data by removing NaN values or unnecessary columns
#tit_data.isnull().


# In[158]:


tit_data.isnull().sum()


# In[159]:


sns.heatmap(tit_data.isnull(),yticklabels = False)


# In[160]:


sns.boxplot(x="Pclass", y="Age",data=tit_data)


# In[161]:


tit_data.head()


# In[162]:


tit_data.drop("Cabin",axis=1,inplace = True)


# In[163]:


tit_data.head(5)


# In[164]:


tit_data.dropna(inplace=True)


# In[165]:


sns.heatmap(tit_data.isnull(),yticklabels = False)


# In[166]:


tit_data.isnull().sum() #now we can see the data is clean


# In[167]:


pd.get_dummies(tit_data["Sex"]) #press shift+tab to get the more information


# In[168]:


gender = pd.get_dummies(tit_data["Sex"],drop_first=True) #one column is enough so drop one
gender.head()


# In[169]:


embark = pd.get_dummies(tit_data["Embarked"])
embark.head()


# In[170]:


embark = pd.get_dummies(tit_data["Embarked"],drop_first=True)
embark.head()


# In[171]:


pcl = pd.get_dummies(tit_data["Pclass"],drop_first=True)
pcl.head()


# In[172]:


tit_data.drop(['Sex','Embarked','PassengerId','Pclass','Name','Ticket'],axis=1,inplace=True)


# In[173]:


tit_data.head()


# In[174]:


tit_data=pd.concat([tit_data,gender,embark,pcl],axis=1)


# In[176]:


tit_data.head()


# In[177]:


#train data
x =tit_data.drop("Survived",axis=1)
y = tit_data["Survived"]


# In[178]:


from sklearn.cross_validation import train_test_split


# In[180]:


train_test_split   #press shift and tab to get more details


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[183]:


from sklearn.linear_model import LogisticRegression


# In[184]:


logmodel=LogisticRegression()


# In[ ]:


logmodel.fit(X_train, y_train)

