
# coding: utf-8

# ### Topics to Discuss:
# >What is linear regression?<br>
# >Analyzing Advertisement dataset.<br>
# >Building a simple linear regression model & multiple linear regression model.<br>
# >Understanding OLS methods to estimate model parameters.<br>
# >How to use statsmodel API in python?<br>
# >Interpreting the coefficients of the model.<br>
# >How to find if the parameters estimated are significant?<br>
# >Making predictions using the model.<br>
# >Finding model residuals and analyzing it.<br>
# >Evaluating model efficiency using RMSE and R-Square values.<br>
# >Understanding gradient descent approach to find model parameters.<br>
# >Splitting dataseta and cross validating models.<br>

# In[ ]:


import pandas as pd
import numpy as np


# ### Adverstiment Dataset
# >The adverstiting dataset captures sales revenue generated with respect to advertisement spends across multiple channles 
# >like radio, tv and newspaper.
# 
# ### Attribution Descriptions
# >TV - Spend on TV Advertisements <br>
# >Radio - Spend on radio Advertisements <br>
# >Newspaper - Spend on newspaper Advertisements <br>
# >Sales - Sales revenue generated <br>
# Note: The amounts are in diffrent units

# In[ ]:


# load the data set
advt = pd.read_csv( "./Advertising.csv" )


# In[ ]:


advt.head(5)


# In[ ]:


advt.info()


# In[ ]:


#Remove the first column

advt = advt[["TV", "Radio", "Newspaper", "Sales"]]


# In[ ]:


advt.head()


# In[ ]:


# Creating Data audit Report
# Use a general function that returns multiple values
def var_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.quantile(0.01), x.quantile(0.05),x.quantile(0.10),x.quantile(0.25),x.quantile(0.50),x.quantile(0.75), x.quantile(0.90),x.quantile(0.95), x.quantile(0.99),x.max()], 
                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])

advt.apply(lambda x: var_summary(x)).T


# In[ ]:


#Handling Outliers
advt['Sales']= advt['Sales'].clip_upper(advt['Sales'].quantile(0.99)) 
advt['Sales']= advt['Sales'].clip_lower(advt['Sales'].quantile(0.01)) 

advt['Newspaper']= advt['Newspaper'].clip_upper(advt['Newspaper'].quantile(0.99)) 
advt['Newspaper']= advt['Newspaper'].clip_lower(advt['Newspaper'].quantile(0.01))


# In[ ]:


#Handling Missings
# Fill with mean
advt['Sales']=advt['Sales'].fillna(advt['Sales'].mean())


# In[ ]:


#Dummy variable creation
#print df_G.join(pd.get_dummies(advt['key'], prefix='dummy')).drop('key', axis=1).drop('dummy_c', axis=1)


# In[ ]:


# exploring data
# Distribution of variables
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.distplot(advt.Sales)


# In[ ]:


sns.distplot(advt.Newspaper)


# In[ ]:


sns.distplot( advt.Radio )


# In[ ]:


sns.distplot( advt.TV )


# ### Notes:
# >Sales seems to be normal distribution. Spending on newspaper advertisement seems to be righ skewed.
# Most of the spends on newspaper is fairly low where are spend on radio and tv seems be uniform distribution. 
# Spends on tv are comparatively higher then spens on radio and newspaper.

# In[ ]:


#Is there a relationship between sales and spend various advertising channels?

#Sales Vs. Newspaper advertisement spends

sns.jointplot(  advt.Newspaper, advt.Sales )


# In[ ]:


#Sales Vs. TV advertisement spends
sns.jointplot(  advt.TV, advt.Sales )


# # Notes
# >Sales and spend on newpaper is not highly correlaed where are sales and spend on tv is highly correlated.

# In[ ]:


get_ipython().run_line_magic('pinfo', 'sns.pairplot')


# In[ ]:


# Visualizing pairwise correleation

sns.pairplot( advt )


# In[ ]:


# Calculating correlations
advt.TV.corr( advt.Sales )


# In[ ]:


advt.corr()


# In[ ]:


# Visualizing the correlations
#The darker is the color, the stronger is the correlation
sns.heatmap( advt.corr() )


# ### NOTES:
# > The diagonal of the above matirx shows the auto-correlation of the variables. It is always 1. You can observe that the correlation betweeb TV and Sales is highest i.e. 0.78 and then betweeb sales and radio i.e. 0.576. <br>
# > correlations can vary from -1 to +1. Closer to +1 means strong positive correlation and close -1 means strong negative correlation. Closer to 0 means not very strongly correlated. variables with strong correlations are mostly probably candidates for model builing

# ### Building Regression Model
# > Linear regression is an approach for modeling the relationship between a scalar dependent variable y and one or more explanatory variables (or independent variables) denoted X. The case of one explanatory variable is called simple linear regression. For more than one explanatory variable, the process is called multiple linear regression <br>
# > A simple linear regression model is given by Y=mX+b <br>
# > where m is the slope and b is the y-intercept. Y is the dependent variable and X is the explanatory variable. <br>
# > Very briefly and simplistically, Linear Regression is a class of techniques for fitting a straight line to a set of data points

# In[ ]:


import statsmodels.formula.api as smf


# In[ ]:


advt.columns


# In[ ]:


lm = smf.ols( 'Sales ~ TV+Radio', advt ).fit()


# In[ ]:


lm.summary()


# In[ ]:


# Getting the f value
lm.f_pvalue


# In[ ]:


# Getting model parameters
lm.params


# In[ ]:


# Parameters at 95% confidence intervals

lm.conf_int()


# In[ ]:


# Verifying parameter significance
lm.pvalues


# ### Notes:
# > Parameters estimated are considered to be significant if p-value is less than 0.05 <br>
# > This indicates intercept and TV both are significant parameters. And the parameter estimates can be accepted. <br>
# > So, the linear model is Sales=7.032+0.047∗TV
# 

# ### Evaluating Model Accuracy
# > R-squared is a statistical measure of how close the data are to the fitted regression line. <br>
# > R-square signifies percentage of variations in the reponse variable that can be explained by the model. <br>
# > R-squared = Explained variation / Total variation <br>
# > Total variation is variation of response variable around it's mean. <br>
# > R-squared value varies between 0 and 100%. 0% signifies that the model explains none of the variability, <br>
# > while 100% signifies that the model explains all the variability of the response. <br>
# > The closer the r-square to 100%, the better is the model. <br>

# In[ ]:


lm.rsquared


# In[ ]:


round( float( lm.rsquared ), 2 )


# In[ ]:


lm.predict


# In[ ]:


fit.predict(example_df["c"])


# In[ ]:


### MAKING PREDICTION`aS
lmpredict = lm.predict( {'TV': advt.TV,'Radio':advt.Radio } )


# In[ ]:


lmpredict[0:10]


# In[ ]:


from sklearn import metrics


# ### Calculating mean square error ... RMSE
# > RMSE calculate the difference between the actual value and predicted value of the response variable <br>
# > The square root of the mean/average of the square of all of the error. <br> 
# > Compared to the similar Mean Absolute Error, RMSE amplifies and severely punishes large errors. <br>
# > The lesser the RMSE value, the better is the model.

# In[ ]:


mse = metrics.mean_squared_error( advt.Sales, lmpredict )


# In[ ]:


rmse = np.sqrt( mse )


# In[ ]:


rmse


# In[ ]:


#Get the residuals and plot them
lm.resid[1:10]


# > One of the assumptions is that the residuals should be normally distributed i.e. it should be random.
# The residuals should be plotted against the response variable and it should not show any pattern

# In[ ]:


sns.jointplot(  advt.Sales, lm.resid )


# ### Multiple Linear Regression Model

# In[ ]:


lm = smf.ols( 'Sales ~ TV + Radio + Newspaper', advt ).fit()


# In[ ]:


lm.summary()


# In[ ]:


lm.params


# In[ ]:


lm.pvalues


# In[ ]:


lm = smf.ols( 'Sales ~ TV + Radio', advt ).fit()


# In[ ]:


lm.params


# In[ ]:


lm.pvalues


# In[ ]:


lmpredict = lm.predict( {'TV': advt.TV, 'Radio':advt.Radio } )


# In[ ]:


mse = metrics.mean_squared_error( advt.Sales, lmpredict )
rmse = np.sqrt( mse )
rmse


# In[ ]:


sns.distplot(lm.resid)


# In[ ]:


sns.jointplot(  advt.Sales, lm.resid )


# ### USING sklearn Library to build the model
# > sklearn library has a comprehensive set of APIs to split datasets, build models, test models and calculate accuracy metrics

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


# Splitting into Train and test data sets
# Typically the model should be built on a training dataset and validated against a test dataset
# Let's split the dataset into 70/30 ratio. 70% belongs to training and 30% belongs to test.

from sklearn.cross_validation import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
  advt[["TV", "Radio", "Newspaper"]],
  advt.Sales,
  test_size=0.3,
  random_state = 42 )


# In[ ]:


print len( X_train )
print len( X_test)


# In[ ]:


## Building the model with train set and make predictions on test set
linreg = LinearRegression()
linreg.fit( X_train, y_train )
y_pred = linreg.predict( X_test )


# In[ ]:


rmse = np.sqrt( metrics.mean_squared_error( y_test, y_pred ) )
rmse


# In[ ]:


metrics.r2_score( y_test, y_pred )


# In[ ]:


list( zip( ["TV", "Radio", "Newspaper"], list( linreg.coef_ ) ) )


# In[ ]:


residuals = y_test - y_pred


# In[ ]:


sns.jointplot(  advt.Sales, residuals )


# In[ ]:


sns.distplot( residuals )


# In[ ]:


# To ensure residues are random i.e. normally distributed a Q-Q plot can be used
# Q-Q plot shows if the residuals are plotted along the line.
from scipy import stats
import pylab

stats.probplot( residuals, dist="norm", plot=pylab )
pylab.show()


# The residuals are randomly distributed. There are no visible relationship. The model can be assumed to be correct

# In[ ]:


### K-FOLD CROSS VALIDATION
from sklearn.cross_validation import cross_val_score


# In[ ]:


linreg = LinearRegression()


# In[ ]:


cross_val_score( linreg, X_train, y_train, scoring = 'r2', cv = 10 )


# In[ ]:


round( np.mean( cross_val_score( linreg,
                              X_train,
                              y_train,
                              scoring = 'r2',
                              cv = 10 ) ), 2 )


# In[ ]:


# Feature Selection based on importance
from sklearn.feature_selection import f_regression


# In[ ]:


F_values, p_values  = f_regression(  X_train, y_train )


# In[ ]:


F_values


# In[ ]:


['%.3f' % p for p in p_values]


# As p - values are less than 5% - the variables are siginificant in the regression equation.

# In[ ]:


### Exporting and importing the model
import pickle


# In[ ]:


from sklearn.externals import joblib
joblib.dump(linreg, 'lin_model.pkl', compress=9)


# In[ ]:


model_clone = joblib.load('lin_model.pkl')

