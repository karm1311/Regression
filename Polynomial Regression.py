
# coding: utf-8

# Polynomial Regression is a form of linear regression in which the relationship between the independent variable x and dependent variable y is modeled as an nth degree polynomial. Polynomial regression fits a nonlinear relationship between the value of x and the corresponding conditional mean of y, denoted E(y |x)
# 
# To obtain the degree of a polynomial defined by the following expression x3+x2+1, enter : degree(x^3+x^2+1) after calculation, the result 3 is returned.
# 
# Why Polynomial Regression:
# 
# There are some relationships that a researcher will hypothesize is curvilinear. Clearly, such type of cases will include a polynomial term.
# Inspection of residuals. If we try to fit a linear model to curved data, a scatter plot of residuals (Y axis) on the predictor (X axis) will have patches of many positive residuals in the middle. Hence in such situation it is not appropriate.
# An assumption in usual multiple linear regression analysis is that all the independent variables are independent. In polynomial regression model, this assumption is not satisfied.
# Uses of Polynomial Regression:
# These are basically used to define or describe non-linear phenomenon such as:
# 
# Growth rate of tissues.
# Progression of disease epidemics
# Distribution of carbon isotopes in lake sediments

# # Before we go into details of polynomial regression, let us consider our case study first.

# Problem Statement:
# Consider our company wants to recruit certain people for certain positions. They have found a potential employee who is currently working as a vice president for last 2 years. He is expecting a salary of 190000 for his 2 years experience as vice president.
# 
# Now the question is, does his demand fit into our company’s salary structure? If his demand fits, how much can we really offer him?
# Looking at the dataset, it does not seem linear. As higher positions are concerned, salaries are changing non linearly. Let us first check whether linear regression is providing any good predictions.

# In[5]:


dataset= pd.read_csv("D:\DATA SCIENCE\data set/Position_Salaries.csv")
dataset


# Usually in our company, an employee can rise from Vice President to President level in 6 years, so we will predict salary for 8.3 level, because the employee has worked one third of tenure to become President.
# 
# Why to use Polynomial Regression?
# Let us understand why we are using polynomial regression instead of linear regression. Looking at the dataset, it does not seem linear. As higher positions are concerned, salaries are changing non linearly. Let us first check whether linear regression is providing any good predictions.

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importing the dataset
dataset = pd.read_csv("D:\DATA SCIENCE\data set/Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values # This line creates a matrix
y = dataset.iloc[:, 2].values
 
# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
 
salary = lin_reg.predict(8.3)
 
# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Linear Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Note that red dots are actual salaries plotted against position levels. for position 8.3, our model has predicted a salary of approximately 239851, which is way more than potential employee wanted.
# 
# Blue line is regression line and its predictions are far from the reality in most of the cases. We can have a different model to consider and hence we can look into polynomial regression.

# Polynomial Regression Model:
# Equation of polynomial regression model is
# 
# y = b0 + b1x1 + b2x12 + b3x13 + … + bnx1n

# In[9]:


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Polynomial Regression
salary = lin_reg_2.predict(poly_reg.fit_transform(8.3))


# import PolynomialFeatures class from sklearn.preprocessing.
# create poly_reg object with 4th degree Polynomial features. i.e. for our case, equation will become y = b0 + b1x1 + b2x12 + b3x13 + b4x14
# X_poly = poly_reg.fit_transform(X), will expand x matrix into X_poly where each column will contain values of powers of x.

# Salary predicted by polynomial regression is approximately 189117 which is quite close to what he was asking for and it fits our company salary model.
# 
# Our company can hire the new vice president and he will happily come to us :

# In[4]:


#This is minimal Example of Polynomial Regression with One Variable as we Know when we add polynomial terms to 
#Our regression hypothesis  the function will no more linear so it will lead to better fit our model to non linear data set

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.linear_model import LinearRegression
#loading the dataset
dataset= pd.read_csv("D:\DATA SCIENCE\data set/Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

#adding Polynomial for better fitting of data 
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree= 2)
poly_features = poly.fit_transform(X)
poly.fit(X,y)
poly_regression = LinearRegression()
poly_regression.fit(poly_features,y)
#normal regression
regressor=LinearRegression()
regressor.fit(X,y)
#ploting the data  
plt.scatter(X,y)
plt.plot(X,poly_regression.predict(poly_features))
plt.title("PolyNomial Regression Experiance Vs Salary with degree 2 ")
plt.xlabel("Experiance ")
plt.ylabel("Salary data ")
plt.show()
#plotting linear hypothesis
plt.scatter(X,y)
plt.plot(X,regressor.predict(X))
plt.title("Linear  Regression Experiance Vs Salary  ")
plt.xlabel("Experiance ")
plt.ylabel("Salary data ")
plt.show() 
# higher degree
# Adding Polynominals to the hypothesis 
poly = PolynomialFeatures(degree= 3)
poly_features = poly.fit_transform(X)
poly.fit(X,y)
poly_regression = LinearRegression()
poly_regression.fit(poly_features,y)
#ploting the data  for polynomial regression 
plt.scatter(X,y)
plt.plot(X,poly_regression.predict(poly_features))
plt.title("PolyNomial Regression Experiance Vs Salary with degree 3 ")
plt.xlabel("Experiance ")
plt.ylabel("Salary data ")
plt.show()
#ploting the Linear regresson
plt.scatter(X,y)
plt.plot(X,regressor.predict(X))
plt.title("Linear  Regression Experiance Vs Salary  ")
plt.xlabel("Experiance ")
plt.ylabel("Salary data ")
plt.show() 


# Advantages of using Polynomial Regression:
# 
# Broad range of function can be fit under it.
# Polynomial basically fits wide range of curvature.
# Polynomial provides the best approximation of the relationship between dependent and independent variable.
# Disadvantages of using Polynomial Regression
# 
# These are too sensitive to the outliers.
# The presence of one or two outliers in the data can seriously affect the results of a nonlinear analysis.
# In addition there are unfortunately fewer model validation tools for the detection of outliers in nonlinear regression than there are for linear regression.
