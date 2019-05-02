
# coding: utf-8

# Linear Regression using two dimensional data 
# 
# First, let’s understand Linear Regression using just one dependent and independent variable.
# 
# I create two lists  xs and ys.

# In[76]:


xs = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
ys = [10,12,20,22,21,25,30,21,32,34,35,30,50,45,55,60,66,64,67,72,74,80,79,84]


# In[77]:


len(xs), len(ys)


# I plot these lists using a scatter plot. I assume xs as the independent variable and ys as the dependent variable.

# In[78]:


import numpy as np
import matplotlib.pyplot as plt
plt.scatter(xs,ys)
plt.ylabel("dependent variable")
plt.xlabel("independent variable")
plt.show()


# You can see that the dependent variable has a linear distribution with respect to the independent variable.
# 
# A linear regression line has the equation Y = mx+c, where m is the coefficient of independent variable and c is the intercept.
# 
# The mathematical formula to calculate slope (m) is:
# 
# (mean(x) * mean(y) – mean(x*y)) / ( mean (x)^2 – mean( x^2))
# 
# The formula to calculate intercept (c) is:
# 
# mean(y) – mean(x) * m
# 
# Now, let’s write a function for intercept and slope (coefficient):

# In[96]:


def slope_intercept(x_val, y_val):
    x = np.array(x_val)
    y = np.array(y_val)
    m = ((np.mean(x)*np.mean(y)) - np.mean(x*y)) / ((np.mean(x)*np.mean(x)) - np.mean(x*x))
    m = round(m,2)
    b = (np.mean(y) - np.mean(x)*m)
    b = round(b,2)
    return m,b


# To see the slope and intercept for xs and ys, we just need to call the function slope_intercept:

# In[97]:


slope_intercept(xs,ys)


# In[98]:


m,b = slope_intercept(xs,ys)


# In[99]:


#reg_line is the equation of the regression line:
reg_line = [(m*x)+b for x in xs]


# In[100]:


#Now, let’s plot a regression line on xs and ys:
plt.scatter(xs,ys, color = "green")
plt.plot(xs, reg_line)
plt.ylabel("dependent variable")
plt.xlabel("independent variable")
plt.title("making a regression line")
plt.show()


# Root Mean Squared Error(RMSE)
# 
# RMSE is the standard deviation of the residuals (prediction errors). Residuals are a measure of how far from the regression line data points are, and RMSE is a measure of how spread out these residuals are.
# 
# If Yi is the actual data point and Y^i is the predicted value by the equation of line then RMSE is the square root of (Yi – Y^i)**2
# 
# Let’s define a function for RMSE:

# In[110]:


from math import sqrt
def rmse(y1 , y_hat):
    y_actual = np.array(y1)
    y_pred = np.array(y_hat)
    error = (y_actual - y_pred)**2
    error_mean = round(np.mean(error))
    err_sq = sqrt(error_mean)
    return err_sq


# In[111]:


rmse(ys, reg_line)


# Linear regression is a statistical approach for modelling relationship between a dependent variable with a given set of independent variables.

# In[118]:


import numpy as np 
import matplotlib.pyplot as plt 
  
def estimate_coef(x, y): 
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x - n*m_y*m_x) 
    SS_xx = np.sum(x*x - n*m_x*m_x) 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
  
    return(b_0, b_1) 
  
def plot_regression_line(x, y, b): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "m", 
               marker = "o", s = 30) 
  
    # predicted response vector 
    y_pred = b[0] + b[1]*x 
  
    # plotting the regression line 
    plt.plot(x, y_pred, color = "g") 
  
    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 
  
    # function to show plot 
    plt.show() 
  
def main(): 
    # observations 
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12]) 
  
    # estimating coefficients 
    b = estimate_coef(x, y) 
    print("Estimated coefficients:\nb_0 = {}nb_1 = {}".format(b[0], b[1])) 
  
    # plotting regression line 
    plot_regression_line(x, y, b) 
  
if __name__ == "__main__": 
    main() 


# Given below is the implementation of multiple linear regression technique on the Boston house pricing dataset dataset using Scikit-learn.

# In[120]:


import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets, linear_model, metrics 
  
# load the boston dataset 
boston = datasets.load_boston(return_X_y=False) 
  
# defining feature matrix(X) and response vector(y) 
X = boston.data 
y = boston.target 
  
# splitting X and y into training and testing sets 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 
                                                    random_state=1) 
  
# create linear regression object 
reg = linear_model.LinearRegression() 
  
# train the model using the training sets 
reg.fit(X_train, y_train) 
  
# regression coefficients 
print('Coefficients: \n', reg.coef_) 
  
# variance score: 1 means perfect prediction 
print('Variance score: {}'.format(reg.score(X_test, y_test))) 
  
# plot for residual error 
  
## setting plot style 
plt.style.use('fivethirtyeight') 
  
## plotting residual errors in training data 
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, 
            color = "green", s = 10, label = 'Train data') 
  
## plotting residual errors in test data 
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, 
            color = "blue", s = 10, label = 'Test data') 
  
## plotting line for zero residual error 
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2) 
  
## plotting legend 
plt.legend(loc = 'upper right') 
  
## plot title 
plt.title("Residual errors") 
  
## function to show plot 
plt.show() 

