
# coding: utf-8

# Linear Regression is one of the most common, years old and most easily understandable in statistics and machine learning. It comes under predictive modelling. Predictive modelling is a kind of modelling here the possible output(Y) for the given input(X) is predicted based on the previous data or values.
# 
# Types
# 1)Simple Linear Regression: It is characterized by one independent variable. Consider the price of the house based only one field that is the size of the plot then that would be a simple linear regression.
# 
# 2)Multiple Linear Regression: It is characterized by multiple independent variables. The price of the house if depends on more that one like the size of the plot area, the economy then it is considered as multiple linear regression which is in most real-world scenarios.
# 
# Equation:
# Y = aX + b
# 
# where,
# a: slope of the line, b: constant(Y-intercept,where X=0, X:Independent variable, Y:Dependent variable
# 
# Here the main aim is to find the best fit line, which minimizes error(the sum of the square of the distance between points and the line). The distance between the points and line are taken and each of them is squared to get rid of negative values and then the values are summed which gives the error which needs to be minimized.

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
plt.scatter(x,y)
plt.show()


# Now, the task is to find a line which fits best in above scatter plot so that we can predict the response for any new feature values.(i.e a value of x not present in dataset)
# This line is called regression line.
# 
# The equation of regression line is represented as:
# 
#  h(xi) = b0 + b1xi 
# 
# Here,
# 
# h(x_i) represents the predicted response value for ith observation.
# b_0 and b_1 are regression coefficients and represent y-intercept and slope of regression line respectively.
# To create our model, we must “learn” or estimate the values of regression coefficients b_0 and b_1. And once we’ve estimated these coefficients, we can use the model to predict responses!
# 
# In this article, we are going to use the Least Squares technique.
# 
# Now consider:
# 
#  yi = b0 + b1xi + ei = h(xi) + ei =>ei = yi -h(xi) 
# 
# Here, e_i is residual error in ith observation.
# So, our aim is to minimize the total residual error.
# 
# We define the squared error or cost function,
# 
# and our task is to find the value of b_0 and b_1 for which J(b_0,b_1) is minimum!
# 
# Note: The complete derivation for finding least squares estimates in simple linear regression in SLR_Leastsquares.pdf

# In[3]:


import numpy as np 
import matplotlib.pyplot as plt 

def estimate_coef(x, y): 
	# number of observations/points 
	n = np.size(x) 

	# mean of x and y vector 
	m_x, m_y = np.mean(x), np.mean(y) 

	# calculating cross-deviation and deviation about x 
	SS_xy = np.sum(y*x) - n*m_y*m_x 
	SS_xx = np.sum(x*x) - n*m_x*m_x 

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
	print("Estimated coefficients:\nb_0 = {}      \nb_1 = {}".format(b[0], b[1])) 

	# plotting regression line 
	plot_regression_line(x, y, b) 

if __name__ == "__main__": 
	main() 


# The sign of each coefficient indicates the direction of the relationship between a predictor variable and the response variable.
# 
# A positive sign indicates that as the predictor variable increases, the response variable also increases.
# A negative sign indicates that as the predictor variable increases, the response variable decreases.

# In[8]:


#Now prepare your data for your practice:
#X — House size from 1K sq feet to 10K sq feet.
#Y — Cost of the house from 300K to 1200K.


# In[15]:


x = [1,2,3,4,5,6,7,8,9,10]
y = [300,350,500,700,800,850,900,900,1000,1200]
plt.scatter(x,y)
plt.show()


# Now we need to find the regression line(a line which fits best in the above scatter plot so that we can predict the response y(ie. cost of the house) for any new values of x(ie. size of the house).

# # Function1: It is a function to determine or estimate the coefficients where x and y values are passed into this function.

# Steps include:
# 1)Calculate n
# 2)Calculate the mean of both x and y numpy array.
# 3)Calculate cross-deviation and deviation: Just remember here we are calculating SS_xy and SS_xx which is Sum of Squared Errors.
# 4)Calculate regression coefficients: The amount or value by which the regression line needs to be moved.

# In[16]:


def estimate_coef(x, y): 
	# number of observations/points 
	n = np.size(x) 

	# mean of x and y vector 
	m_x, m_y = np.mean(x), np.mean(y) 

	# calculating cross-deviation and deviation about x 
	SS_xy = np.sum(y*x) - n*m_y*m_x 
	SS_xx = np.sum(x*x) - n*m_x*m_x 

	# calculating regression coefficients 
	b_1 = SS_xy / SS_xx 
	b_0 = m_y - b_1*m_x 

	return(b_0, b_1) 


# Function2: It is a function to plot the graph based on calculated values.
# Steps include:
# 1)Plot the points: “plt.scatter” plots the points on the graph where
# -“x and y” are the locations of the points on the graph
# -“color” is the colour of the plotted points change it to red or green or orange and play around for more possible colours check: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html
# -“marker” is the shape of the points like a circle or any other symbols for different types of marker check: https://matplotlib.org/api/markers_api.html#module-matplotlib.markers
# 
# 2. Predict the regression line value: Take the minimum error possible, the regression line is decided here.
# 
# 3. Plot the regression line
# 
# 4. Labels are put here instead of just x and y ie the name for x and y are put on the graph here.
# 
# 5. Show the plotted graph

# In[17]:


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


# Function3: Main function
# Steps include:
# 1)Gather the data sets required ie. x and y.
# 2)Calculate coefficients required ie. the value of moving of regression line in both x and y-direction.
# 3)Plot the graph
# 4)Lastly, write the main and call the main function:

# In[18]:


def main(): 
	# observations 
	x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
	y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12]) 

	# estimating coefficients 
	b = estimate_coef(x, y) 
	print("Estimated coefficients:\nb_0 = {}      \nb_1 = {}".format(b[0], b[1])) 

	# plotting regression line 
	plot_regression_line(x, y, b) 

if __name__ == "__main__": 
	main() 


# A linear regression model with two predictor variables can be expressed with the following equation:
# 
# Y = B0 + B1*X1 + B2*X2 + e.
# 
# The variables in the model are:
# Y, the response variable;
# X1, the first predictor variable;
# X2, the second predictor variable; and
# e, the residual error, which is an unmeasured variable.
# 
# The parameters in the model are:
# 
# B0, the Y-intercept;
# B1, the first regression coefficient; and
# B2, the second regression coefficient.
