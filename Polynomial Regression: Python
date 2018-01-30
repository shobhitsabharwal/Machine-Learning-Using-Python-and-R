# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 22:14:59 2018

@author: Shobhit Sabharwal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import data set
dataset = pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

"""
#spliting dataset into train and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.8, random_state = 0)
"""

#Fitting linear regerssion to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#Fitting polynomial regerssion to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

#Visualising the Linear Regression Results

plt.scatter(x,y, color = "Red")
plt.plot(x, lin_reg.predict(x), color ="Blue")
plt.title("Linear Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#visualizing the polynomial Results
#x_grid = np.arange(min(x),max(x),0.1)  #this mkae vector x_grid
#x_grid = x_grid.reshape((len(x_grid),1)) #covert into matrix
plt.scatter(x,y, color = "Red")
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color ="Blue")
plt.title("Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


#predicting a new result with linear regression
input_posLevel = float(input("Input Position Level: "))
lin_reg.predict(input_posLevel)


#Predicting new result with polynominal regression
input_posLevel2 = float(input("Input Position Level: "))
lin_reg_2.predict(poly_reg.fit_transform(input_posLevel2))
