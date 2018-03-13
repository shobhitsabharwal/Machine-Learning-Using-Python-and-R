# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 21:57:00 2018

@author: Shobhit Sabharwal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import Dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

#Making Decision regression model 
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

#this is not continues model 
#We need to make more points for independent variable
#Visualizing SVM model
x_grid = np.arange(min(x), max(x), 0.001)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('SVR model')
plt.show()


#Prediction for specific input value using SVM model 
val = float(input("Input Value to predict Salary: "))
y_pred = regressor.predict(val)
print("Salary with position Level %.2f is %.2f" % (val,y_pred))
