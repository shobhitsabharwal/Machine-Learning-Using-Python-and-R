# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 19:13:30 2018

@author: Shobhit Sabharwal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv("50_Startups.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()

#Avoiding dummy variable trap
x=x[:,1:]

#spliting the dataset into training and test 
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=.20, random_state=0)

#fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#predition of test set
y_pred=regressor.predict(x_test)

#####################Backward Elimination###############
#building the optimal model using backward elimination
import statsmodels.formula.api as sm
x = np.append(arr=np.ones((50,1)).astype(int), values = x, axis = 1)
x_opt = x[:,[0, 1, 2, 3, 4, 5]]  #intialization of x_opt matrix 
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()    

#removing column index 2
x_opt = x[:,[0, 1, 3, 4, 5]]  #intialization of x_opt matrix 
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()    

#removing column index 1
x_opt = x[:,[0, 3, 4, 5]]  #intialization of x_opt matrix 
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()   

#removing column index 4
x_opt = x[:,[0, 3, 5]]  #intialization of x_opt matrix 
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()   

#removing column index 5
x_opt = x[:,[0, 3]]  #intialization of x_opt matrix 
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()   
