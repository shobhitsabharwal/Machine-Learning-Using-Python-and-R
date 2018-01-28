#Polimonial Regression##

#Import dataset
dataset = read.csv("Position_Salaries.csv")
dataset = dataset[2:3]  #Taking only Position level and Salary part

# Fitting linear regression to the dataset
lin_reg = lm(dataset$Salary ~ dataset$Level, data = dataset)
#summary(lin_reg)

#Fitting Polynomial Regression to the dataset
dataset$level2 = dataset$Level^2
dataset$level3 = dataset$Level^3
dataset$level4 = dataset$Level^4
poly_reg = lm(dataset$Salary~., data= dataset)
#summary(poly_reg)

#visualizing linear regression model 
library(ggplot2)
ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary, colour = "Red"))+
  geom_line(aes(x=dataset$Level, y = predict(lin_reg, newdata = dataset), colour="Blue"))+
  ggtitle("Linear Regression")+
  xlab("Positon Level")+
  ylab("Salary")

#Visualizing Polynomial regression model 
  library(ggplot2)
  ggplot() +
    geom_point(aes(x=dataset$Level, y=dataset$Salary, colour = "Red"))+
    geom_line(aes(x=dataset$Level, y = predict(poly_reg, newdata = dataset), colour="Blue"))+
    ggtitle("Polynomial Regression")+
    xlab("Positon Level")+
    ylab("Salary")
