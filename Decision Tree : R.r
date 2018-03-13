rm(list=ls())

#import Dataset
dataset = read.csv("Position_Salaries.csv")
dataset = dataset[2:3]

#Creating Decision tree regressor 
#install.packages("rpart")
library(rpart)
regressor = rpart(formula = Salary ~ .,
                  data = dataset,
                  control = rpart.control(minsplit = 1))


#Visualizing decision tree model with more x axis points
x_grid = seq(min(dataset$Level), max(dataset$Level), .01)

library(ggplot2)
ggplot()+
  geom_point(aes(x = dataset$Level, y = dataset$Salary),  colour = 'red')+
  geom_line(aes(x= x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
                colour = 'blue')+
  ggtitle("Decision Tree Model")+
  xlab("Position Level")+
  ylab("Salart")

#prediction specific input result
level_inp = 6.5
y_pred = predict(regressor, newdata = data.frame(Level = level_inp))
