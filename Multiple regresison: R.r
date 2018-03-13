#importing Dataset

dataset<-read.csv('50_Startups.csv')

#encode categorical data
dataset$State<-factor(dataset$State,
                          levels = c('New York','California', 'Florida'),
                          labels = c(1,2,3))

#spliting dataset into train and test
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit,SplitRatio = 0.8)
training_set = subset(dataset, split == 'TRUE')
test_set = subset(dataset, split == 'FALSE')


#Fitting Multiple Linear regression to training set
#regressor = lm(formula = Profit ~ ., data =training_set)
#regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, data = training_set)
regressor = lm(formula = Profit ~ R.D.Spend, data = training_set)
summary(regressor)

#Predicting the test results 
y_pred = predict(regressor, newdata = test_set)

#Building the optimal model using backword elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, data = training_set)
summary(regressor)

##Removing state2
regressor = lm(formula = Profit ~ R.D.Spend+Administration+Marketing.Spend+
                 factor(State ,exclude=c(2)),
               data = training_set)
summary(regressor)

#removing state
regressor = lm(formula = Profit ~ R.D.Spend+Administration+Marketing.Spend,
               data = training_set)
summary(regressor)

#removing Administration
regressor = lm(formula = Profit ~ R.D.Spend+Marketing.Spend,
               data = training_set)
summary(regressor)

#removing Marketing.spend
regressor = lm(formula = Profit ~ R.D.Spend,
               data = training_set)
summary(regressor)

y_pred = predict(regressor, newdata=test_set)

