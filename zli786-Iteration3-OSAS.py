# -*- coding: utf-8 -*-
"""
@author: zli786
"""
import pandas as pd
import seaborn as sns
import numpy as np
import sklearn.preprocessing as mpl
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,explained_variance_score,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Set the console to print more output
pd.set_option('display.max_columns', 20)

# Read the dataset
dataset = pd.read_csv("D:/1UOA MIT/Semester Two/INFOSYS 722/ASSIGNMENT/zli786-Iteration3-OSAS/dataset.csv")
print(dataset.head())
print(dataset.columns)
# The size of the dataset
print(dataset.shape)
# The details of dataset 
print(dataset.info())
# Generate descriptive statistics of the dataset 
print(dataset.describe(include="all"))
# count the null value
null = dataset.isnull().sum()
print("Null value in dataset: \n", null)

# Find the Missing values of more than 50%
remove = []
for i in dataset.columns:
    print("The columns need to be treated: ", i)
    if dataset[i].isnull().sum() > (dataset.shape[0] * 0.5):
        remove.append(i)
# Remove the variable
dataset1 = dataset.drop(remove, axis=1)
# Remove the missing value
dataset1 = dataset1.dropna()
print("The columns after removed the missing value:\n", dataset1.columns)
# Check the dataset if the missing value has been removed 
check_null = dataset1.isnull().any()
print(check_null)
print(dataset1.shape)

# The boxplot of each variables
sns.boxplot(x=dataset1['PM2.5'])
# sns.boxplot(x=dataset1['PM10'])
# sns.boxplot(x=dataset1['SO2'])
# sns.boxplot(x=dataset1['NO2'])
# sns.boxplot(x=dataset1['CO'])
# sns.boxplot(x=dataset1['O3'])
# sns.boxplot(x=dataset1['TEMP'])
# sns.boxplot(x=dataset1['PRES'])
# sns.boxplot(x=dataset1['DEWP'])
# sns.boxplot(x=dataset1['RAIN'])
# sns.boxplot(x=dataset1['WSPM'])

# Test the columns of the correlation with PM2.5  
print(dataset1.corr()["PM2.5"])

plt.figure(figsize=(9,9))
sns.heatmap(dataset1.corr(),linewidths=0.5,annot=True)

# Get the columns from the dataframe.
columns = dataset1.columns.tolist()
# Remove the columns with low correlation
columns = [c for c in columns if c not in ['No', 'year', 'PM10', 'station', 'wd', 'day', 'PRES']]
# Print the valid columns
print("Valid columns name: \n", columns)

# Remove the useless columns
dataset1 = dataset1.drop(['No', 'year', 'PM10', 'station', 'wd', 'day', 'PRES'], axis=1)
# Scale the data (data standardization) 
dataset2 = pd.DataFrame(mpl.scale(dataset1.drop(['PM2.5'], axis=1)))
# Set the target column - PM2.5 
target_column = dataset1['PM2.5']
# Reset the index of the columns starting from 0
target_column = target_column.reset_index()
target_column = target_column.iloc[:, 1:]
new_dataset = pd.concat([dataset2, target_column], axis=1)
# Standardization the target
new_dataset = new_dataset[new_dataset['PM2.5'] >= 0]
new_dataset['PM2.5'] = np.log2(new_dataset['PM2.5'] + 1)
new_dataset.rename(columns={0: 'month', 1: 'hour', 2: 'SO2', 3: 'NO2', 4: 'CO', 5: 'O3',
                            6: 'TEMP', 7: 'DEWP', 8: 'RAIN', 9: 'WSPM'}, inplace=True)
# print the new dataset 
print(new_dataset)
print(new_dataset.columns)

# Store the variable we'll be predicting on
# Generate the training set.  Set random_state to be able to replicate results.
x_train, x_test, y_train, y_test = train_test_split(
    new_dataset.drop(['PM2.5'], axis=1), new_dataset['PM2.5'], test_size=0.33, random_state=123)
# Print the shapes of both sets.
print("The size of x train: ",x_train.shape,", The size of y train: ", y_train.shape)
print("The size of x test: ",x_test.shape,", The size of y test: ", y_test.shape)
# Print the independent variables name in the trainning set
print(x_train.shape)
# Print the dependent variables name in the trainning set
print(y_train.name)


#### linear regression model
# Initialize the model class.
linear_model = LinearRegression(fit_intercept=True,normalize=False)
# Fit the model to the training data.
linear_model.fit(x_train, y_train)
importances = linear_model.coef_
# Returns the indices that would sort an array.
indices = np.argsort(importances)[::-1]
# Generate predictions for the test set
test_predictions = linear_model.predict(x_test)
# Generate predictions for the train set
train_predictions = linear_model.predict(x_train)
# compute the explained variance score for the model
score = explained_variance_score(y_test,test_predictions)
# Compute error between our test predictions and the actual values
linear_model_mse = mean_squared_error(y_test,test_predictions)
# Compute the absolute error between test predictions and the actual values
linear_model_mae = mean_absolute_error(y_test,test_predictions)
# print the score
print("Linear Regression model evaluation - testing set: ")
print("Explained variance score: ", score)
# print the mean squared error
print("Mean squared error: ", linear_model_mse)
# print the mean absolute error
print("Mean absolute error: ", linear_model_mae)
print("Linear Regression model evaluation - training set: ")
# print the score
print("Explained variance score: ", explained_variance_score(y_train,train_predictions))
# print the mean squared error
print("Mean squared error: ", mean_squared_error(y_train,train_predictions))
# print the mean absolute error
print("Mean absolute error: ", mean_absolute_error(y_train,train_predictions))
print(linear_model.coef_)
# show the result in bar chart
plt.figure()
plt.title("Feature importances(Linear Regression)")
plt.barh(range(x_train.shape[1]), importances[indices]
         , color='r', align="center")
plt.yticks(range(x_train.shape[1]), x_train.columns[indices])
plt.ylim([-1, x_train.shape[1]])
plt.show()

####Random forest model
# Initialize the model class.
random_forest_model = RandomForestRegressor(n_estimators=10,max_features="auto",random_state=123)
# Fit the model to the training data.
random_forest_model.fit(x_train, y_train)
# Compute the feature importance
importances = random_forest_model.feature_importances_
# Returns the indices that would sort an array.
indices = np.argsort(importances)[::-1]
# Generate predictions for the test set
test_predictions = random_forest_model.predict(x_test)
# Generate predictions for the train set
train_predictions = random_forest_model.predict(x_train)
# Compute the accuracy score for the model
accuracy = explained_variance_score(y_test,test_predictions)
# Compute error between our test predictions and the actual values
random_forest_model_mse = mean_squared_error(y_test, test_predictions)
# Compute the absolute error between test predictions and the actual values
random_forest_model_mae = mean_absolute_error(y_test,test_predictions)

# print the score
print("Random Forest model evaluation - testing set: ")
print("Explained variance score: ", accuracy)
# print the mean squared error
print("Mean squared error: ", random_forest_model_mse)
# print the mean absolute error
print("Mean absolute error: ", random_forest_model_mae)
print("Random Forest model evaluation - training set: ")
# print the score
print("Explained variance score: ", explained_variance_score(y_train,train_predictions))
# print the mean squared error
print("Mean squared error: ", mean_squared_error(y_train,train_predictions))
# print the mean absolute error
print("Mean absolute error: ", mean_absolute_error(y_train,train_predictions))
# Show the result in bar chart
plt.figure()
plt.title("Feature importances(Random Forest)")
plt.barh(range(x_train.shape[1]), importances[indices]
         , color='r', align="center")
plt.yticks(range(x_train.shape[1]), x_train.columns[indices])
plt.ylim([-1, x_train.shape[1]])
plt.show()

###XGBoost model
# Initialize the model class.
xgboost_model = xgb.XGBRFRegressor(n_estimators=10,seed=123)
# Fit the model to the training data.
xgboost_model.fit(x_train, y_train)
# Compute the feature importance
importances = xgboost_model.feature_importances_
# Returns the indices that would sort an array.
indices = np.argsort(importances)[::-1]

# Generate predictions for the test set
test_predictions = xgboost_model.predict(x_test)
# Generate predictions for the train set
train_predictions = xgboost_model.predict(x_train)
# Compute the accuracy score for the model
accuracy = explained_variance_score(y_test,test_predictions)
# Compute error between our test predictions and the actual values
xgboost_model_mse = mean_squared_error(y_test, test_predictions)
# Compute the absolute error between test predictions and the actual values
xgboost_model_mae = mean_absolute_error(y_test,test_predictions)

# print the score
print("XGBoost model evaluation - testing set: ")
print("Explained variance score: ", accuracy)
# print the mean squared error
print("Mean squared error: ", xgboost_model_mse)
# print the mean absolute error
print("Mean absolute error: ", xgboost_model_mae)
print("XGBoost model evaluation - training set: ")
# print the score
print("Explained variance score: ", explained_variance_score(y_train,train_predictions))
# print the mean squared error
print("Mean squared error: ", mean_squared_error(y_train,train_predictions))
# print the mean absolute error
print("Mean absolute error: ", mean_absolute_error(y_train,train_predictions))

# plot the feature importances of the XGBoost model 
plt.figure()
plt.title("Feature importances(XGBoost model)")
plt.barh(range(x_train.shape[1]), importances[indices]
         , color='r', align="center")
plt.yticks(range(x_train.shape[1]), x_train.columns[indices])
plt.ylim([-1, x_train.shape[1]])
plt.show()

plt(y='CO',x='PM2.5',data=new_dataset)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(linear_model,x_test,y_test)
