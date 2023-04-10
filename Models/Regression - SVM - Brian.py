# Import necessary libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


# Load the COVID-19 dietary train and test data set 
X_train = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/regression_Xtrain.csv') 
X_test = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/regression_Xtest.csv')
y_train = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/regression_ytrain.csv')
y_test = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/regression_ytest.csv')

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the SVM regression model
svr = SVR(kernel='poly', degree=2)
svr.fit(X_train_scaled, y_train)

# Predict on the test set and calculate the mean squared error
y_pred_test = svr.predict(X_test_scaled)
y_pred_train = svr.predict(X_train_scaled)

# Evaluate the model's performance on the testing data
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)


# Print the performance metrics
print('Train Mean squared error:', mse_train)
print('Train R-Squared:', r2_train)
print('Test Mean squared error:', mse_test)
print('Test R-Squared:', r2_test)

# Plot the predicted values against the actual values
plt.scatter(y_test, y_pred_test)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('SVM Scatter Plot')
plt.show()