# Import necessary libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the COVID-19 dietary train and test data set 
X_train = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/regression_Xtrain.csv') 
X_test = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/regression_Xtest.csv')
y_train = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/regression_ytrain.csv')
y_test = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/regression_ytest.csv')

# Initialize the ExtraTrees Regressor model with hyperparameters
et_reg = ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42)

# Fit the model to the training data
et_reg.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_test = et_reg.predict(X_test)
y_pred_train = et_reg.predict(X_train)

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
plt.title('ExtraTrees Scatter Plot')
plt.show()