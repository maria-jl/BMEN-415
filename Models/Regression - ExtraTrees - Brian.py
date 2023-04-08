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
y_pred = et_reg.predict(X_test)

# Evaluate the model's performance on the testing data
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the performance metrics
print('Mean squared error:', mse)
print('R-Squared:', r2)

# Plot the predicted values against the actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('ExtraTrees Scatter Plot')
plt.show()