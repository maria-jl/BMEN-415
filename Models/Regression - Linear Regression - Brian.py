import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression


# Load split data into training and testing sets
X_train = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/regression_Xtrain.csv') 
X_test = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/regression_Xtest.csv')
y_train = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/regression_ytrain.csv')
y_test = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/regression_ytest.csv')

# Perform feature selection on the training set only
selector = SelectKBest(f_regression, k=2)
X_train_new = selector.fit_transform(X_train, y_train)


# Create a LinearRegression instance and fit the model to the training data
linreg = LinearRegression()
linreg.fit(X_train_new, y_train)

# Transform the testing set using the same feature selector
X_test_new = selector.transform(X_test)

# Make predictions on the testing data and evaluate the performance of the model
y_pred = linreg.predict(X_test_new)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean squared error:', mse)
print('R-squared:', r2)

# Plot the predicted values against the actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression Scatter Plot')
plt.show()
