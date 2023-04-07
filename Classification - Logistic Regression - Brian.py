import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Load split data into training and testing sets
X_train = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/classification_Xtrain.csv') 
X_test = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/classification_Xtest.csv')
y_train = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/classification_ytrain.csv')
y_test = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/classification_ytest.csv')

# Feature selection
selector = SelectKBest(f_classif, k=10) # Here, we are selecting top 10 features using ANOVA F-value
selector.fit(X_train, y_train)
X_train_new = selector.transform(X_train)
X_test_new = selector.transform(X_test)

# Create a logistic regression model
model = LogisticRegression()

# Train the model using the training data
model.fit(X_train_new, y_train)

# Predict target variable for testing data
y_pred_test = model.predict(X_test_new)
y_pred_train = model.predict(X_train_new)

# Calculate model accuracy
accuracy_test = accuracy_score(y_test, y_pred_test)
accuracy_train = accuracy_score(y_train, y_pred_train)
print("Train Accuracy:", accuracy_train)
print("Test Accuracy:", accuracy_test)

# Confusion matrix
confusion = confusion_matrix(y_test, y_pred_test)
sns.heatmap(confusion, square = True, annot = True, cbar = False, cmap = 'icefire', 
            xticklabels = ['benign', 'malignant'], yticklabels = ['benign', 'malignant'])

plt.xlabel('predicted value')
plt.ylabel('true value')
plt.title('Logistic Regression Confusion Matrix')
plt.show()
