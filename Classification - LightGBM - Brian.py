import lightgbm as ltb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Load split data into training and testing sets
X_train = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/classification_Xtrain.csv') 
X_test = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/classification_Xtest.csv')
y_train = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/classification_ytrain.csv')
y_test = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/classification_ytest.csv')

# Building model
model = ltb.LGBMClassifier()

# Fit the model and predict for test set
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

# Evaluate model performance
accuracy_test = accuracy_score(y_test, y_pred_test)
accuracy_train = accuracy_score(y_train, y_pred_train)
print("Train Accuracy:", accuracy_train)
print("Test Accuracy:", accuracy_test)

#confusion matrix
confusion = confusion_matrix(y_test, y_pred_test)
sns.heatmap(confusion, square = True, annot = True, cbar = False, cmap = 'icefire', 
            xticklabels = ['benign', 'malignant'], yticklabels = ['benign', 'malignant'])

plt.xlabel('predicted value')
plt.ylabel('true value')
plt.title('LightGBM Confusion Matrix')
plt.show()
