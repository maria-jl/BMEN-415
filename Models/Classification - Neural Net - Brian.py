import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  


X_train = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/classification_Xtrain.csv') 
X_test = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/classification_Xtest.csv')
y_train = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/classification_ytrain.csv')
y_test = pd.read_csv('C:/Users/brian/.vscode/BMEN 415 Code/classification_ytest.csv')

# Normalize
scaler = StandardScaler()  

# fit only on training data
scaler.fit(X_train)  
X_train_new = scaler.transform(X_train)  

# apply same transformation to test data
X_test_new = scaler.transform(X_test)  

# Create Model and fit 
NN = MLPClassifier(hidden_layer_sizes=(50, 50, 50), activation='tanh', solver='lbfgs', learning_rate_init=0.001)
NN.fit(X_train_new, y_train)

# Predict for test and train set
y_pred_test = NN.predict(X_test_new)
y_pred_train = NN.predict(X_train_new)

# Evaluate Model performance
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
plt.title('Neural Net Confusion Matrix')
plt.show()
