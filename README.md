<H3>NAME. ARAVINDAN SD</H3>
<H3>REG NO. 212224243001</H3>
<H3>EX. NO.07</H3>
<H3>DATE:25.05.2026</H3>
<H1 ALIGN =CENTER>Breast Cancer Detection Using MLP in Python</H1>
<H3>Aim:</H3>  To construct a Python program to determine breast cancer using tabulated data with a Multi Layer Perceptron (MLP) classifier.<BR>
<H3>Algorithm:</H3>
Step 1:Import required libraries.<BR>
Step 2:Load the Breast Cancer dataset.<BR>
Step 3:Split the dataset into training and testing data.<BR>
Step 4:Standardize the dataset using StandardScaler.<BR>
Step 5:Create the MLP classifier model.<BR>
Step 6:Train the model using training data.<BR>
Step 7:Predict the output for test data.<BR>
Step 8:Calculate accuracy and display classification report.<BR>
Step 9:Print the result.<BR>
<H3>Program: </H3>
# Breast Cancer Detection using MLP (Multi-Layer Perceptron)<BR>
# Dataset: Breast Cancer Wisconsin Dataset<BR>

import pandas as pd<BR>
from sklearn.datasets import load_breast_cancer<BR>
from sklearn.model_selection import train_test_split<BR>
from sklearn.preprocessing import StandardScaler<BR>
from sklearn.neural_network import MLPClassifier<BR>
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix<BR>

data = load_breast_cancer()<BR>

X = data.data<BR>
y = data.target<BR>

X_train, X_test, y_train, y_test = train_test_split(<BR>
    X, y, test_size=0.2, random_state=42<BR>
)<BR>

scaler = StandardScaler()<BR>
X_train = scaler.fit_transform(X_train)<BR>
X_test = scaler.transform(X_test)<BR>

mlp = MLPClassifier(<BR>
    hidden_layer_sizes=(100, 50),<BR>
    max_iter=1000,<BR>
    learning_rate_init=0.001,<BR>
    random_state=42<BR>
)<BR>

mlp.fit(X_train, y_train)<BR>


y_pred = mlp.predict(X_test)<BR>

accuracy = accuracy_score(y_test, y_pred)<BR>

print("Breast Cancer Detection using MLP")<BR>
print("----------------------------------")<BR>
print("Accuracy:", accuracy)<BR>

print("\nConfusion Matrix:")<BR>
print(confusion_matrix(y_test, y_pred))<BR>

print("\nClassification Report:")<BR>
print(classification_report(y_test, y_pred))<BR>
<H3>Output:</H3>

<img width="1296" height="744" alt="image" src="https://github.com/user-attachments/assets/4adcb5e2-1f85-4920-841c-f132acbcde37" />


<H3>Results:</H3>
Thus, the Python program to determine breast cancer using Multi Layer Perceptron (MLP) was successfully implemented and tested.
