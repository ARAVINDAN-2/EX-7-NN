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
```
# Breast Cancer Detection using MLP (Multi-Layer Perceptron)
# Dataset: Breast Cancer Wisconsin Dataset
# Import required libraries
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, 
confusion_matrix
# Load dataset
data = load_breast_cancer()
# Features and target
X = data.data
y = data.target
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Create MLP model
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # Two hidden layers
    max_iter=1000,
    learning_rate_init=0.001,
    random_state=42
)
# Train the model
mlp.fit(X_train, y_train)
# Predict on test data
y_pred = mlp.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Breast Cancer Detection using MLP")
print("----------------------------------")
print("Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Example prediction
sample = X_test[0].reshape(1, -1)
prediction = mlp.predict(sample)
if prediction[0] == 1:
    print("\nPrediction: Benign (Non-Cancerous)")
else:
    print("\nPrediction: Malignant (Cancerous)")
```
<H3>Output:</H3>

<img width="575" height="395" alt="image" src="https://github.com/user-attachments/assets/63338137-ebf2-41ae-95a3-4dbe76d22885" />


<H3>Results:</H3>
Thus, the Python program to determine breast cancer using Multi Layer Perceptron (MLP) was successfully implemented and tested.
