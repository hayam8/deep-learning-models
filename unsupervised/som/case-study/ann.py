# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 08:30:22 2019

@author: hayam
"""

# Data Proprocessing 
# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset= pd.read_csv('Churn_Modelling.csv')
# ind vars : credit score, country, gender, age, tenure, balance, # products, CC card, active member, est salary
X = dataset.iloc[:, 3:13].values # take all but last column
y = dataset.iloc[:, 13].values

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder() # Country
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder() # Gender
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1]) # Dummy var to equalize values
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Import Keras libraries & packages
import keras
from keras.models import Sequential
from keras.layers import Dense # Used to create layers in ANN

# Initialize ANN
classifier = Sequential()

# Add input layer & first hidden layer
#classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(activation = 'relu', units = 6, kernel_initializer = 'uniform', input_dim = 11)) # units = (11 + 2) / 2, kernel_initializer initiate units randomly, rectifier activation function

# Add second hidden layer
classifier.add(Dense(activation = 'relu', units = 6, kernel_initializer = 'uniform'))

# Add output layer
classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform')) # activation = sigmoid activation function

# Compile ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit classifier ANN to training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predict test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predict single new observation
"""
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000
new_prediction = classifier.predict(sc_X.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)
"""

# Make confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

