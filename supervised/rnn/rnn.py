# -*- coding: utf-8 -*-
"""
Reccurent Neural Network (RNN)
Deep learning model which predicts Google stock price

Created on Mon Aug 12 14:07:57 2019

@author: Haya Majeed
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv') # import training set
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set) # fit, scale, and transform sc to training set

# Create data structure with 60 timesteps and 1 output
X_train = []
y_train = []

# populate X_train with stock prices
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60: i, 0]) # 60 previous stock prices
    y_train.append(training_set_scaled[i, 0]) # stock price at time T + 1
X_train, y_train = np.array(X_train), np.array(y_train)
    
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialize RNN
regressor = Sequential()

# Add first LSTM layer & dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2)) # 20% neurons of neurons of LSTM layer ignored during training (forward/backward propogation)

# Second LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Third LSTM Layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Fourth LSTM layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Output layer
regressor.add(Dense(units = 1)) # Stock price at time T + 1

# Compile RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fit RNN to Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32) 

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv') # import actual stock price of 2017
real_stock_price = dataset_test.iloc[:, 1:2].values

# Get predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0) # concat against vertical axis
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values # Jan 3 - 3 months up to end of year
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
# populate X_test with stock prices
for i in range(60, 80):
    X_test.append(inputs[i-60: i, 0]) # 60 previous stock prices
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) # 3D structure of inputs
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price) # inverse scaling of predictions

# Visualize results
plt.plot(real_stock_price, color = 'red', label = 'Actual Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('RNN Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()





