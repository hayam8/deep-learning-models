# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:01:12 2019

@author: hayam
"""

# Unsupervised Learning to create self-organizing map of potential CC application frauds

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training SOM
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualize results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x) # winning node of customer x
    # plot at center of square
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Find potential frauds
mappings = som.win_map(X)
# Coordinates of white marker for winning node. Use concatenate if more than one white winning node
frauds = np.concatenate((mappings[(2, 7)], mappings[(4, 7)]), axis = 0)
frauds = sc.inverse_transform(frauds)


# Unsupervised Learning to Supervised Learning

# Matrix of features
customers = dataset.iloc[:, 1:].values # all columns except first column of customer ID

is_fraud = np.zeros(len(dataset)) # Dependent variable
for i in range(len(dataset)):
    if(dataset.iloc[i, 0] in frauds):
        is_fraud[i] = 1

# Artificial Neural Network (ANN) 
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Import Keras libraries & packages
import keras
from keras.models import Sequential
from keras.layers import Dense # Used to create layers in ANN

# Initialize ANN
classifier = Sequential()

# Add input layer & first hidden layer
#classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(activation = 'relu', units = 2, kernel_initializer = 'uniform', input_dim = 15))

# Add output layer
classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform')) 

# Compile ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit classifier ANN to training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)

# Predict probablities of frauds
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1) # Add column for customer ids to prediction
y_pred = y_pred[y_pred[:, 1].argsort()] # sort array by column for probability 





























