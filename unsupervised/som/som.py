# -*- coding: utf-8 -*-
"""
Self Organizing Map (SOM)
Unsupervised deep learning model which organizes credit card applications for customers into a map
in order to predict which customer was likely to submit a fraudulent application.

Created on Tue Aug 13 12:34:41 2019

@author: Haya Majeed
"""

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
frauds = np.concatenate((mappings[(1, 8)], mappings[(4, 8)]), axis = 0)
frauds = sc.inverse_transform(frauds)















