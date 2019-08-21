# -*- coding: utf-8 -*-
"""
Auto Encoder (AE)
Stacked auto encoder that is trained on ratings users have given movies and 
gives a rating 1-5 on whether a user will like a movie they have not yet rated.

Created on Sat Aug 17 13:32:42 2019

@author: hayam
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# Prep training set & test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# Get total number users & movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Convert data to array with users in lines & movies in cols
def convert(data):
    new_data = []
    for id_user in range(1, nb_users + 1):
        # Get all movies rated by specific user
        id_movies = data[:, 1][data[:, 0] == id_user]
        id_ratings = data[:, 2][data[:, 0] == id_user]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Convert data to Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

"""
    Create architecture of Neural Network
    Stacked Auto Encoder - subclass of torch.nn.Module
"""
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        # Full connections
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        # Begin decoding
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activatation = nn.Sigmoid()
    
    """
        Action of establishing full connections by applying activation functions
        to activate right neurons in network
        Params: x - input vector of features
        
        @return x - vector of predicted ratings
    """
    def forward(self, x):
        x = self.activatation(self.fc1(x)) # first encoded vector
        x = self.activatation(self.fc2(x))
        x = self.activatation(self.fc3(x)) # first decoding
        x = self.fc4(x)
        return x


sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

# Train Stacked Auto Encoder (SAE)
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0. # count num users that rated at least 1 movie
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0) # create batch of a single input vector
        target = input.clone()
        # optimize memory by ignoring users that rated no movies
        if torch.sum(target.data > 0) > 0:
            output = sae(input) # forward function
            target.require_grad = False # won't compute gradient with respect to target
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10) # average of error for rated movies, make sure denominator is not 0
            loss.backward()
            train_loss += np.sqrt(loss.data * mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))

# Test SAE
test_loss = 0
s= 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data * mean_corrector)
        s += 1.
print('test loss: ' + str(test_loss / s))














        