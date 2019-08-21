# -*- coding: utf-8 -*-
"""
Restricted Boltzmann Machine (RBM)
Unsupervised deep learning model which takes in a data set of movies, ratings, and details. 
Takes in another data set for information about users, which movies they rated, and what they rated the movie
in order to predict if a user will like a specific movie.

Created on Thu Aug 15 12:43:07 2019

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

# Data set
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1') # movieID, movie title, genre
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1') # userID, gender, age, work, zip code
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1') # userID, movieID, rating (1-5), timestamp

# Training set & test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t') # userID, movieID, rating, timestamp
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t') # userID, movieID, rating, timestamp
test_set = np.array(test_set, dtype='int')

# Get num users & movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Convert data into array of users & movies
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users] # col for movieIDs of specific user
        id_rating = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(1682)
        ratings[id_movies - 1] = id_rating # list of ratings for specific user
        new_data.append(list(ratings)) # add list of movie ratings for specific user to list of ratings
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Convert data to Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Convert ratings to binary 1 (liked) 0 (did not like)
training_set[training_set == 0] = -1 # convert all unrated movies (0s) in set to -1
# ratings of 1 & 2 converted to 0 (unliked)
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
# ratings of 3, 4, & 5 converted to 1 (liked)
training_set[training_set >= 3] = 1
# repeat for test_set
test_set[test_set == 0] = -1 
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Create architecture of Neural Network
class RBM():
    """
        Params: nv = number of visible nodes,
                nh = number of hidden nodes
    """
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv) # random weights
        self.a = torch.randn(1, nh) # bias for hidden nodes given visible nodes
        self.b = torch.randn(1, nv) # bias for visible nodes given hidden nodes
        
    """
        Params: x = v (visible neurons) in probability of h (hidden neuron) given v (visible neuron)
    
        p_h_given_v = probability hidden node activated given visible node
        
        @return p_h_given_v and sampling of hidden neurons
    """
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t()) # two torch tensors
        # activation function
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    """
        Params: y = h (hidden neurons) in probability of v (visible neuron) being 1 given h (hidden neuron)
    
        p_v_given_h = probability each visible node equals 1 given values of activated hidden nodes
        
        @return p_v_given_h and sampling of visible nodes
    """
    def sample_v(self, y):
        wy = torch.mm(y, self.W) # two torch tensors
        # activation function
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    """
        Approximating the RBL Log-Likelihood 
        Params: v0 == input vector containing ratings of all movies by one user
                vk == visible nodes obtained after k-samplings
                ph0 == vector of probabilities that at first iteration hidden nodes == 1 given the values of v0
                phk == probabilities of hidden nodes after k-samplings given the values of the visible nodes vk
    """
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0) # difference between input vector of operations and visible nodes after k-samplings
        self.a += torch.sum((ph0 - phk), 0)
    

nv = len(training_set[0]) # len first line of training set, features
nh = 100 # hidden nodes
batch_size = 100
rbm = RBM(nv, nh)

# Train the RBM
nb_epoch = 10
for epoch in range(nb_epoch):
    train_loss = 0
    counter = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user : id_user + batch_size]
        v0 = training_set[id_user : id_user + batch_size] # visible nodes
        ph0,_ = rbm.sample_h(v0)
        # k step contrastive divergence
        for k in range(10):
            _,hk = rbm.sample_h(vk) # hidden nodes obtained at k-th step contrastive divergence
            _,vk = rbm.sample_v(hk) # sampled visible nodes after step of sampling
            vk[v0<0] = v0[v0<0] # make sure training isn't done on movies that aren't rated (-1)
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>-1] - vk[vk>-1]))
        counter += 1.
    # Print updates during training of epoch and normalized loss
    print("epoch: " + str(epoch) + " loss: " + str(train_loss / counter))

# Test the RBM
test_loss = 0
counter = 0.
for id_user in range(nb_users):
    v = training_set[id_user : id_user + 1] # input
    vt = test_set[id_user : id_user + 1] # target
    # k step contrastive divergence
    if(len(vt[vt > -1]) > 0):
        _,h = rbm.sample_h(v) # hidden node
        _,v = rbm.sample_v(h) # visible node
        test_loss += torch.mean(torch.abs(vt[vt >- 1] - v[vt >- 1]))
        counter += 1.
# Print updates of normalized test loss
print("test loss: " + str(test_loss / counter))













        
        
        
        
        
        
        