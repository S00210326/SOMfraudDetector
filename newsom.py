# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 15:31:02 2023
UNSUPERVISED DEEP LEARNING USING SOM TO DETECT
CUSTOMER FRAUD WITH CREDIT CARDS
@author: pgonigle
"""

"SELF ORGANIZING MAPS"

#Importing the libraries
import numpy as np
import matplotlib as plt
import pandas as pd


#Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#feature Scaling- normalise(get all between 0 and 1)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0, 1))
X = sc.fit_transform(X)

#training the SOM
from minisom.minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(X, num_iteration=100)

#VIsualise the results using pylab tools
from pylab import bone, pcolor, colorbar, plot, show
bone()#used to create blank figure with white background
pcolor(som.distance_map().T)#creates pseudocolor plot of matrix representing distances in the som
colorbar()#ADS color bar
markers = ['o', 's']#plot markers
colors = ['r', 'g']
for i, x in enumerate(X):
    W = som.winner(x)
    plot(W[0] + 0.5,
         W[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
    #these lines plot marker at posiion of winning nodees
show()

#finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(1,1)], mappings[(1,2)]), axis = 0)
frauds = sc.inverse_transform(frauds)