# import csv

fn = "tmdb_5000_movies.csv"

# d = open("tmdb_5000_movies.csv")

# from numpy import genfromtxt
# data = genfromtxt(fn, delimiter=',')

# print data
# lines = 4807
# features = 20
# print "f"
# data = list(csv.reader(d, delimiter=','))

import numpy as np
import pandas as pd

movies = pd.read_csv(fn)
print movies.head()

# from keras.models import Sequential
# from keras.layers import Dense, Activation
# from keras.layers import LSTM

# model = Sequential([
#     Dense(32, input_shape=(features,)),
#     Activation('relu'),
#     Dense(10),
#     Activation('softmax'),
# ])