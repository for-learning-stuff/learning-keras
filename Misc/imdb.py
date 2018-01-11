import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


np.random.seed(7)

top_words = 5000 #the top frequent words to keep, replace others with oov_char

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)
# print x_train[0]

max_review_length = 100

#limit the length of reviews
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

print x_train.shape

model = Sequential()
model.add(Embedding(top_words, 100)) #input_length is only necessary if being passed to Flatten and Dense
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
print model.summary()
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)