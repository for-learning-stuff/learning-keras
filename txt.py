fn = "glove.6B.50d.txt"

f = open(fn)

from keras.layers import LSTM

model = Sequential()
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))