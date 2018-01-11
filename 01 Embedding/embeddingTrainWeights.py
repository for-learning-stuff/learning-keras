from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding

docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']

# define class labels
labels = [1,1,1,1,1,0,0,0,0,0]

# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs] #hash each word
print(encoded_docs)

max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post') #add zeros after
print(padded_docs)

# define the model
model = Sequential()
#output shape of embedding: (lines of data (shown as None), number of words, number of features for each)
model.add(Embedding(vocab_size, 8, input_length=max_length))
#output shape of flatten: (lines of data (shown as None), number of words * number of features for each)
model.add(Flatten())

#does what??
#for linear activation
#[np.array([ [.1, .2, .5], [.1, .2, .5] ,[.1, .2, .5] ])   weights
# np.array([0,0,0])]            bias 
#input [[10,20,30]]

#[[10,20,30]]     [[.1, .2, .5],     [[6, 12, 30]]
#              X   [.1, .2, .5],  =  
#                  [.1, .2, .5]]
# 
#
# so for each line of 32 weights it will output a 0 or 1 based on the result of the sigmoid function
# output shape (None, 1), None is variabel describing number of lines of data and 1 is the result of the activation function       
model.add(Dense(1, activation='sigmoid')) 
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())

# epoches is an iteration over the entire x and y data provided
# a batch can be a subset of the entire data
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))
