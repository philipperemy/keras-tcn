'''
#Trains a TCN on the IMDB sentiment classification task.
Output after 1 epochs on CPU: ~0.8611
Time per epoch on CPU (Core i7): ~64s.
Based on: https://github.com/keras-team/keras/blob/master/examples/imdb_bidirectional_lstm.py
'''
import keras
import numpy as np
from tensorflow.keras import Model, Input
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Dropout, Embedding
from tensorflow.keras.preprocessing import sequence

from tcn import TCN


def print_text(index_from_, x_):
    word_to_id = keras.datasets.imdb.get_word_index()
    word_to_id = {k: (v + index_from_) for k, v in word_to_id.items()}
    word_to_id['<PAD>'] = 0
    word_to_id['<START>'] = 1
    word_to_id['<UNK>'] = 2
    word_to_id['<UNUSED>'] = 3
    id_to_word = {value: key for key, value in word_to_id.items()}
    print(' '.join(id_to_word[ii] for ii in x_))


max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
max_len = 100
index_from = 3
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features, index_from=index_from)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

# print_text(index_from, x_test[0])
# print_text(index_from, x_test[1])
# print_text(index_from, x_test[2])
# print_text(index_from, x_test[3])
# print_text(index_from, x_test[4])

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

for i in range(10):
    print_text(index_from, x_test[i])

i = Input(shape=(max_len,))
x = Embedding(max_features, 128)(i)
x = TCN(nb_filters=64,
        kernel_size=6,
        dilations=[1, 2, 4, 8, 16, 32, 64])(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[i], outputs=[x])

model.summary()

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=3,
          validation_data=[x_test, y_test])
