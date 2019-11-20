"""
#Trains a TCN on the IMDB sentiment classification task.
Output after 1 epochs on CPU: ~0.8611
Time per epoch on CPU (Core i7): ~64s.
Based on: https://github.com/keras-team/keras/blob/master/examples/imdb_bidirectional_lstm.py
"""
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Dropout, Embedding
from tensorflow.keras.preprocessing import sequence

from tcn import TCN

max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 100
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(max_features, 128, input_shape=(maxlen,)))
model.add(TCN(nb_filters=64,
              kernel_size=6,
              dilations=[1, 2, 4, 8, 16, 32, 64]))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])


class TestCallback(Callback):

    def on_epoch_end(self, epoch, logs=None):
        print(logs)
        acc_key = 'val_accuracy' if 'val_accuracy' in logs else 'val_acc'
        assert logs[acc_key] > 0.78


print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1,
          validation_data=[x_test, y_test],
          callbacks=[TestCallback()])
