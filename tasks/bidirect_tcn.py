"""
#Trains a TCN on the IMDB sentiment classification task.
Output after 1 epochs on CPU: ~0.8611
Time per epoch on CPU (Core i7): ~64s.
Based on: https://github.com/keras-team/keras/blob/master/examples/imdb_bidirectional_lstm.py
"""
import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Embedding, Bidirectional
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

inputs = Input(shape=(None,), dtype="int32")
x = Embedding(max_features, 128)(inputs)
x = Bidirectional(TCN(64))(x)
# Add a classifier
outputs = Dense(1, activation="sigmoid")(x)
model = Model(inputs, outputs)
model.summary()

print(f'Backward TCN receptive field: {model.layers[2].backward_layer.receptive_field}.')
print(f'Forward TCN receptive field: {model.layers[2].forward_layer.receptive_field}.')

model.summary()
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(
    x_train, y_train,
    batch_size=batch_size,
    validation_data=[x_test, y_test]
)
