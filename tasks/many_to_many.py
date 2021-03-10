"""
#Trains a TCN on the IMDB sentiment classification task.
Output after 1 epochs on CPU: ~0.8611
Time per epoch on CPU (Core i7): ~64s.
Based on: https://github.com/keras-team/keras/blob/master/examples/imdb_bidirectional_lstm.py
"""
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector

from tcn import TCN

# many to many example.
# the input to the TCN model has the shape (batch_size, 24, 8),
# and the TCN output shape should have the shape (batch_size, 6, 2).

# We apply the TCN on the input sequence of length 24 to produce a vector of size 64
# (comparable to the last state of an LSTM). We repeat this vector 6 times to match the length
# of the output. We obtain an output_shape = (output_timesteps, 64) where each vector of size 64
# is identical (just duplicated output_timesteps times).
# From there, we apply a fully connected layer to go from a dim of 64 to output_dim.
# The kernel of this FC layer is (64, output_dim). That means each output_dim is parametrized by 64
# weights + 1 bias (applied at the TCN output, the RepeatVector does not have weights, just a reshape).

batch_size, timesteps, input_dim = 64, 24, 8
output_timesteps, output_dim = 6, 2

# dummy values here. There is nothing to learn. It's just to show how to do it.
batch_x = np.random.uniform(size=(batch_size, timesteps, input_dim))
batch_y = np.random.uniform(size=(batch_size, output_timesteps, output_dim))

model = Sequential(
    layers=[
        TCN(input_shape=(timesteps, input_dim)),  # output.shape = (batch, 64)
        RepeatVector(output_timesteps),  # output.shape = (batch, output_timesteps, 64)
        Dense(output_dim)  # output.shape = (batch, output_timesteps, output_dim)
    ]
)

model.summary()
model.compile('adam', 'mse')

print('Train...')
model.fit(batch_x, batch_y, batch_size=batch_size)
