import numpy as np
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

from tcn import TCN

i = Input(batch_shape=(None, 5, 300))
o = TCN(nb_filters=30, return_sequences=False, padding='same')(i)  # The TCN layers are here.
o = Dense(1)(o)
m = Model(inputs=[i], outputs=[o])
m.compile(optimizer='adam', loss='mse')
pred = m(np.random.rand(1, 5, 300))
