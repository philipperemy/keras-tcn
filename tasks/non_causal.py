import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from tcn import TCN

# Look at the README.md to know what is a non-causal case.

model = Sequential([
    TCN(nb_filters=30, padding='same', input_shape=(5, 300)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
pred = model.predict(np.random.rand(1, 5, 300))
print(pred.shape)
