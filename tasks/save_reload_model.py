import os

import numpy as np
from keras import Model, Input
from keras.layers import Dense, Dropout, Embedding

from tcn import TCN

# simple TCN model.
max_len = 100
max_features = 50
i = Input(shape=(max_len,))
x = Embedding(max_features, 16)(i)
x = TCN(nb_filters=12,
        dropout_rate=0.5,  # with dropout here.
        kernel_size=6,
        dilations=[1, 2, 4])(x)
x = Dropout(0.5)(x)  # and dropout here.
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[i], outputs=[x])

if os.path.exists('tcn.npz'):
    # Load checkpoint if file exists.
    w = np.load('tcn.npz', allow_pickle=True)['w']
    print('Model reloaded.')
    model.set_weights(w.tolist())
else:
    # Save the checkpoint.
    w = np.array(model.get_weights())
    np.savez_compressed(file='tcn.npz', w=w, allow_pickle=True)
    print('First time.')

# Make inference.
# The value for [First time] and [Model reloaded] should be the same. Run the script twice!
inputs = np.ones(shape=(1, 100))
out1 = model.predict(inputs)[0, 0]
print('*' * 80)
print(out1)
print('*' * 80)
