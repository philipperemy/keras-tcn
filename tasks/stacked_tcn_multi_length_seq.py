import numpy as np
from keras.layers import Dense
from keras.models import Input, Model

from tcn import TCN

# if you increase the sequence length make sure the receptive field of the TCN is big enough.
MAX_TIME_STEP = 30

"""
Input: sequence of length 7
Input: sequence of length 25
Input: sequence of length 29
Input: sequence of length 21
Input: sequence of length 20
Input: sequence of length 13
Input: sequence of length 9
Input: sequence of length 7
Input: sequence of length 4
Input: sequence of length 14
Input: sequence of length 10
Input: sequence of length 11
...
"""


def get_x_y(max_time_steps):
    for k in range(int(1e9)):
        time_steps = np.random.choice(range(1, max_time_steps), size=1)[0]
        if k % 2 == 0:
            x_train = np.expand_dims([np.insert(np.zeros(shape=(time_steps, 1)), 0, 1)], axis=-1)
            y_train = [1]
        else:
            x_train = np.array([np.zeros(shape=(time_steps, 1))])
            y_train = [0]
        print('\nInput: sequence of length {}\n'.format(time_steps))
        yield x_train, np.expand_dims(y_train, axis=-1)


i = Input(batch_shape=(1, None, 1))

o = TCN(return_sequences=False)(i)  # regression problem here.
o = Dense(1, activation='sigmoid')(o)

m = Model(inputs=[i], outputs=[o])
m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

size = 1000
gen = get_x_y(max_time_steps=MAX_TIME_STEP)
m.fit_generator(gen, epochs=3, steps_per_epoch=size, max_queue_size=1)
