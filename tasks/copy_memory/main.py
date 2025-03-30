from uuid import uuid4

import numpy as np
from tensorflow.keras.callbacks import Callback

from tcn import compiled_tcn
from utils import data_generator

x_train, y_train = data_generator(601, 10, 30000)
x_test, y_test = data_generator(601, 10, 6000)


class PrintSomeValues(Callback):

    def on_epoch_begin(self, epoch, logs={}):
        print('y_true')
        print(np.array(y_test[:5, -10:].squeeze(), dtype=int))
        print('y_pred')
        print(self.model.predict(x_test[:5])[:, -10:].argmax(axis=-1))


def run_task():
    model = compiled_tcn(num_feat=1,
                         num_classes=10,
                         nb_filters=10,
                         kernel_size=8,
                         dilations=[2 ** i for i in range(9)],
                         nb_stacks=1,
                         max_len=x_train[0:1].shape[1],
                         use_skip_connections=True,
                         opt='rmsprop',
                         lr=5e-4,
                         # use_weight_norm=True,
                         return_sequences=True)

    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')

    psv = PrintSomeValues()

    # Using sparse softmax.
    # http://chappers.github.io/web%20micro%20log/2017/01/26/quick-models-in-keras/
    model.summary()

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100,
              callbacks=[psv], batch_size=256)

    test_acc = model.evaluate(x=x_test, y=y_test)[1]  # accuracy.
    with open(f'copy_memory_{str(uuid4())[0:5]}.txt', 'w') as w:
        w.write(str(test_acc) + '\n')


if __name__ == '__main__':
    run_task()
