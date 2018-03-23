import keras

from copy_memory.utils import data_generator
from tcn import tcn

x_train, y_train = data_generator(601, 10, 100000)
x_test, y_test = data_generator(601, 10, 20000)


class PrintSomeValues(keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs={}):
        print(f'x_test[0:1] = {x_test[0:1].flatten()}.')
        print(f'y_test[0:1] = {y_test[0:1].flatten()}.')
        print(f'p(x_test[0:1]) = {self.model.predict(x_test[0:1]).argmax(axis=2).flatten()}.')


def run_task():
    print(sum(x_train[0].tolist(), []))
    print(sum(y_train[0].tolist(), []))

    model, param_str = tcn.dilated_tcn(num_feat=1,
                                       num_classes=10,
                                       nb_filters=10,
                                       dilation_depth=6,
                                       nb_stacks=6,
                                       max_len=x_train[0:1].shape[1],
                                       activation='wavenet',
                                       use_skip_connections=False,
                                       causal=True,
                                       return_param_str=True)

    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')

    psv = PrintSomeValues()

    # Using sparse softmax.
    # http://chappers.github.io/web%20micro%20log/2017/01/26/quick-models-in-keras/
    model.summary()

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100,
              callbacks=[psv])


if __name__ == '__main__':
    run_task()
