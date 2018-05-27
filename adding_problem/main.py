import keras

from tcn import tcn
from utils import data_generator

x_train, y_train = data_generator(n=200000, seq_length=600)
x_test, y_test = data_generator(n=40000, seq_length=600)


class PrintSomeValues(keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs={}):
        print(f'x_test[0:1] = {x_test[0:1]}.')
        print(f'y_test[0:1] = {y_test[0:1]}.')
        print(f'pred = {self.model.predict(x_test[0:1])}.')


def run_task():
    model, param_str = tcn.dilated_tcn(output_slice_index='last',
                                       num_feat=x_train.shape[2],
                                       num_classes=0,
                                       nb_filters=24,
                                       kernel_size=8,
                                       dilatations=[1, 2, 4, 8],
                                       nb_stacks=8,
                                       max_len=x_train.shape[1],
                                       activation='norm_relu',
                                       use_skip_connections=False,
                                       return_param_str=True,
                                       regression=True)

    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')

    psv = PrintSomeValues()

    # Using sparse softmax.
    # http://chappers.github.io/web%20micro%20log/2017/01/26/quick-models-in-keras/
    model.summary()

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=500,
              callbacks=[psv], batch_size=128)


if __name__ == '__main__':
    run_task()
