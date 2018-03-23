import numpy as np

import tcn
from mnist_pixel.utils import data_generator


def run_task():
    model, param_str = tcn.dilated_tcn(num_feat=1,
                                       num_classes=10,
                                       nb_filters=64,
                                       dilation_depth=1,
                                       nb_stacks=1,
                                       max_len=784,
                                       activation='wavenet',
                                       causal=True,
                                       return_param_str=True)

    (x_train, y_train), (x_test, y_test) = data_generator()

    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')
    print(f'x_test.shape = {x_test.shape}')
    print(f'y_test.shape = {y_test.shape}')

    model.fit(x_train, y_train, epochs=10)


if __name__ == '__main__':
    run_task()
