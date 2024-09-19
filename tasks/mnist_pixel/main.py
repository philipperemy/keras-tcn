from utils import data_generator

from tcn import compiled_tcn


def run_task():
    (x_train, y_train), (x_test, y_test) = data_generator()

    model = compiled_tcn(return_sequences=False,
                         num_feat=1,
                         num_classes=10,
                         nb_filters=20,
                         kernel_size=6,
                         dilations=[2 ** i for i in range(9)],
                         nb_stacks=1,
                         max_len=x_train[0:1].shape[1],
                         use_skip_connections=True)

    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')
    print(f'x_test.shape = {x_test.shape}')
    print(f'y_test.shape = {y_test.shape}')

    model.summary()

    model.fit(x_train, y_train.squeeze().argmax(axis=1), epochs=100,
              validation_data=(x_test, y_test.squeeze().argmax(axis=1)))


if __name__ == '__main__':
    run_task()
