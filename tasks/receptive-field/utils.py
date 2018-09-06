import numpy as np


def data_generator(batch_size=1024, sequence_length=32):
    # input image dimensions
    pos_indices = np.random.choice(batch_size, size=int(batch_size // 2), replace=False)

    x_train = np.zeros(shape=(batch_size, sequence_length))
    y_train = np.zeros(shape=(batch_size, 1))
    x_train[pos_indices, 0] = 1.0
    y_train[pos_indices, 0] = 1.0

    # y_train = to_categorical(y_train, num_classes=2)

    return np.expand_dims(x_train, axis=2), y_train


if __name__ == '__main__':
    print(data_generator(batch_size=3, sequence_length=4))
