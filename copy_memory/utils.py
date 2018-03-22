import numpy as np


def data_generator(t, mem_length, b_size):
    """
    Generate data for the copying memory task
    :param t: The total blank time length
    :param mem_length: The length of the memory to be recalled
    :param b_size: The batch size
    :return: Input and target data tensor
    """
    seq = np.array(np.random.randint(1, 9, size=(b_size, mem_length)), dtype=float)
    zeros = np.zeros((b_size, t))
    marker = 9 * np.ones((b_size, mem_length + 1))
    placeholders = np.zeros((b_size, mem_length))

    x = np.concatenate((seq, zeros[:, :-1], marker), 1)
    y = np.array(np.concatenate((placeholders, zeros, seq), 1), dtype=int)
    return x, y


if __name__ == '__main__':
    print(data_generator(2, 3, 1))
