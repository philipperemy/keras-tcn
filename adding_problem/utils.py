import numpy as np


def data_generator(n, seq_length):
    """
    Args:
        seq_length: Length of the adding problem data
        n: # of data in the set
    """
    x_num = np.random.uniform(0, 1, (n, 1, seq_length))
    x_mask = np.zeros([n, 1, seq_length])
    y = np.zeros([n, 1])
    for i in range(n):
        positions = np.random.choice(seq_length, size=2, replace=False)
        x_mask[i, 0, positions[0]] = 1
        x_mask[i, 0, positions[1]] = 1
        y[i, 0] = x_num[i, 0, positions[0]] + x_num[i, 0, positions[1]]
    x = np.concatenate((x_num, x_mask), axis=1)
    x = np.transpose(x, (0, 2, 1))
    return x, y


if __name__ == '__main__':
    print(data_generator(n=20, seq_length=10))
