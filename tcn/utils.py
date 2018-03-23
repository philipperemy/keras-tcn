import numpy as np


def mask_data(x, y, max_len=None, mask_value=0):
    if max_len is None:
        max_len = np.max([x.shape[0] for x in x])
    x_ = np.zeros([len(x), max_len, x[0].shape[1]]) + mask_value
    y_ = np.zeros([len(x), max_len, y[0].shape[1]]) + mask_value
    mask = np.zeros([len(x), max_len])
    for i in range(len(x)):
        l_ = x[i].shape[0]
        x_[i, :l_] = x[i]
        y_[i, :l_] = y[i]
        mask[i, :l_] = 1
    return x_, y_, mask[:, :, None]
