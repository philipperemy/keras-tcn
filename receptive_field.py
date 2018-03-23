# simulation from deepmind paper: https://arxiv.org/pdf/1609.03499.pdf
import numpy as np


def res_block():
    pass


def simulation():
    d = np.ones(shape=(4, 16))
    print(d)
    dilated_factor = 2
    for j in range(4)[::-1]:
        for i in list(range(0, 16, dilated_factor ** (3 - j))):
            a = 2


if __name__ == '__main__':
    simulation()
