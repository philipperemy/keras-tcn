import numpy as np
from sklearn.preprocessing import MinMaxScaler


def get_xy_kfolds(split_index=[0.5, 0.6, 0.7, 0.8, 0.9], timesteps=1000):
    """
    load exchange rate dataset and preprecess it, then split it into k-folds for CV
    :param split_index: list, the ratio of whole dataset as train set
    :param timesteps: length of a single train x sample
    :return: list, [train_x_set,train_y_set,test_x_single,test_y_single]
    """
    df = np.loadtxt('exchange_rate.txt', delimiter=',')
    n = len(df)
    folds = []
    enc = MinMaxScaler()
    df = enc.fit_transform(df)
    for split_point in split_index:
        train_end = int(split_point * n)
        train_x, train_y = [], []
        for i in range(train_end - timesteps):
            train_x.append(df[i:i + timesteps])
            train_y.append(df[i + timesteps])
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = df[train_end - timesteps + 1:train_end + 1]
        test_y = df[train_end + 1]
        folds.append((train_x, train_y, test_x, test_y))
    return folds, enc


if __name__ == '__main__':
    print(get_xy_kfolds())
