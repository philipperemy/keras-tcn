import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# export CUDA_VISIBLE_DEVICES=0; nohup python -u train.py --use_lstm --batch_size 256 --task char > lstm.log 2>&1 &
# export CUDA_VISIBLE_DEVICES=1; nohup python -u train.py --batch_size 256 --task char > tcn.log 2>&1 &
# Usage: python plot.py tcn.log lstm.log

def keras_output_to_data_frame(filename) -> pd.DataFrame:
    suffix = Path(filename).stem
    headers = None
    data = []
    with open(filename) as r:
        lines = r.read().strip().split('\n')
    for line in lines:
        if 'ETA' not in line and 'loss' in line:
            matches = re.findall('[a-z_]+: [0-9]+.[0-9]+', line)
            headers = [m.split(':')[0] + '_' + suffix for m in matches]
            data.append([float(m.split(':')[1]) for m in matches])
    return pd.DataFrame(data, columns=headers)


def main():
    dfs = []
    colors = ['darkviolet', 'violet', 'deepskyblue', 'skyblue']
    for i, argument in enumerate(sys.argv):
        if i == 0:
            continue
        dfs.append(keras_output_to_data_frame(argument))
    m = pd.concat(dfs, axis=1)
    accuracy_columns = [c for c in list(m.columns) if 'acc' in c]
    loss_columns = [c for c in list(m.columns) if 'loss' in c]
    _, axs = plt.subplots(ncols=2, figsize=(12, 7), dpi=150)
    m.plot(y=accuracy_columns, title='Accuracy', legend=True, xlabel='epoch', color=colors,
           ylabel='accuracy', sort_columns=True, grid=True, ax=axs[0])
    plt.figure(1, figsize=(2, 5))
    m.plot(y=loss_columns, title='Loss', legend=True, color=colors,
           xlabel='epoch', ylabel='loss', sort_columns=True, grid=True, ax=axs[1])
    plt.savefig('result.png')
    plt.close()


if __name__ == '__main__':
    main()
