import argparse

import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.python.keras.layers import Dropout
from tqdm import tqdm

from tcn import TCN

nltk.download('punkt')


def split_to_sequences(ids, len_):
    x = np.zeros(shape=(len(ids) - len_ - 1, len_))
    y = np.zeros(shape=(len(ids) - len_ - 1, 1))
    for index in tqdm(range(0, len(ids) - len_ - 1)):
        x[index] = ids[index:index + len_]
        y[index] = ids[index + len_]
    return x, y


def main():
    parser = argparse.ArgumentParser(description='Sequence Modeling - The Word PTB')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--emb_size', type=int, default=200, help='embedding size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--seq_len', type=int, default=80, help='sequence length')
    parser.add_argument('--use_lstm', action='store_true')
    parser.add_argument('--task', choices=['char', 'word'])
    args = parser.parse_args()
    print(args)

    # Prepare dataset...
    with open('data/ptb.train.txt', 'r') as f1, \
            open('data/ptb.valid.txt', 'r') as f2, \
            open('data/ptb.test.txt', 'r') as f3:
        seq_train = f1.read().replace('<unk>', '')
        seq_valid = f2.read().replace('<unk>', '')
        seq_test = f3.read().replace('<unk>', '')

    if args.task == 'word':
        # split into words: [I, am, a, cat].
        seq_train = word_tokenize(seq_train)
        seq_valid = word_tokenize(seq_valid)
        seq_test = word_tokenize(seq_test)
    else:
        # split into characters: [I, ,a,m, ,a, ,c,a,t] ...
        seq_train = list(seq_train)
        seq_valid = list(seq_valid)
        seq_test = list(seq_test)

    vocab_train = set(seq_train)
    vocab_valid = set(seq_valid)
    vocab_test = set(seq_test)

    assert vocab_valid.issubset(vocab_train)
    assert vocab_test.issubset(vocab_train)
    size_vocab = len(vocab_train)

    # must have deterministic ordering for word2id dictionary to be reproducible
    vocab_train = sorted(vocab_train)
    word2id = {w: i for i, w in enumerate(vocab_train)}

    ids_train = [word2id[word] for word in seq_train]
    ids_valid = [word2id[word] for word in seq_valid]
    ids_test = [word2id[word] for word in seq_test]

    print(len(ids_train), len(ids_valid), len(ids_test))

    # Prepare inputs to model...
    x_train, y_train = split_to_sequences(ids_train, args.seq_len)
    x_val, y_val = split_to_sequences(ids_valid, args.seq_len)

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)

    # Define the model.
    if args.use_lstm:
        model = Sequential(layers=[
            Embedding(size_vocab, args.emb_size),
            Dropout(rate=0.2),
            LSTM(128),
            Dense(size_vocab, activation='softmax')
        ])
    else:
        # noinspection PyArgumentEqualDefault
        tcn = TCN(
            nb_filters=70,
            kernel_size=3,
            dilations=(1, 2, 4, 8, 16),
            use_skip_connections=True,
            use_layer_norm=True
        )
        print(f'TCN.receptive_field: {tcn.receptive_field}.')
        model = Sequential(layers=[
            Embedding(size_vocab, args.emb_size),
            Dropout(rate=0.2),
            tcn,
            Dense(size_vocab, activation='softmax')
        ])

    # Compile and train.
    model.summary()
    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        x_train, y_train,
        batch_size=args.batch_size,
        validation_data=(x_val, y_val),
        epochs=args.epochs
    )


if __name__ == '__main__':
    main()
