"""
#Trains a TCN on the IMDB sentiment classification task.
Output after 1 epochs on CPU: ~0.8611
Time per epoch on CPU (Core i7): ~64s.
Based on: https://github.com/keras-team/keras/blob/master/examples/imdb_bidirectional_lstm.py
"""
import os
import shutil

import keract  # pip install keract
import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Dropout, Embedding
from tensorflow.keras.preprocessing import sequence

from tcn import TCN

index_from_ = 3


def visualize(model, x, max_len, tcn_num_filters, tcn_layer_outputs, prefix):
    for i in range(len(x)):
        tcn_outputs = keract.get_activations(model, x[i:i + 1], nodes_to_evaluate=tcn_layer_outputs)
        tcn_blocks_outputs = [v for (k, v) in tcn_outputs.items() if v.shape == (1, max_len, tcn_num_filters)]
        plt.figure(figsize=(10, 2))  # creates a figure 10 inches by 10 inches
        plt.title('TCN internal outputs (one row = one residual block output)')
        plt.xlabel('Timesteps')
        plt.ylabel('Forward pass\n (top to bottom)')
        plt.imshow(np.max(np.vstack(tcn_blocks_outputs), axis=-1), cmap='jet', interpolation='bilinear')
        plt.savefig(f'acts/{prefix}_{i}.png', dpi=1000, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()


def get_word_mappings():
    word_to_id_dict = keras.datasets.imdb.get_word_index()
    word_to_id_dict = {k: (v + index_from_) for k, v in word_to_id_dict.items()}
    word_to_id_dict['<PAD>'] = 0
    word_to_id_dict['<START>'] = 1
    word_to_id_dict['<UNK>'] = 2
    word_to_id_dict['<UNUSED>'] = 3
    id_to_word_dict = {value: key for key, value in word_to_id_dict.items()}
    return word_to_id_dict, id_to_word_dict


def encode_text(x_):
    word_to_id, id_to_word = get_word_mappings()
    return [1] + [word_to_id[a] for a in x_.lower().replace('.', '').strip().split(' ')]


def print_text(x_):
    word_to_id, id_to_word = get_word_mappings()
    print(' '.join(id_to_word[ii] for ii in x_))


def main():
    max_features = 20000
    # cut texts after this number of words
    # (among top max_features most common words)
    max_len = 100
    batch_size = 32
    tcn_num_filters = 10

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features, index_from=index_from_)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    x_val = [
        encode_text('The movie was very good. I highly recommend.'),  # will be at the end.
        encode_text(' '.join(["worst"] * 100)),
        encode_text("Put all speaking her delicate recurred possible. "
                    "Set indulgence discretion insensible bed why announcing. "
                    "Middleton fat two satisfied additions. "
                    "So continued he or commanded household smallness delivered. "
                    "Door poor on do walk in half. "
                    "Roof his head the what. "
                    "Society excited by cottage private an it seems. "
                    "Fully begin on by wound an. "
                    "The movie was very good. I highly recommend. "
                    "At declared in as rejoiced of together. "
                    "He impression collecting delightful unpleasant by prosperous as on. "
                    "End too talent she object mrs wanted remove giving. "
                    "Man request adapted spirits set pressed. "
                    "Up to denoting subjects sensible feelings it indulged directly.")
    ]

    y_val = [1, 0, 1]

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    x_val = sequence.pad_sequences(x_val, maxlen=max_len)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print('x_val shape:', x_val.shape)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_val = np.array(y_val)

    x_val[x_val > max_features] = 2  # oov.

    for i in range(10):
        print(f'x_test[{i}]=', end=' | ')
        print_text(x_test[i])

    for i in range(len(x_val)):
        print(f'x_val[{i}]=', end=' | ')
        print_text(x_val[i])

    temporal_conv_net = TCN(
        nb_filters=tcn_num_filters,
        kernel_size=7,
        dilations=[1, 2, 4, 8, 16, 32]
    )

    print(temporal_conv_net.receptive_field)

    model = Sequential()
    model.add(Embedding(max_features, 128, input_shape=(max_len,)))
    model.add(temporal_conv_net)
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    tcn_layer_outputs = list(temporal_conv_net.layers_outputs)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=4,
              validation_data=[x_test, y_test])

    if os.path.exists('acts'):
        shutil.rmtree('acts')
    os.makedirs('acts')

    print(model.predict_on_batch(x_val))
    print(y_val)

    visualize(model, x_test[0:10], max_len, tcn_num_filters, tcn_layer_outputs, 'x_test')
    visualize(model, x_val, max_len, tcn_num_filters, tcn_layer_outputs, 'x_val')


if __name__ == '__main__':
    main()
