import unittest

import numpy as np
from tensorflow.keras import Input
from tensorflow.keras import Model

from tcn import TCN

NB_FILTERS = 16
TIME_STEPS = 20

SEQ_LEN_1 = 5
SEQ_LEN_2 = 1
SEQ_LEN_3 = 10


def predict_with_tcn(time_steps=None, padding='causal', return_sequences=True) -> list:
    input_dim = 4
    i = Input(batch_shape=(None, time_steps, input_dim))
    o = TCN(nb_filters=NB_FILTERS, return_sequences=return_sequences, padding=padding)(i)
    m = Model(inputs=[i], outputs=[o])
    m.compile(optimizer='adam', loss='mse')
    if time_steps is None:
        np.random.seed(123)
        return [
            m(np.random.rand(1, SEQ_LEN_1, input_dim)),
            m(np.random.rand(1, SEQ_LEN_2, input_dim)),
            m(np.random.rand(1, SEQ_LEN_3, input_dim))
        ]
    else:
        np.random.seed(123)
        return [m(np.random.rand(1, time_steps, input_dim))]


class TCNCallTest(unittest.TestCase):

    def test_causal_time_dim_known_return_sequences(self):
        r = predict_with_tcn(time_steps=TIME_STEPS, padding='causal', return_sequences=True)
        self.assertListEqual([list(b.shape) for b in r], [[1, TIME_STEPS, NB_FILTERS]])

    def test_causal_time_dim_unknown_return_sequences(self):
        r = predict_with_tcn(time_steps=None, padding='causal', return_sequences=True)
        self.assertListEqual([list(b.shape) for b in r],
                             [[1, SEQ_LEN_1, NB_FILTERS],
                              [1, SEQ_LEN_2, NB_FILTERS],
                              [1, SEQ_LEN_3, NB_FILTERS]])

    def test_non_causal_time_dim_known_return_sequences(self):
        r = predict_with_tcn(time_steps=TIME_STEPS, padding='same', return_sequences=True)
        self.assertListEqual([list(b.shape) for b in r], [[1, TIME_STEPS, NB_FILTERS]])

    def test_non_causal_time_dim_unknown_return_sequences(self):
        r = predict_with_tcn(time_steps=None, padding='same', return_sequences=True)
        self.assertListEqual([list(b.shape) for b in r],
                             [[1, SEQ_LEN_1, NB_FILTERS],
                              [1, SEQ_LEN_2, NB_FILTERS],
                              [1, SEQ_LEN_3, NB_FILTERS]])

    def test_causal_time_dim_known_return_no_sequences(self):
        r = predict_with_tcn(time_steps=TIME_STEPS, padding='causal', return_sequences=False)
        self.assertListEqual([list(b.shape) for b in r], [[1, NB_FILTERS]])

    def test_causal_time_dim_unknown_return_no_sequences(self):
        r = predict_with_tcn(time_steps=None, padding='causal', return_sequences=False)
        self.assertListEqual([list(b.shape) for b in r], [[1, NB_FILTERS], [1, NB_FILTERS], [1, NB_FILTERS]])

    def test_non_causal_time_dim_known_return_no_sequences(self):
        r = predict_with_tcn(time_steps=TIME_STEPS, padding='same', return_sequences=False)
        self.assertListEqual([list(b.shape) for b in r], [[1, NB_FILTERS]])

    def test_non_causal_time_dim_unknown_return_no_sequences(self):
        r = predict_with_tcn(time_steps=None, padding='same', return_sequences=False)
        self.assertListEqual([list(b.shape) for b in r], [[1, NB_FILTERS], [1, NB_FILTERS], [1, NB_FILTERS]])


if __name__ == '__main__':
    unittest.main()
