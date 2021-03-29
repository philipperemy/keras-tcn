import unittest

import numpy as np
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential

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

    def test_compute_output_for_multiple_config(self):
        # with time steps None.
        o1 = TCN(nb_filters=NB_FILTERS, return_sequences=True, padding='same').compute_output_shape((None, None, 4))
        self.assertListEqual(list(o1), [None, None, NB_FILTERS])

        o2 = TCN(nb_filters=NB_FILTERS, return_sequences=True, padding='causal').compute_output_shape((None, None, 4))
        self.assertListEqual(list(o2), [None, None, NB_FILTERS])

        o3 = TCN(nb_filters=NB_FILTERS, return_sequences=False, padding='same').compute_output_shape((None, None, 4))
        self.assertListEqual(list(o3), [None, NB_FILTERS])

        o4 = TCN(nb_filters=NB_FILTERS, return_sequences=False, padding='causal').compute_output_shape((None, None, 4))
        self.assertListEqual(list(o4), [None, NB_FILTERS])

        # with time steps known.
        o5 = TCN(nb_filters=NB_FILTERS, return_sequences=True, padding='same').compute_output_shape((None, 5, 4))
        self.assertListEqual(list(o5), [None, 5, NB_FILTERS])

        o6 = TCN(nb_filters=NB_FILTERS, return_sequences=True, padding='causal').compute_output_shape((None, 5, 4))
        self.assertListEqual(list(o6), [None, 5, NB_FILTERS])

        o7 = TCN(nb_filters=NB_FILTERS, return_sequences=False, padding='same').compute_output_shape((None, 5, 4))
        self.assertListEqual(list(o7), [None, NB_FILTERS])

        o8 = TCN(nb_filters=NB_FILTERS, return_sequences=False, padding='causal').compute_output_shape((None, 5, 4))
        self.assertListEqual(list(o8), [None, NB_FILTERS])

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

    def test_norms(self):
        Sequential(layers=[TCN(input_shape=(20, 2), use_weight_norm=True)]).compile(optimizer='adam', loss='mse')
        Sequential(layers=[TCN(input_shape=(20, 2), use_weight_norm=False)]).compile(optimizer='adam', loss='mse')
        Sequential(layers=[TCN(input_shape=(20, 2), use_layer_norm=True)]).compile(optimizer='adam', loss='mse')
        Sequential(layers=[TCN(input_shape=(20, 2), use_layer_norm=False)]).compile(optimizer='adam', loss='mse')
        Sequential(layers=[TCN(input_shape=(20, 2), use_batch_norm=True)]).compile(optimizer='adam', loss='mse')
        Sequential(layers=[TCN(input_shape=(20, 2), use_batch_norm=False)]).compile(optimizer='adam', loss='mse')
        try:
            Sequential(layers=[TCN(input_shape=(20, 2), use_batch_norm=True, use_weight_norm=True)]).compile(
                optimizer='adam', loss='mse')
            raise AssertionError('test failed.')
        except ValueError:
            pass
        try:
            Sequential(layers=[TCN(input_shape=(20, 2), use_batch_norm=True,
                                   use_weight_norm=True, use_layer_norm=True)]).compile(
                optimizer='adam', loss='mse')
            raise AssertionError('test failed.')
        except ValueError:
            pass

    def test_receptive_field(self):
        self.assertEqual(37, TCN(kernel_size=3, dilations=(1, 3, 5), nb_stacks=1).receptive_field)
        self.assertEqual(379, TCN(kernel_size=4, dilations=(1, 2, 4, 8, 16, 32), nb_stacks=1).receptive_field)
        self.assertEqual(253, TCN(kernel_size=3, dilations=(1, 2, 4, 8, 16, 32), nb_stacks=1).receptive_field)
        self.assertEqual(125, TCN(kernel_size=3, dilations=(1, 2, 4, 8, 16), nb_stacks=1).receptive_field)
        self.assertEqual(61, TCN(kernel_size=3, dilations=(1, 2, 4, 8), nb_stacks=1).receptive_field)
        self.assertEqual(29, TCN(kernel_size=3, dilations=(1, 2, 4), nb_stacks=1).receptive_field)
        self.assertEqual(57, TCN(kernel_size=3, dilations=(1, 2, 4), nb_stacks=2).receptive_field)
        self.assertEqual(121, TCN(kernel_size=3, dilations=(1, 2, 4, 8), nb_stacks=2).receptive_field)
        self.assertEqual(91, TCN(kernel_size=4, dilations=(1, 2, 4, 8), nb_stacks=1).receptive_field)
        self.assertEqual(25, TCN(kernel_size=5, dilations=(1, 2), nb_stacks=1).receptive_field)
        self.assertEqual(31, TCN(kernel_size=6, dilations=(1, 2), nb_stacks=1).receptive_field)
        # 1+(3-1)*1*(1+3+5)*2 = 37
        # 1+(4-1)*1*(1+2+4+8+16+32)*2 = 379
        # 1+(3-1)*1*(1+2+4+8+16+32)*2 = 253
        # 1+(3-1)*1*(1+2+4+8+16)*2 = 125
        # 1+(3-1)*1*(1+2+4+8)*2 = 61
        # 1+(3-1)*1*(1+2+4)*2 = 29
        # 1+(3-1)*2*(1+2+4)*2 = 57
        # 1+(3-1)*2*(1+2+4+8)*2 = 121
        # 1+(4-1)*1*(1+2+4+8)*2 = 91
        # 1+(5-1)*1*(1+2)*2 = 25
        # 1+(6-1)*1*(1+2)*2 = 31


if __name__ == '__main__':
    unittest.main()
