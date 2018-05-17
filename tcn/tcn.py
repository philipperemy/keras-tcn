import keras.backend as K
from keras import optimizers
from keras.layers import Conv1D, SpatialDropout1D
from keras.layers import Activation, Lambda
from keras.layers import Convolution1D, Dense
from keras.models import Input, Model
import keras.layers


def channel_normalization(x):
    # Normalize by the highest activation
    max_values = K.max(K.abs(x), 2, keepdims=True) + 1e-5
    out = x / max_values
    return out


def wave_net_activation(x):
    tanh_out = Activation('tanh')(x)
    sigm_out = Activation('sigmoid')(x)
    return keras.layers.multiply([tanh_out, sigm_out])


def residual_block(x, s, i, activation, nb_filters, kernel_size):
    original_x = x
    conv = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=2 ** i, padding='causal',
                  name='dilated_conv_%d_tanh_s%d' % (2 ** i, s))(x)
    if activation == 'norm_relu':
        x = Activation('relu')(conv)
        x = Lambda(channel_normalization)(x)
    elif activation == 'wavenet':
        x = wave_net_activation(conv)
    else:
        x = Activation(activation)(conv)

    x = SpatialDropout1D(0.05)(x)

    # 1x1 conv.
    x = Convolution1D(nb_filters, 1, padding='same')(x)
    res_x = keras.layers.add([original_x, x])
    return res_x, x


def dilated_tcn(num_feat, num_classes, nb_filters,
                kernel_size, dilatations, nb_stacks, max_len,
                activation='wavenet', use_skip_connections=True,
                return_param_str=False, output_slice_index=None,
                regression=False):
    """
    dilation_depth : number of layers per stack
    nb_stacks : number of stacks.
    """
    input_layer = Input(name='input_layer', shape=(max_len, num_feat))
    x = input_layer
    x = Convolution1D(nb_filters, kernel_size, padding='causal', name='initial_conv')(x)

    skip_connections = []
    for s in range(nb_stacks):
        for i in dilatations:
            x, skip_out = residual_block(x, s, i, activation, nb_filters, kernel_size)
            skip_connections.append(skip_out)

    if use_skip_connections:
        x = keras.layers.add(skip_connections)
    x = Activation('relu')(x)

    if output_slice_index is not None:  # can test with 0 or -1.
        if output_slice_index == 'last':
            output_slice_index = -1
        if output_slice_index == 'first':
            output_slice_index = 0
        x = Lambda(lambda tt: tt[:, output_slice_index, :])(x)

    print('x.shape=', x.shape)

    if not regression:
        # classification
        x = Dense(num_classes)(x)
        x = Activation('softmax', name='output_softmax')(x)
        output_layer = x
        print(f'model.x = {input_layer.shape}')
        print(f'model.y = {output_layer.shape}')
        model = Model(input_layer, output_layer)

        adam = optimizers.Adam(lr=0.002, clipnorm=1.)
        model.compile(adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print('Adam with norm clipping.')
    else:
        # regression
        x = Dense(1)(x)
        x = Activation('linear', name='output_dense')(x)
        output_layer = x
        print(f'model.x = {input_layer.shape}')
        print(f'model.y = {output_layer.shape}')
        model = Model(input_layer, output_layer)
        adam = optimizers.Adam(lr=0.002, clipnorm=1.)
        model.compile(adam, loss='mean_squared_error')

    if return_param_str:
        param_str = 'D-TCN_C{}_B{}_L{}'.format(2, nb_stacks, dilatations)
        return model, param_str
    else:
        return model
