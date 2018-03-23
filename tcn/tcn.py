import keras.backend as K
from keras.layers import ZeroPadding1D, AtrousConvolution1D, Cropping1D, SpatialDropout1D, Activation, Lambda, \
    Convolution1D, Merge
from keras.models import Input, Model


def channel_normalization(x):
    # Normalize by the highest activation
    max_values = K.max(K.abs(x), 2, keepdims=True) + 1e-5
    out = x / max_values
    return out


def wave_net_activation(x):
    tanh_out = Activation('tanh')(x)
    sigm_out = Activation('sigmoid')(x)
    return Merge(mode='mul')([tanh_out, sigm_out])


def residual_block(x, s, i, activation, causal, nb_filters):
    original_x = x

    if causal:
        x = ZeroPadding1D(((2 ** i) // 2, 0))(x)
        conv = AtrousConvolution1D(nb_filters, kernel_size=6, atrous_rate=2 ** i, padding='same',
                                   name='dilated_conv_%d_tanh_s%d' % (2 ** i, s))(x)
        conv = Cropping1D((0, (2 ** i) // 2))(conv)
    else:
        conv = AtrousConvolution1D(nb_filters, kernel_size=6, atrous_rate=2 ** i, padding='same',
                                   name='dilated_conv_%d_tanh_s%d' % (2 ** i, s))(x)

    conv = SpatialDropout1D(0.3)(conv)
    # x = WaveNet_activation(conv)

    if activation == 'norm_relu':
        x = Activation('relu')(conv)
        x = Lambda(channel_normalization)(x)
    elif activation == 'wavenet':
        x = wave_net_activation(conv)
    else:
        x = Activation(activation)(conv)

        # res_x  = Convolution1D(nb_filters, 1, padding='same')(x)
    # skip_x = Convolution1D(nb_filters, 1, padding='same')(x)
    x = Convolution1D(nb_filters, 1, padding='same')(x)

    res_x = Merge(mode='sum')([original_x, x])

    # return res_x, skip_x
    return res_x, x


def dilated_tcn(num_feat, num_classes, nb_filters,
                dilation_depth, nb_stacks, max_len,
                activation='wavenet', tail_conv=1, use_skip_connections=True, causal=False,
                optimizer='adam', return_param_str=False):
    """
    dilation_depth : number of layers per stack
    nb_stacks : number of stacks.
    """

    input_layer = Input(name='input_layer', shape=(max_len, num_feat))

    skip_connections = []
    x = input_layer
    if causal:
        x = ZeroPadding1D((1, 0))(x)
        x = Convolution1D(nb_filters, 2, padding='same', name='initial_conv')(x)
        x = Cropping1D((0, 1))(x)
    else:
        x = Convolution1D(nb_filters, 3, padding='same', name='initial_conv')(x)

    for s in range(nb_stacks):
        for i in range(0, dilation_depth + 1):
            x, skip_out = residual_block(x, s, i, activation, causal, nb_filters)
            skip_connections.append(skip_out)

    if use_skip_connections:
        x = Merge(mode='sum')(skip_connections)
    x = Activation('relu')(x)
    x = Convolution1D(nb_filters, tail_conv, padding='same')(x)
    x = Activation('relu')(x)
    x = Convolution1D(num_classes, tail_conv, padding='same')(x)
    x = Activation('softmax', name='output_softmax')(x)
    output_layer = x

    print(f'model.x = {input_layer.shape}')
    print(f'model.y = {output_layer.shape}')
    model = Model(input_layer, output_layer)
    model.compile(optimizer, loss='sparse_categorical_crossentropy',
                  sample_weight_mode='temporal', metrics=['accuracy'])

    if return_param_str:
        param_str = 'D-TCN_C{}_B{}_L{}'.format(2, nb_stacks, dilation_depth)
        if causal:
            param_str += '_causal'

        return model, param_str
    else:
        return model
