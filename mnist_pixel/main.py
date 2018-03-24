import keras.backend as K

from utils import data_generator
from tcn import tcn


def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations
    # np.sum(activations[15].squeeze(), axis=1)


def run_task():
    (x_train, y_train), (x_test, y_test) = data_generator()

    model, param_str = tcn.dilated_tcn(output_slice_index='last', # try 'first'.
                                       num_feat=1,
                                       num_classes=10,
                                       nb_filters=64,
                                       kernel_size=8,
                                       dilatations=[1, 2, 4, 8],
                                       nb_stacks=8,
                                       max_len=x_train[0:1].shape[1],
                                       activation='norm_relu',
                                       use_skip_connections=False,
                                       return_param_str=True)

    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')
    print(f'x_test.shape = {x_test.shape}')
    print(f'y_test.shape = {y_test.shape}')

    model.summary()

    # a = np.zeros_like(x_train[0:1])
    # a[:, 0, :] = 1.0
    # print(get_activations(model, a))

    model.fit(x_train, y_train.squeeze().argmax(axis=1), epochs=100,
              validation_data=(x_test, y_test.squeeze().argmax(axis=1)))


if __name__ == '__main__':
    run_task()
