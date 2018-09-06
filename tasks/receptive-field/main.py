from tcn import tcn
from utils import data_generator


def run_task(sequence_length=8):
    x_train, y_train = data_generator(batch_size=2048, sequence_length=sequence_length)
    print(x_train.shape)
    print(y_train.shape)
    model, param_str = tcn.dilated_tcn(output_slice_index='last',
                                       num_feat=1,
                                       num_classes=10,
                                       nb_filters=10,
                                       kernel_size=10,
                                       dilations=[1, 2, 4, 8, 16, 32],
                                       nb_stacks=6,
                                       max_len=x_train[0:1].shape[1],
                                       activation='norm_relu',
                                       use_skip_connections=False,
                                       return_param_str=True)

    print(param_str)

    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')

    # model.summary()

    model.fit(x_train, y_train, epochs=5)
    return model.evaluate(x_train, y_train)[1]


def main():
    ks = 3
    print('acc =', run_task(sequence_length=630))


if __name__ == '__main__':
    main()
