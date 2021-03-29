import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from tcn import TCN


# if time_steps > tcn_layer.receptive_field, then we should not
# be able to solve this task.


def get_x_y(time_steps, size=1000):
    pos_indices = np.random.choice(size, size=int(size // 2), replace=False)
    x_train = np.zeros(shape=(size, time_steps, 1))
    y_train = np.zeros(shape=(size, 1))
    x_train[pos_indices, 0] = 1.0  # we introduce the target in the first timestep of the sequence.
    y_train[pos_indices, 0] = 1.0  # the task is to see if the TCN can go back in time to find it.
    return x_train, y_train


def new_bounds(dilations, bounds, input_dim, kernel_size, nb_stacks):
    # similar to the bisect algorithm.
    middle = int(np.mean(bounds))
    t1 = could_task_be_learned(dilations, bounds[0], input_dim, kernel_size, nb_stacks)
    t_middle = could_task_be_learned(dilations, middle, input_dim, kernel_size, nb_stacks)
    t2 = could_task_be_learned(dilations, bounds[1], input_dim, kernel_size, nb_stacks)
    go_left = t1 and not t_middle
    go_right = t_middle and not t2
    if go_left:
        assert not go_right
    if go_right:
        assert not go_left
    assert go_left or go_right

    if go_left:
        return np.array([bounds[0], middle])
    else:
        return np.array([middle, bounds[1]])


def est_receptive_field(kernel_size, nb_stacks, dilations):
    print('K', 'S', 'D', kernel_size, nb_stacks, dilations)
    input_dim = 1
    bounds = np.array([5, 800])
    while True:
        bounds = new_bounds(dilations, bounds, input_dim, kernel_size, nb_stacks)
        if bounds[1] - bounds[0] <= 1:
            print(f'Receptive field: {bounds[0]}.')
            break


def could_task_be_learned(dilations, guess, input_dim, kernel_size, nb_stacks):
    tcn_layer = TCN(
        kernel_size=kernel_size,
        dilations=dilations,
        nb_stacks=nb_stacks,
        input_shape=(guess, input_dim)
    )

    m = Sequential([
        tcn_layer,
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    x, y = get_x_y(guess)
    m.fit(x, y, validation_split=0.2, verbose=0, epochs=2)
    accuracy = m.evaluate(x, y, verbose=0)[1]
    task_is_learned = accuracy > 0.95
    return task_is_learned


if __name__ == '__main__':
    est_receptive_field(kernel_size=2, nb_stacks=1, dilations=(1, 2, 4))
