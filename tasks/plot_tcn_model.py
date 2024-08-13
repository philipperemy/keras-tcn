from tcn import TCN
import tensorflow as tf

timesteps = 32
input_dim = 5
input_shape = (timesteps, input_dim)
forecast_horizon = 3
num_features = 4

inputs = tf.keras.layers.Input(shape=input_shape, name='input')
tcn_out = TCN(nb_filters=64, kernel_size=3, nb_stacks=1, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(forecast_horizon * num_features, activation='linear')(tcn_out)
outputs = tf.keras.layers.Reshape((forecast_horizon, num_features), name='ouput')(outputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

tf.keras.utils.plot_model(
    model,
    to_file='TCN_model.png',
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    rankdir='TB',
    dpi=200,
    layer_range=None,
)
