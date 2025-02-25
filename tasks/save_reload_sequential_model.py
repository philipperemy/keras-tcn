import numpy as np
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.models import Sequential, model_from_json, load_model

from tcn import TCN, tcn_full_summary

# define input shape
max_len = 100
max_features = 50

# make model
model = Sequential(layers=[Embedding(max_features, 16, input_shape=(max_len,)),
                           TCN(nb_filters=12,
                               dropout_rate=0.5,
                               kernel_size=6,
                               use_batch_norm=True,
                               dilations=[1, 2, 4]),
                           Dense(units=1, activation='sigmoid')])

model.compile(loss='mae')
model.fit(x=np.random.random((max_features, 100)), y=np.random.random((max_features, 1)))

# get model as json string and save to file
model_as_json = model.to_json()
with open('model.json', "w") as json_file:
    json_file.write(model_as_json)
# save weights to file (for this format, need h5py installed)
model.save_weights('model.weights.h5')

# Make inference.
inputs = np.ones(shape=(1, 100))
out1 = model.predict(inputs)[0, 0]
print('*' * 80)
print('Inference after creation:', out1)

# load model from file
loaded_json = open('model.json', 'r').read()
reloaded_model = model_from_json(loaded_json, custom_objects={'TCN': TCN})

tcn_full_summary(model, expand_residual_blocks=False)

# restore weights
reloaded_model.load_weights('model.weights.h5')

# Make inference.
out2 = reloaded_model.predict(inputs)[0, 0]
print('*' * 80)
print('Inference after loading:', out2)

assert abs(out1 - out2) < 1e-6

model.save('model.keras')
out11 = load_model('model.keras').predict(inputs)[0, 0]
out22 = model.predict(inputs)[0, 0]
assert abs(out11 - out22) < 1e-6
