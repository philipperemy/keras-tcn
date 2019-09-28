import numpy as np
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Embedding
from tcn.tcn import TCN

# define input shape
max_len = 100
max_features = 50

# make model
model = Sequential(layers=[Embedding(max_features, 16, input_shape=(max_len,)),
                           TCN(nb_filters=12,
                               dropout_rate=0.5,
                               kernel_size=6,
                               dilations=[1, 2, 4]),
                           Dropout(0.5),
                           Dense(units=1, activation='sigmoid')])

# get model as json string and save to file
model_as_json = model.to_json()
with open(r'model.json', "w") as json_file:
    json_file.write(model_as_json)
# save weights to file (for this format, need h5py installed)
model.save_weights('weights.h5')

# Make inference.
inputs = np.ones(shape=(1, 100))
out1 = model.predict(inputs)[0, 0]
print('*' * 80)
print('Inference after creation:', out1)

# load model from file
loaded_json = open(r'model.json', 'r').read()
reloaded_model = model_from_json(loaded_json, custom_objects={'TCN': TCN})

# restore weights
reloaded_model.load_weights(r'weights.h5')

# Make inference.
out2 = reloaded_model.predict(inputs)[0, 0]
print('*' * 80)
print('Inference after loading:', out2)
