import importlib.util


spec = importlib.util.find_spec('tensorflow')
if spec is None:
    print('No tensorflow installation detected. tensorflow.TCN is not available.')
else:
    from tcn import tensorflow


spec = importlib.util.find_spec('keras')
if spec is None:
    print('No Keras installation detected. keras.TCN is not available.')
else:
    from tcn import keras


del importlib

__version__ = '2.8.3'
