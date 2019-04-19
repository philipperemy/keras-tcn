# https://datamarket.com/data/set/22ox/monthly-milk-production-pounds-per-cow-jan-62-dec-75#!ds=22ox&display=line
import numpy as np
import pandas as pd
from keras import Input, Model
from keras.layers import Dense

from tcn import TCN

milk = pd.read_csv('monthly-milk-production-pounds-p.csv', index_col=0, parse_dates=True)

print(milk.head())

lookback_window = 12  # months.

milk = milk.values  # just keep np array here for simplicity.

x, y = [], []
for i in range(lookback_window, len(milk)):
    x.append(milk[i - lookback_window:i])
    y.append(milk[i])
x = np.array(x)
y = np.array(y)

print(x.shape)
print(y.shape)

i = Input(shape=(lookback_window, 1))
m = TCN()(i)
m = Dense(1, activation='linear')(m)

model = Model(inputs=[i], outputs=[m])

model.summary()

# try using different optimizers and different optimizer configs
model.compile('adam', 'mae')

print('Train...')
model.fit(x, y, epochs=1000, verbose=2)
