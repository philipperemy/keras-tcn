import keras.backend as K
from keras import optimizers
from keras.layers import Layer
from keras.layers import Activation, Lambda, add
from keras.layers import Conv1D, SpatialDropout1D
from keras.layers import Dense, BatchNormalization
from keras.models import Input, Model

import tcn.hollow_class as hollow

TCN, compiled_tcn = hollow.hollow_class.__init__(K,
                                                 optimizers,
                                                 Layer,
                                                 Activation,
                                                 Lambda,
                                                 add,
                                                 Conv1D,
                                                 SpatialDropout1D,
                                                 Dense,
                                                 BatchNormalization,
                                                 Input,
                                                 Model)

del hollow
del K
del optimizers
del Layer
del Activation
del Lambda
del add
del Conv1D
del SpatialDropout1D
del Dense
del BatchNormalization
del Input
del Model