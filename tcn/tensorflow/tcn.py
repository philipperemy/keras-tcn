import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Activation, Lambda, add
from tensorflow.keras.layers import Conv1D, SpatialDropout1D
from tensorflow.keras.layers import Convolution1D, Dense, BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras import Model

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