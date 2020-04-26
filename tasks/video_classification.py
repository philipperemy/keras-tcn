import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import MaxPool2D

from tcn import TCN

num_samples = 1000  # number of videos.
num_frames = 240  # 10 seconds of video at 24 ips.
h, w, c = 32, 32, 3  # def not a HD video! 32x32 color.


def data():
    # very very dummy example. The purpose is more to show how to use a RNN/TCN
    # in the context of video processing.
    inputs = np.zeros(shape=(num_samples, num_frames, h, w, c))
    targets = np.zeros(shape=(num_samples, 1))
    # class 0 => only 0.

    # class 1 => will contain some 1s.
    for i in range(num_samples):
        if np.random.uniform(low=0, high=1) > 0.50:
            for j in range(num_frames):
                inputs[i, j] = (np.random.uniform(low=0, high=1) > 0.90)
            targets[i] = 1
    return inputs, targets


def train():
    # Good exercise: https://www.crcv.ucf.edu/data/UCF101.php
    # replace data() by this dataset.
    # Useful links:
    # - https://www.pyimagesearch.com/2019/07/15/video-classification-with-keras-and-deep-learning/
    # - https://github.com/sujiongming/UCF-101_video_classification
    x_train, y_train = data()

    inputs = Input(shape=(num_frames, h, w, c))
    # push num_frames in batch_dim to process all the frames independently of their orders (CNN features).
    x = Lambda(lambda y: K.reshape(y, (-1, h, w, c)))(inputs)
    # apply convolutions to each image of each video.
    x = Conv2D(16, 5)(x)
    x = MaxPool2D()(x)
    # re-creates the videos by reshaping.
    # 3D input shape (batch, timesteps, input_dim)
    num_features_cnn = np.prod(K.int_shape(x)[1:])
    x = Lambda(lambda y: K.reshape(y, (-1, num_frames, num_features_cnn)))(x)
    # apply the RNN on the time dimension (num_frames dim).
    x = TCN(16)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[inputs], outputs=[x])
    model.summary()
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    print('Train...')
    model.fit(x_train, y_train, validation_split=0.2, epochs=5)


if __name__ == '__main__':
    train()
