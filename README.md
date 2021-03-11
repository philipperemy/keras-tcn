# Keras TCN

*Keras Temporal Convolutional Network*. [[paper](https://arxiv.org/abs/1803.01271)]

[![Downloads](https://pepy.tech/badge/keras-tcn)](https://pepy.tech/project/keras-tcn)
[![Downloads](https://pepy.tech/badge/keras-tcn/month)](https://pepy.tech/project/keras-tcn)
![Keras TCN CI](https://github.com/philipperemy/keras-tcn/workflows/Keras%20TCN%20CI/badge.svg?branch=master)
```bash
pip install keras-tcn
pip install keras-tcn --no-dependencies  # without the dependencies if you already have TF/Numpy.
```

## Why Temporal Convolutional Network instead of LSTM/GRU?

- TCNs exhibit longer memory than recurrent architectures with the same capacity.
- Constantly performs better than LSTM/GRU architectures on a vast range of tasks (Seq. MNIST, Adding Problem, Copy Memory, Word-level PTB...).
- Parallelism, flexible receptive field size, stable gradients, low memory requirements for training, variable length inputs...

<p align="center">
  <img src="misc/Dilated_Conv.png">
  <b>Visualization of a stack of dilated causal convolutional layers (Wavenet, 2016)</b><br><br>
</p>


## Index

   * [Keras TCN](#keras-tcn)
      * [API](#api)
         * [Arguments](#arguments)
         * [Input shape](#input-shape)
         * [Output shape](#output-shape)
         * [Receptive field](#receptive-field)
         * [Non-causal TCN](#non-causal-tcn)
      * [Installation from the sources](#installation-from-the-sources)
      * [Run](#run)
      * [Reproducible results](#reproducible-results)
      * [Tasks](#tasks)
         * [Adding Task](#adding-task)
            * [Explanation](#explanation)
            * [Implementation results](#implementation-results)
         * [Copy Memory Task](#copy-memory-task)
            * [Explanation](#explanation-1)
            * [Implementation results (first epochs)](#implementation-results-first-epochs)
         * [Sequential MNIST](#sequential-mnist)
            * [Explanation](#explanation-2)
            * [Implementation results](#implementation-results-1)
      * [Testing](#testing)
      * [References](#references)
      * [Related](#related)


## API

The usual way is to import the TCN layer and use it inside a Keras model. An example is provided below for a regression task (cf. [tasks](tasks) for other examples):

```python
from tcn import TCN, tcn_full_summary
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# if time_steps > tcn_layer.receptive_field, then we should not
# be able to solve this task.
batch_size, time_steps, input_dim = None, 20, 1


def get_x_y(size=1000):
    import numpy as np
    pos_indices = np.random.choice(size, size=int(size // 2), replace=False)
    x_train = np.zeros(shape=(size, time_steps, 1))
    y_train = np.zeros(shape=(size, 1))
    x_train[pos_indices, 0] = 1.0  # we introduce the target in the first timestep of the sequence.
    y_train[pos_indices, 0] = 1.0  # the task is to see if the TCN can go back in time to find it.
    return x_train, y_train


tcn_layer = TCN(input_shape=(time_steps, input_dim))
print('Receptive field size =', tcn_layer.receptive_field)

m = Sequential([
    tcn_layer,
    Dense(1)
])

m.compile(optimizer='adam', loss='mse')

tcn_full_summary(m, expand_residual_blocks=False)

x, y = get_x_y()
m.fit(x, y, epochs=10, validation_split=0.2)
```

A ready-to-use TCN model can be used that way (cf. `tasks/` for the full code):

```python
from tcn import compiled_tcn

model = compiled_tcn(...)
model.fit(x, y) # Keras model.
```

### Arguments

```python
TCN(
    nb_filters=64,
    kernel_size=3,
    nb_stacks=1,
    dilations=(1, 2, 4, 8, 16, 32),
    padding='causal',
    use_skip_connections=True,
    dropout_rate=0.0,
    return_sequences=False,
    activation='relu',
    kernel_initializer='he_normal',
    use_batch_norm=False,
    use_layer_norm=True,
    use_weight_norm=False,
    **kwargs
)
```

- `nb_filters`: Integer. The number of filters to use in the convolutional layers. Would be similar to `units` for LSTM. Can be a list.
- `kernel_size`: Integer. The size of the kernel to use in each convolutional layer.
- `dilations`: List/Tuple. A dilation list. Example is: [1, 2, 4, 8, 16, 32, 64].
- `nb_stacks`: Integer. The number of stacks of residual blocks to use.
- `padding`: String. The padding to use in the convolutions. 'causal' for a causal network (as in the original implementation) and 'same' for a non-causal network.
- `use_skip_connections`: Boolean. If we want to add skip connections from input to each residual block.
- `return_sequences`: Boolean. Whether to return the last output in the output sequence, or the full sequence.
- `dropout_rate`: Float between 0 and 1. Fraction of the input units to drop.
- `activation`: The activation used in the residual blocks o = activation(x + F(x)).
- `kernel_initializer`: Initializer for the kernel weights matrix (Conv1D).
- `use_batch_norm`: Whether to use batch normalization in the residual layers or not.
- `use_layer_norm`: Whether to use layer normalization in the residual layers or not.
- `use_weight_norm`: Whether to use weight normalization in the residual layers or not.
- `kwargs`: Any other set of arguments for configuring the parent class Layer. For example "name=str", Name of the model. Use unique names when using multiple TCN.

### Input shape

3D tensor with shape `(batch_size, timesteps, input_dim)`.

`timesteps` can be None. This can be useful if each sequence is of a different length: [Multiple Length Sequence Example](tasks/multi_length_sequences.py).

### Output shape

- if `return_sequences=True`: 3D tensor with shape `(batch_size, timesteps, nb_filters)`.
- if `return_sequences=False`: 2D tensor with shape `(batch_size, nb_filters)`.

### Receptive field

The receptive field can be calculated using the following formula:

<p align="center">
  <img src="https://user-images.githubusercontent.com/12395799/106308730-6d4c5180-6261-11eb-82e9-a12a1958058d.png">
</p>

where *N<sub>s</sub>* is the number of stacks, *N<sub>b</sub>* is the number of residual blocks per stack, **d** is a vector containing the dilations of each residual block in one stack, and **k** is a vector containing the lengths of the filters of each residual block in one stack.

- If a TCN has only one stack of residual blocks with a kernel size of 2 and dilations [1, 2, 4, 8], its receptive field is 1 + 1 * (1 * 1 + 2 * 1 + 4 * 1 + 8 * 1) = 16. The image below illustrates it:

<p align="center">
  <img src="https://user-images.githubusercontent.com/40159126/41830054-10e56fda-7871-11e8-8591-4fa46680c17f.png">
  <b>ks = 2, dilations = [1, 2, 4, 8], 1 block</b><br><br>
</p>

- If the TCN has now 2 stacks of residual blocks, you would get the situation below, that is, an increase in the receptive field up to 1 + 2 * (1 * 1 + 2 * 1 + 4 * 1 + 8 * 1) = 31:

<p align="center">
  <img src="https://user-images.githubusercontent.com/40159126/41830618-a8f82a8a-7874-11e8-9d4f-2ebb70a31465.jpg">
  <b>ks = 2, dilations = [1, 2, 4, 8], 2 blocks</b><br><br>
</p>


- If we increased the number of stacks to 3, the size of the receptive field would increase again, such as below:

<p align="center">
  <img src="https://user-images.githubusercontent.com/40159126/41830628-ae6e73d4-7874-11e8-8ecd-cea37efa33f1.jpg">
  <b>ks = 2, dilations = [1, 2, 4, 8], 3 blocks</b><br><br>
</p>


### Non-causal TCN

Making the TCN architecture non-causal allows it to take the future into consideration to do its prediction as shown in the figure below.

However, it is not anymore suitable for real-time applications.

<p align="center">
  <img src="misc/Non_Causal.png">
  <b>Non-Causal TCN - ks = 3, dilations = [1, 2, 4, 8], 1 block</b><br><br>
</p>

To use a non-causal TCN, specify `padding='valid'` or `padding='same'` when initializing the TCN layers.

## Installation from the sources

```bash
git clone git@github.com:philipperemy/keras-tcn.git && cd keras-tcn
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
pip install .
```

## Run

Once `keras-tcn` is installed as a package, you can take a glimpse of what is possible to do with TCNs. Some tasks examples are available in the repository for this purpose:

```bash
cd adding_problem/
python main.py # run adding problem task

cd copy_memory/
python main.py # run copy memory task

cd mnist_pixel/
python main.py # run sequential mnist pixel task
```

## Reproducible results

Reproducible results are possible on (NVIDIA) GPUs using the [tensorflow-determinism](https://github.com/NVIDIA/tensorflow-determinism) library. It was tested with keras-tcn by @lingdoc and he got reproducible results.

## Tasks

### Word PTB

Language modeling remains one of the primary applications of recurrent networks. In this example, we show that TCN can beat LSTM without too much tuning.

<p align="center">
  <img src="tasks/word_ptb/result.png" width="800"><br>
  <i>TCN vs LSTM (comparable number of weights)</i><br><br>
</p>

### Adding Task

The task consists of feeding a large array of decimal numbers to the network, along with a boolean array of the same length. The objective is to sum the two decimals where the boolean array contain the two 1s.

#### Explanation

<p align="center">
  <img src="misc/Adding_Task.png">
  <b>Adding Problem Task</b><br><br>
</p>

#### Implementation results

```
782/782 [==============================] - 154s 197ms/step - loss: 0.8437 - val_loss: 0.1883
782/782 [==============================] - 154s 196ms/step - loss: 0.0702 - val_loss: 0.0111
782/782 [==============================] - 153s 195ms/step - loss: 0.0053 - val_loss: 0.0038
782/782 [==============================] - 154s 196ms/step - loss: 0.0035 - val_loss: 0.0027
782/782 [==============================] - 153s 196ms/step - loss: 0.0030 - val_loss: 0.0065
782/782 [==============================] - 151s 193ms/step - loss: 0.0027 - val_loss: 0.0018
782/782 [==============================] - 152s 194ms/step - loss: 0.0025 - val_loss: 0.0036
782/782 [==============================] - 153s 196ms/step - loss: 0.0024 - val_loss: 0.0018
782/782 [==============================] - 152s 194ms/step - loss: 0.0023 - val_loss: 0.0016
782/782 [==============================] - 152s 194ms/step - loss: 0.0014 - val_loss: 3.7456e-04
782/782 [==============================] - 153s 196ms/step - loss: 9.4740e-04 - val_loss: 7.0205e-04
782/782 [==============================] - 152s 194ms/step - loss: 6.9630e-04 - val_loss: 3.7180e-04
```

### Copy Memory Task

The copy memory consists of a very large array:
- At the beginning, there's the vector x of length N. This is the vector to copy.
- At the end, N+1 9s are present. The first 9 is seen as a delimiter.
- In the middle, only 0s are there.

The idea is to copy the content of the vector x to the end of the large array. The task is made sufficiently complex by increasing the number of 0s in the middle.

#### Explanation

<p align="center">
  <img src="misc/Copy_Memory_Task.png">
  <b>Copy Memory Task</b><br><br>
</p>

#### Implementation results (first epochs)

```
118/118 [==============================] - 17s 143ms/step - loss: 1.1732 - accuracy: 0.6725 - val_loss: 0.1119 - val_accuracy: 0.9796
118/118 [==============================] - 15s 125ms/step - loss: 0.0645 - accuracy: 0.9831 - val_loss: 0.0402 - val_accuracy: 0.9853
118/118 [==============================] - 15s 125ms/step - loss: 0.0393 - accuracy: 0.9856 - val_loss: 0.0372 - val_accuracy: 0.9857
118/118 [==============================] - 15s 125ms/step - loss: 0.0361 - accuracy: 0.9858 - val_loss: 0.0344 - val_accuracy: 0.9860
118/118 [==============================] - 15s 125ms/step - loss: 0.0345 - accuracy: 0.9860 - val_loss: 0.0335 - val_accuracy: 0.9864
118/118 [==============================] - 15s 125ms/step - loss: 0.0325 - accuracy: 0.9867 - val_loss: 0.0268 - val_accuracy: 0.9886
118/118 [==============================] - 15s 125ms/step - loss: 0.0268 - accuracy: 0.9885 - val_loss: 0.0206 - val_accuracy: 0.9908
118/118 [==============================] - 15s 125ms/step - loss: 0.0228 - accuracy: 0.9900 - val_loss: 0.0169 - val_accuracy: 0.9933
```

### Sequential MNIST

#### Explanation

The idea here is to consider MNIST images as 1-D sequences and feed them to the network. This task is particularly hard because sequences are 28*28 = 784 elements. In order to classify correctly, the network has to remember all the sequence. Usual LSTM are unable to perform well on this task.

<p align="center">
  <img src="misc/Sequential_MNIST_Task.png">
  <b>Sequential MNIST</b><br><br>
</p>

#### Implementation results

```
1875/1875 [==============================] - 46s 25ms/step - loss: 0.0949 - accuracy: 0.9706 - val_loss: 0.0763 - val_accuracy: 0.9756
1875/1875 [==============================] - 46s 25ms/step - loss: 0.0831 - accuracy: 0.9743 - val_loss: 0.0656 - val_accuracy: 0.9807
1875/1875 [==============================] - 46s 25ms/step - loss: 0.0752 - accuracy: 0.9763 - val_loss: 0.0604 - val_accuracy: 0.9802
1875/1875 [==============================] - 46s 25ms/step - loss: 0.0685 - accuracy: 0.9785 - val_loss: 0.0588 - val_accuracy: 0.9813
1875/1875 [==============================] - 46s 25ms/step - loss: 0.0624 - accuracy: 0.9801 - val_loss: 0.0545 - val_accuracy: 0.9822
1875/1875 [==============================] - 46s 25ms/step - loss: 0.0603 - accuracy: 0.9812 - val_loss: 0.0478 - val_accuracy: 0.9835
1875/1875 [==============================] - 46s 25ms/step - loss: 0.0566 - accuracy: 0.9821 - val_loss: 0.0546 - val_accuracy: 0.9826
1875/1875 [==============================] - 46s 25ms/step - loss: 0.0503 - accuracy: 0.9843 - val_loss: 0.0441 - val_accuracy: 0.9853
1875/1875 [==============================] - 46s 25ms/step - loss: 0.0486 - accuracy: 0.9840 - val_loss: 0.0572 - val_accuracy: 0.9832
1875/1875 [==============================] - 46s 25ms/step - loss: 0.0453 - accuracy: 0.9858 - val_loss: 0.0424 - val_accuracy: 0.9862
```

## Testing

Testing is based on Tox.

```
pip install tox
tox
```

## References
- https://github.com/locuslab/TCN/ (TCN for Pytorch)
- https://arxiv.org/pdf/1803.01271 (An Empirical Evaluation of Generic Convolutional and Recurrent Networks
for Sequence Modeling)
- https://arxiv.org/pdf/1609.03499 (Original Wavenet paper)

## Related
- https://github.com/Baichenjia/Tensorflow-TCN (Tensorflow Eager implementation of TCNs)

## Citation

```
@misc{KerasTCN,
  author = {Philippe Remy},
  title = {Temporal Convolutional Networks for Keras},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/philipperemy/keras-tcn}},
}
```

Special thanks to:
- @alextheseal
- @qlemaire22
