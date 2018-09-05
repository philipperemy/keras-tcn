# Keras TCN
*Keras Temporal Convolutional Network*

 * [Keras TCN](#keras-tcn)
    * [Why Temporal Convolutional Network?](#why-temporal-convolutional-network)
    * [API](#api)
       * [Regression (Many to one) e.g. adding problem](#--regression-many-to-one-eg-adding-problem)
       * [Classification (Many to one) e.g. copy memory task](#--classification-many-to-one-eg-copy-memory-task)
       * [Classification (Many to one) e.g. sequential mnist task](#--classification-many-to-one-eg-sequential-mnist-task)
    * [Installation](#installation)
    * [Run](#run)
    * [Tasks](#tasks)
    * [References](#references)

## Why Temporal Convolutional Network?

- TCNs exhibit longer memory than recurrent architectures with the same capacity.
- Constantly performs better than LSTM/GRU architectures on a vast range of tasks (Seq. MNIST, Adding Problem, Copy Memory, Word-level PTB...).
- Parallelism, flexible receptive field size, stable gradients, low memory requirements for training, variable length inputs...

<p align="center">
  <img src="misc/Dilated_Conv.png">
  <b>Visualization of a stack of dilated causal convolutional layers (Wavenet, 2016)</b><br><br>
</p>

## API

After installation, the model can be imported like this:

```
from tcn import tcn
```

In the following examples, we assume the input to have a shape `(batch_size, timesteps, input_dim)`.

The model is a Keras model. The model functions (`model.summary`, `model.fit`, `model.predict`...) are all functional.



### - Regression (Many to one) e.g. adding problem

```
model = tcn.dilated_tcn(output_slice_index='last',
                        num_feat=input_dim,
			num_classes=None,
                        nb_filters=24,
                        kernel_size=8,
                        dilations=[1, 2, 4, 8],
                        nb_stacks=8,
                        max_len=timesteps,
                        activation='norm_relu',
                        regression=True)
```

For a Many to Many regression, a cheap fix for now is to change the [number of units of the final Dense layer](https://github.com/philipperemy/keras-tcn/blob/8151b4a87f906fd856fd1c113c48392d542d0994/tcn/tcn.py#L90).

### - Classification (Many to many) e.g. copy memory task

```
model = tcn.dilated_tcn(num_feat=input_dim,
                        num_classes=10,
                        nb_filters=10,
                        kernel_size=8,
                        dilations=[1, 2, 4, 8],
                        nb_stacks=8,
                        max_len=timesteps,
                        activation='norm_relu')
```

### - Classification (Many to one) e.g. sequential mnist task

```
model = tcn.dilated_tcn(output_slice_index='last',
                        num_feat=input_dim,
                        num_classes=10,
                        nb_filters=64,
                        kernel_size=8,
                        dilations=[1, 2, 4, 8],
                        nb_stacks=8,
                        max_len=timesteps,
                        activation='norm_relu')
```

## Installation

```
git clone git@github.com:philipperemy/keras-tcn.git
cd keras-tcn
virtualenv -p python3.6 venv
source venv/bin/activate
pip install -r requirements.txt # change to tensorflow if you dont have a gpu.
python setup.py install # install keras-tcn as a package
```

## Run

Once `keras-tcn` is installed as a package, you can take a glimpse of what's possible to do with TCNs. Some tasks examples are  available in the repository for this purpose:

```
cd adding_problem/
python main.py # run adding problem task

cd copy_memory/
python main.py # run copy memory task

cd mnist_pixel/
python main.py # run sequential mnist pixel task
```

## Tasks

### Adding Task

The task consists of feeding a large array of decimal numbers to the network, along with a boolean array of the same length. The objective is to sum the two decimals where the boolean array contain the two 1s.

#### Explanation

<p align="center">
  <img src="misc/Adding_Task.png">
  <b>Adding Problem Task</b><br><br>
</p>

#### Implementation results

The model takes time to learn this task. It's symbolized by a very long plateau (could take ~8 epochs on some runs).

```
200000/200000 [==============================] - 293s 1ms/step - loss: 0.1731 - val_loss: 0.1662
200000/200000 [==============================] - 289s 1ms/step - loss: 0.1675 - val_loss: 0.1665
200000/200000 [==============================] - 287s 1ms/step - loss: 0.1670 - val_loss: 0.1665
200000/200000 [==============================] - 288s 1ms/step - loss: 0.1668 - val_loss: 0.1669
200000/200000 [==============================] - 285s 1ms/step - loss: 0.1085 - val_loss: 0.0019
200000/200000 [==============================] - 285s 1ms/step - loss: 0.0011 - val_loss: 4.1667e-04
200000/200000 [==============================] - 282s 1ms/step - loss: 6.0470e-04 - val_loss: 6.7708e-04
200000/200000 [==============================] - 282s 1ms/step - loss: 4.3099e-04 - val_loss: 7.3898e-04
200000/200000 [==============================] - 282s 1ms/step - loss: 3.9102e-04 - val_loss: 1.8727e-04
200000/200000 [==============================] - 280s 1ms/step - loss: 3.1040e-04 - val_loss: 0.0010
200000/200000 [==============================] - 281s 1ms/step - loss: 3.1166e-04 - val_loss: 2.2333e-04
200000/200000 [==============================] - 281s 1ms/step - loss: 2.8046e-04 - val_loss: 1.5194e-04
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
30000/30000 [==============================] - 30s 1ms/step - loss: 0.1174 - acc: 0.9586 - val_loss: 0.0370 - val_acc: 0.9859
30000/30000 [==============================] - 26s 874us/step - loss: 0.0367 - acc: 0.9859 - val_loss: 0.0363 - val_acc: 0.9859
30000/30000 [==============================] - 26s 852us/step - loss: 0.0361 - acc: 0.9859 - val_loss: 0.0358 - val_acc: 0.9859
30000/30000 [==============================] - 26s 872us/step - loss: 0.0355 - acc: 0.9859 - val_loss: 0.0349 - val_acc: 0.9859
30000/30000 [==============================] - 25s 850us/step - loss: 0.0339 - acc: 0.9864 - val_loss: 0.0291 - val_acc: 0.9881
30000/30000 [==============================] - 26s 856us/step - loss: 0.0235 - acc: 0.9896 - val_loss: 0.0159 - val_acc: 0.9944
30000/30000 [==============================] - 26s 872us/step - loss: 0.0169 - acc: 0.9929 - val_loss: 0.0125 - val_acc: 0.9966
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
60000/60000 [==============================] - 118s 2ms/step - loss: 0.2348 - acc: 0.9265 - val_loss: 0.1308 - val_acc: 0.9579
60000/60000 [==============================] - 116s 2ms/step - loss: 0.0973 - acc: 0.9698 - val_loss: 0.0645 - val_acc: 0.9798
[...]
60000/60000 [==============================] - 112s 2ms/step - loss: 0.0075 - acc: 0.9978 - val_loss: 0.0547 - val_acc: 0.9894
60000/60000 [==============================] - 111s 2ms/step - loss: 0.0093 - acc: 0.9968 - val_loss: 0.0585 - val_acc: 0.9895
```


## References
- https://github.com/locuslab/TCN/ (TCN for Pytorch)
- https://arxiv.org/pdf/1803.01271.pdf (An Empirical Evaluation of Generic Convolutional and Recurrent Networks
for Sequence Modeling)
- https://arxiv.org/pdf/1609.03499.pdf (Original Wavenet paper)
