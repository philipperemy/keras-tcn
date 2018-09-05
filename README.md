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
200000/200000 [==============================] - 451s 2ms/step - loss: 0.1749 - val_loss: 0.1662
200000/200000 [==============================] - 449s 2ms/step - loss: 0.1681 - val_loss: 0.1676
200000/200000 [==============================] - 449s 2ms/step - loss: 0.1677 - val_loss: 0.1663
200000/200000 [==============================] - 449s 2ms/step - loss: 0.1676 - val_loss: 0.1652
200000/200000 [==============================] - 449s 2ms/step - loss: 0.1165 - val_loss: 0.0093
200000/200000 [==============================] - 448s 2ms/step - loss: 0.0083 - val_loss: 0.0033
200000/200000 [==============================] - 448s 2ms/step - loss: 0.0040 - val_loss: 0.0012
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

#### Implementation results

```
10000/10000 [==============================] - 20s 2ms/step - loss: 0.3474 - acc: 0.8985 - val_loss: 0.0362 - val_acc: 0.9859
10000/10000 [==============================] - 13s 1ms/step - loss: 0.0360 - acc: 0.9859 - val_loss: 0.0353 - val_acc: 0.9859
10000/10000 [==============================] - 13s 1ms/step - loss: 0.0351 - acc: 0.9859 - val_loss: 0.0345 - val_acc: 0.9859
10000/10000 [==============================] - 13s 1ms/step - loss: 0.0342 - acc: 0.9860 - val_loss: 0.0336 - val_acc: 0.9860
10000/10000 [==============================] - 13s 1ms/step - loss: 0.0332 - acc: 0.9865 - val_loss: 0.0307 - val_acc: 0.9883
10000/10000 [==============================] - 13s 1ms/step - loss: 0.0240 - acc: 0.9898 - val_loss: 0.0157 - val_acc: 0.9933
10000/10000 [==============================] - 13s 1ms/step - loss: 0.0136 - acc: 0.9951 - val_loss: 0.0094 - val_acc: 0.9976
10000/10000 [==============================] - 13s 1ms/step - loss: 0.0087 - acc: 0.9978 - val_loss: 0.0049 - val_acc: 1.0000
10000/10000 [==============================] - 14s 1ms/step - loss: 0.0050 - acc: 0.9992 - val_loss: 0.0020 - val_acc: 1.0000
```

### Sequential MNIST

#### Explanation

The idea here is to consider MNIST images as 1-D sequences and feed them to the network. This task is particularly hard because sequences are 28*28 = 784 elements. In order to classify correctly, the network has to remember all the sequence. Usual LSTM are unable to perform well on this task.

<p align="center">
  <img src="misc/Sequential_MNIST_Task.png">
  <b>Sequential MNIST</b><br><br>
</p>


<details><summary>Implementation results (Script output)</summary>
<p>

```
60000/60000 [==============================] - 118s 2ms/step - loss: 0.2348 - acc: 0.9265 - val_loss: 0.1308 - val_acc: 0.9579
60000/60000 [==============================] - 116s 2ms/step - loss: 0.0973 - acc: 0.9698 - val_loss: 0.0645 - val_acc: 0.9798
60000/60000 [==============================] - 114s 2ms/step - loss: 0.0763 - acc: 0.9761 - val_loss: 0.0629 - val_acc: 0.9800
60000/60000 [==============================] - 115s 2ms/step - loss: 0.0614 - acc: 0.9809 - val_loss: 0.0519 - val_acc: 0.9826
60000/60000 [==============================] - 115s 2ms/step - loss: 0.0543 - acc: 0.9831 - val_loss: 0.0442 - val_acc: 0.9862
60000/60000 [==============================] - 113s 2ms/step - loss: 0.0467 - acc: 0.9853 - val_loss: 0.0613 - val_acc: 0.9822
60000/60000 [==============================] - 116s 2ms/step - loss: 0.0423 - acc: 0.9868 - val_loss: 0.0500 - val_acc: 0.9845
60000/60000 [==============================] - 114s 2ms/step - loss: 0.0409 - acc: 0.9870 - val_loss: 0.0412 - val_acc: 0.9877
60000/60000 [==============================] - 115s 2ms/step - loss: 0.0366 - acc: 0.9880 - val_loss: 0.0464 - val_acc: 0.9865
60000/60000 [==============================] - 115s 2ms/step - loss: 0.0330 - acc: 0.9901 - val_loss: 0.0353 - val_acc: 0.9879
60000/60000 [==============================] - 114s 2ms/step - loss: 0.0299 - acc: 0.9899 - val_loss: 0.0418 - val_acc: 0.9875
60000/60000 [==============================] - 115s 2ms/step - loss: 0.0296 - acc: 0.9905 - val_loss: 0.0415 - val_acc: 0.9870
60000/60000 [==============================] - 115s 2ms/step - loss: 0.0273 - acc: 0.9912 - val_loss: 0.0409 - val_acc: 0.9880
60000/60000 [==============================] - 115s 2ms/step - loss: 0.0251 - acc: 0.9919 - val_loss: 0.0470 - val_acc: 0.9866
60000/60000 [==============================] - 114s 2ms/step - loss: 0.0238 - acc: 0.9926 - val_loss: 0.0359 - val_acc: 0.9898
60000/60000 [==============================] - 115s 2ms/step - loss: 0.0230 - acc: 0.9923 - val_loss: 0.0331 - val_acc: 0.9903
60000/60000 [==============================] - 116s 2ms/step - loss: 0.0202 - acc: 0.9933 - val_loss: 0.0407 - val_acc: 0.9882
60000/60000 [==============================] - 113s 2ms/step - loss: 0.0193 - acc: 0.9937 - val_loss: 0.0389 - val_acc: 0.9897
60000/60000 [==============================] - 114s 2ms/step - loss: 0.0196 - acc: 0.9934 - val_loss: 0.0415 - val_acc: 0.9889
60000/60000 [==============================] - 115s 2ms/step - loss: 0.0193 - acc: 0.9937 - val_loss: 0.0483 - val_acc: 0.9872
60000/60000 [==============================] - 114s 2ms/step - loss: 0.0161 - acc: 0.9947 - val_loss: 0.0410 - val_acc: 0.9885
60000/60000 [==============================] - 115s 2ms/step - loss: 0.0173 - acc: 0.9944 - val_loss: 0.0473 - val_acc: 0.9874
60000/60000 [==============================] - 115s 2ms/step - loss: 0.0157 - acc: 0.9947 - val_loss: 0.0385 - val_acc: 0.9899
60000/60000 [==============================] - 114s 2ms/step - loss: 0.0133 - acc: 0.9955 - val_loss: 0.0479 - val_acc: 0.9871
60000/60000 [==============================] - 114s 2ms/step - loss: 0.0155 - acc: 0.9951 - val_loss: 0.0395 - val_acc: 0.9893
60000/60000 [==============================] - 114s 2ms/step - loss: 0.0127 - acc: 0.9957 - val_loss: 0.0489 - val_acc: 0.9890
60000/60000 [==============================] - 115s 2ms/step - loss: 0.0149 - acc: 0.9953 - val_loss: 0.0527 - val_acc: 0.9871
60000/60000 [==============================] - 115s 2ms/step - loss: 0.0133 - acc: 0.9957 - val_loss: 0.0447 - val_acc: 0.9888
60000/60000 [==============================] - 116s 2ms/step - loss: 0.0135 - acc: 0.9956 - val_loss: 0.0496 - val_acc: 0.9891
60000/60000 [==============================] - 115s 2ms/step - loss: 0.0127 - acc: 0.9960 - val_loss: 0.0517 - val_acc: 0.9890
60000/60000 [==============================] - 115s 2ms/step - loss: 0.0121 - acc: 0.9964 - val_loss: 0.0504 - val_acc: 0.9884
60000/60000 [==============================] - 116s 2ms/step - loss: 0.0124 - acc: 0.9958 - val_loss: 0.0481 - val_acc: 0.9884
60000/60000 [==============================] - 116s 2ms/step - loss: 0.0134 - acc: 0.9954 - val_loss: 0.0502 - val_acc: 0.9896
60000/60000 [==============================] - 116s 2ms/step - loss: 0.0108 - acc: 0.9966 - val_loss: 0.0524 - val_acc: 0.9887
60000/60000 [==============================] - 115s 2ms/step - loss: 0.0112 - acc: 0.9963 - val_loss: 0.0482 - val_acc: 0.9892
60000/60000 [==============================] - 115s 2ms/step - loss: 0.0111 - acc: 0.9966 - val_loss: 0.0561 - val_acc: 0.9882
60000/60000 [==============================] - 116s 2ms/step - loss: 0.0114 - acc: 0.9964 - val_loss: 0.0544 - val_acc: 0.9880
60000/60000 [==============================] - 116s 2ms/step - loss: 0.0107 - acc: 0.9966 - val_loss: 0.0521 - val_acc: 0.9897
60000/60000 [==============================] - 116s 2ms/step - loss: 0.0107 - acc: 0.9965 - val_loss: 0.0477 - val_acc: 0.9892
60000/60000 [==============================] - 115s 2ms/step - loss: 0.0105 - acc: 0.9968 - val_loss: 0.0474 - val_acc: 0.9899
60000/60000 [==============================] - 117s 2ms/step - loss: 0.0110 - acc: 0.9969 - val_loss: 0.0489 - val_acc: 0.9889
60000/60000 [==============================] - 117s 2ms/step - loss: 0.0094 - acc: 0.9969 - val_loss: 0.0544 - val_acc: 0.9889
60000/60000 [==============================] - 115s 2ms/step - loss: 0.0098 - acc: 0.9970 - val_loss: 0.0548 - val_acc: 0.9875
60000/60000 [==============================] - 114s 2ms/step - loss: 0.0099 - acc: 0.9967 - val_loss: 0.0598 - val_acc: 0.9891
60000/60000 [==============================] - 116s 2ms/step - loss: 0.0090 - acc: 0.9971 - val_loss: 0.0609 - val_acc: 0.9888
60000/60000 [==============================] - 117s 2ms/step - loss: 0.0098 - acc: 0.9969 - val_loss: 0.0513 - val_acc: 0.9889
60000/60000 [==============================] - 117s 2ms/step - loss: 0.0088 - acc: 0.9973 - val_loss: 0.0533 - val_acc: 0.9888
60000/60000 [==============================] - 115s 2ms/step - loss: 0.0094 - acc: 0.9970 - val_loss: 0.0574 - val_acc: 0.9888
60000/60000 [==============================] - 115s 2ms/step - loss: 0.0079 - acc: 0.9974 - val_loss: 0.0509 - val_acc: 0.9892
60000/60000 [==============================] - 116s 2ms/step - loss: 0.0094 - acc: 0.9970 - val_loss: 0.0548 - val_acc: 0.9888
60000/60000 [==============================] - 115s 2ms/step - loss: 0.0079 - acc: 0.9974 - val_loss: 0.0548 - val_acc: 0.9904
60000/60000 [==============================] - 113s 2ms/step - loss: 0.0073 - acc: 0.9979 - val_loss: 0.0527 - val_acc: 0.9898
60000/60000 [==============================] - 112s 2ms/step - loss: 0.0082 - acc: 0.9974 - val_loss: 0.0538 - val_acc: 0.9897
60000/60000 [==============================] - 112s 2ms/step - loss: 0.0075 - acc: 0.9978 - val_loss: 0.0547 - val_acc: 0.9894
60000/60000 [==============================] - 111s 2ms/step - loss: 0.0093 - acc: 0.9968 - val_loss: 0.0585 - val_acc: 0.9895
```

</p>
</details>





## References
- https://github.com/locuslab/TCN/ (TCN for Pytorch)
- https://arxiv.org/pdf/1803.01271.pdf (An Empirical Evaluation of Generic Convolutional and Recurrent Networks
for Sequence Modeling)
- https://arxiv.org/pdf/1609.03499.pdf (Original Wavenet paper)
