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
                        nb_filters=24,
                        kernel_size=8,
                        dilatations=[1, 2, 4, 8],
                        nb_stacks=8,
                        max_len=timesteps,
                        activation='norm_relu',
                        regression=True)
```

### - Classification (Many to one) e.g. copy memory task

```
model = tcn.dilated_tcn(num_feat=input_dim,
                        num_classes=10,
                        nb_filters=10,
                        kernel_size=8,
                        dilatations=[1, 2, 4, 8],
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
                        dilatations=[1, 2, 4, 8],
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
pip install . # install keras-tcn
```

## Run

```
cd copy_memory/
python main.py # run copy memory task

cd mnist_pixel/
python main.py # run sequential mnist pixel task
```

## Tasks

<p align="center">
  <img src="misc/Adding_Task.png">
  <b>Adding Problem Task</b><br><br>
</p>


<p align="center">
  <img src="misc/Copy_Memory_Task.png">
  <b>Copy Memory Task</b><br><br>
</p>


<p align="center">
  <img src="misc/Sequential_MNIST_Task.png">
  <b>Sequential MNIST</b><br><br>
</p>


## References
- https://github.com/locuslab/TCN/ (TCN for Pytorch)
- https://arxiv.org/pdf/1803.01271.pdf (An Empirical Evaluation of Generic Convolutional and Recurrent Networks
for Sequence Modeling)
- https://arxiv.org/pdf/1609.03499.pdf (Original Wavenet paper)
