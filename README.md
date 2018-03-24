# Keras TCN
*Keras Temporal Convolutional Network*

## Why Temporal Convolutional Network?

- TCNs exhibit longer memory than recurrent architectures with the same capacity.
- Constantly performs better than LSTM/GRU architectures on a vast range of tasks (Seq. MNIST, Adding Problem, Copy Memory, Word-level PTB...).
- Parallelism, flexible receptive field size, stable gradients, low memory requirements for training, variable length inputs...

<p align="center">
  <img src="misc/Dilated_Conv.png">
  <b>Visualization of a stack of dilated causal convolutional layers (Wavenet, 2016)</b><br><br>
</p>

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
