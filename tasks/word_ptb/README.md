## Word-level Language Modeling

Ref: https://arxiv.org/pdf/1803.01271.

### Overview

In word-level language modeling tasks, each element of the sequence is a word, where the model is expected to predict
the next incoming word in the text. We evaluate the temporal convolutional network as a word-level language model on
PennTreebank

### Data

**PennTreebank**: We used the PennTreebank (PTB) (Marcus et al., 1993) for both character-level and word-level
language modeling. When used as a character-level language corpus, PTB contains 5,059K characters for training,
396K for validation, and 446K for testing, with an alphabet
size of 50. When used as a word-level language corpus,
PTB contains 888K words for training, 70K for validation,
and 79K for testing, with a vocabulary size of 10K. This
is a highly studied but relatively small language modeling
dataset (Miyamoto & Cho, 2016; Krueger et al., 2017; Merity et al., 2017).
  
### Results

*Note that the implementation might be a bit different than what is quoted in the paper.*

**Word-level language modeling**. Language modeling remains one of the primary applications of recurrent networks
and many recent works have focused on optimizing LSTMs
for this task (Krueger et al., 2017; Merity et al., 2017).
Our implementation follows standard practice that ties the
weights of encoder and decoder layers for both TCN and
RNNs (Press & Wolf, 2016), which significantly reduces
the number of parameters in the model. For training, we use
SGD and anneal the learning rate by a factor of 0.5 for both
TCN and RNNs when validation accuracy plateaus.
On the smaller PTB corpus, an optimized LSTM architecture (with recurrent and embedding dropout, etc.) outperforms the TCN, while the TCN outperforms both GRU and
vanilla RNN. However, on the much larger Wikitext-103
corpus and the LAMBADA dataset (Paperno et al., 2016),
without any hyperparameter search, the TCN outperforms the LSTM results of Grave et al. (2017), achieving much
lower perplexities.

**Character-level language modeling**. On character-level
language modeling (PTB and text8, accuracy measured in
bits per character), the generic TCN outperforms regularized LSTMs and GRUs as well as methods such as Normstabilized LSTMs (Krueger & Memisevic, 2015). (Specialized architectures exist that outperform all of these, see the
supplement.)