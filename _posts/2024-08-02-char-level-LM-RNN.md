---
layout: post
comments: true
title: "Implementing a Recurrent Neural Network (RNN) From Scratch in Python: Character-Level Language Model Case Study"
excerpt: 
date: 2024-08-02
mathjax: true
---

In this blog post, we'll dive deep into the implementation of a character-level language model using a vanilla Recurrent Neural Network (RNN). This type of model can learn to generate text one character at a time, capturing the patterns and structure of the language it's trained on. We'll walk through the code, explain the key concepts, and provide insights into how this model works.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preparation](#data-preparation)
3. [RNN Model Architecture](#rnn-model-architecture)
4. [Forward Pass](#forward-pass)
5. [Backward Pass and Training](#backward-pass-and-training)
6. [Sampling from the Model](#sampling-from-the-model)
7. [Putting It All Together](#putting-it-all-together)
8. [Conclusion](#conclusion)

## Introduction

Recurrent Neural Networks are a class of neural networks designed to work with sequential data. They're particularly well-suited for tasks like language modeling, where the order and context of the input matter. In this implementation, we'll create a character-level language model, which means our model will learn to predict the next character in a sequence given the previous characters.

## Data Preparation

Before we can train our model, we need to prepare our data. Let's look at the `DataReader` class:

```python
class DataReader:
    def __init__(self, path, seq_length):
        self.fp = open(path, "r")
        self.data = self.fp.read()
        chars = list(set(self.data))
        self.char_to_ix = {ch:i for (i,ch) in enumerate(chars)}
        self.ix_to_char = {i:ch for (i,ch) in enumerate(chars)}
        self.data_size = len(self.data)
        self.vocab_size = len(chars)
        self.pointer = 0
        self.seq_length = seq_length

    def next_batch(self):
        input_start = self.pointer
        input_end = self.pointer + self.seq_length
        inputs = [self.char_to_ix[ch] for ch in self.data[input_start:input_end]]
        targets = [self.char_to_ix[ch] for ch in self.data[input_start+1:input_end+1]]
        self.pointer += self.seq_length
        if self.pointer + self.seq_length + 1 >= self.data_size:
            self.pointer = 0
        return inputs, targets

    def just_started(self):
        return self.pointer == 0

    def close(self):
        self.fp.close()
```

This class handles reading the input text file and preparing the data for our model. Here's what it does:

1. Reads the entire text file into memory.
2. Creates a vocabulary of unique characters in the text.
3. Maps each character to a unique integer index and vice versa.
4. Implements a `next_batch()` method to provide sequences of characters for training.
5. Provides a `just_started()` method to check if we're at the beginning of the data.

The `seq_length` parameter determines how many characters the model will see at once during training. This is important because it affects the model's ability to capture long-range dependencies in the text.

The `just_started()` method is used to check if we've reset our pointer to the beginning of the data. This is useful for initializing the hidden state when we start a new epoch of training.

## RNN Model Architecture

Now, let's look at the core of our implementation: the `RNN` class. We'll break it down into several parts.

```python
class RNN:
    def __init__(self, hidden_size, vocab_size, seq_length, learning_rate):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.U = np.random.randn(hidden_size, vocab_size)*0.01
        self.W = np.random.randn(hidden_size, hidden_size)*0.01
        self.V = np.random.randn(vocab_size, hidden_size)*0.01
        self.b = np.zeros((hidden_size, 1))
        self.c = np.zeros((vocab_size, 1))
        
        # Memory variables for Adagrad optimization
        self.mU, self.mW, self.mV = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        self.mb, self.mc = np.zeros_like(self.b), np.zeros_like(self.c)

```

In the constructor, we initialize the model parameters:

- `U`: Input-to-hidden weight matrix
- `W`: Hidden-to-hidden weight matrix
- `V`: Hidden-to-output weight matrix
- `b`: Hidden layer bias
- `c`: Output layer bias

We also initialize memory variables for the Adagrad optimization algorithm, which we'll use to update our parameters during training.

```python
def update_model(self, dU, dW, dV, db, dc):
    # parameter update with adagrad
    for param, dparam, mem in zip([self.U, self.W, self.V, self.b, self.c],
                                  [dU, dW, dV, db, dc],
                                  [self.mU, self.mW, self.mV, self.mb, self.mc]):
        mem += dparam * dparam
        param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update
```

This `update_model` method implements the Adagrad optimization algorithm to update the model parameters. Adagrad adapts the learning rate for each parameter based on the historical gradients, which can help with convergence, especially when dealing with sparse data. The `update_model` method is called after each backward pass to adjust the model parameters based on the computed gradients. This is a crucial step in the training process, as it's how the model learns and improves its performance over time.

## Forward Pass

The forward pass is where we compute the model's predictions given an input sequence. Let's look at the `forward` method:

```python
def forward(self, inputs, hprev):
    xs, hs, os, ycap = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    for t in range(len(inputs)):
        xs[t] = np.zeros((self.vocab_size,1))
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(self.U,xs[t]) + np.dot(self.W,hs[t-1]) + self.b)
        os[t] = np.dot(self.V,hs[t]) + self.c
        ycap[t] = self.softmax(os[t])
    return xs, hs, ycap
```

This method implements the following steps for each time step in the input sequence:

1. Convert the input character to a one-hot encoded vector (`xs[t]`).
2. Compute the hidden state (`hs[t]`) using the tanh activation function.
3. Compute the output scores (`os[t]`).
4. Apply softmax to get the predicted probabilities for the next character (`ycap[t]`).

The equations governing this process are:

- Hidden state: `h[t] = tanh(U * x[t] + W * h[t-1] + b)`
- Output: `o[t] = V * h[t] + c`
- Predictions: `y[t] = softmax(o[t])`

## Backward Pass and Training

The backward pass is where we compute the gradients of our loss function with respect to the model parameters. This is done using backpropagation through time (BPTT). Let's look at the `backward` method:

```python
def backward(self, xs, hs, ps, targets):
    dU, dW, dV = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
    db, dc = np.zeros_like(self.b), np.zeros_like(self.c)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(self.seq_length)):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        dV += np.dot(dy, hs[t].T)
        dc += dc
        dh = np.dot(self.V.T, dy) + dhnext
        dhrec = (1 - hs[t] * hs[t]) * dh
        db += dhrec
        dU += np.dot(dhrec, xs[t].T)
        dW += np.dot(dhrec, hs[t-1].T)
        dhnext = np.dot(self.W.T, dhrec)
    for dparam in [dU, dW, dV, db, dc]:
        np.clip(dparam, -5, 5, out=dparam)
    return dU, dW, dV, db, dc
```

This method computes the gradients of the loss with respect to all parameters. The process involves:

1. Computing the gradient of the loss with respect to the output (dy).
2. Backpropagating this gradient through the output layer to get dV and dc.
3. Backpropagating through the hidden layer to get dU, dW, and db.
4. Clipping gradients to prevent exploding gradients.

The `train` method ties everything together:

```python
def train(self, data_reader):
    iter_num = 0
    threshold = 0.01
    smooth_loss = -np.log(1.0/data_reader.vocab_size)*self.seq_length
    while (smooth_loss > threshold):
        if data_reader.just_started():
            hprev = np.zeros((self.hidden_size,1))
        inputs, targets = data_reader.next_batch()
        xs, hs, ps = self.forward(inputs, hprev)
        dU, dW, dV, db, dc = self.backward(xs, hs, ps, targets)
        loss = self.loss(ps, targets)
        self.update_model(dU, dW, dV, db, dc)
        smooth_loss = smooth_loss*0.999 + loss*0.001
        hprev = hs[self.seq_length-1]
        if not iter_num%500:
            sample_ix = self.sample(hprev, inputs[0], 200)
            print( ''.join(data_reader.ix_to_char[ix] for ix in sample_ix))
            print( "\n\niter :%d, loss:%f"%(iter_num, smooth_loss))
        iter_num += 1
```

This method repeatedly:

1. Gets a batch of data
2. Performs a forward pass
3. Computes the loss
4. Performs a backward pass to get gradients
5. Updates the model parameters
6. Occasionally samples from the model to check its progress

## Sampling from the Model

Once we've trained our model, we can use it to generate new text. The `sample` method does this:

```python
def sample(self, h, seed_ix, n):
    x = np.zeros((self.vocab_size,1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h = np.tanh(np.dot(self.U, x) + np.dot(self.W, h) + self.b)
        y = np.dot(self.V, h) + self.c
        p = np.exp(y)/np.sum(np.exp(y))
        ix = np.random.choice(range(self.vocab_size), p = p.ravel())
        x = np.zeros((self.vocab_size,1))
        x[ix] = 1
        ixes.append(ix)
    return ixes
```

This method:

1. Starts with a seed character
2. Repeatedly:
   - Computes the next hidden state
   - Computes the probabilities for the next character
   - Samples a character based on these probabilities
   - Uses this character as the input for the next step


## Putting It All Together

Here's how we can use our implementation:

```python
seq_length = 25
data_reader = DataReader('data/input.txt', seq_length)
rnn = RNN(hidden_size=100, vocab_size=data_reader.vocab_size, seq_length=seq_length, learning_rate=1e-1)
rnn.train(data_reader)

# Generate some text
generated_text = rnn.predict(data_reader, 'speak', 50)
print(generated_text)
```

This code:

1. Creates a `DataReader` to handle our input data
2. Creates an `RNN` model
3. Trains the model
4. Uses the trained model to generate some text starting with the word "speak"

It's important to note that in this implementation, we're using the "Tiny Shakespeare" dataset. This dataset contains all of Shakespeare's work in a single file under 1 MB, which is considerably smaller than the large-scale datasets used in state-of-the-art language models. The Tiny Shakespeare dataset is perfect for understanding and experimenting with RNN-based language models on a smaller scale.

Using a smaller dataset like Tiny Shakespeare allows for faster training and experimentation, making it ideal for learning purposes. However, it's worth mentioning that the quality and diversity of the generated text will be limited compared to models trained on larger, more diverse datasets.

## Conclusion

In this blog post, we've explored the implementation of a character-level language model using a vanilla RNN, trained on the Tiny Shakespeare dataset. We've seen how to prepare the data, implement the forward and backward passes, train the model, and generate text from the trained model.

This implementation, while simple and trained on a small dataset, captures the core ideas behind RNNs and language modeling. It serves as an excellent starting point for understanding these concepts. However, it's worth noting that modern language models often use more advanced architectures like LSTMs or Transformers, and are trained on much larger and more diverse datasets, often containing billions of words from various sources across the internet.

The use of the Tiny Shakespeare dataset in this implementation allows for quick experimentation and learning, but also highlights the limitations of training on a small, specialized corpus. The generated text will likely have a distinctly Shakespearean flavor and may struggle with more modern language constructs or diverse topics.

Understanding this implementation provides a solid foundation for exploring more advanced topics in natural language processing and deep learning. As you progress, you might want to experiment with larger datasets, more complex architectures, and advanced training techniques to see how they improve the quality and versatility of the generated text.

Happy coding, and may your RNNs speak with the eloquence of the Bard himself!


## References
1. [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/), Andrej Karpathy, May 2015.
2. [Recurrent Neural Networks (RNNs): Implementing an RNN from Scratch in Python](https://towardsdatascience.com/recurrent-neural-networks-rnns-3f06d7653a85), Javaid Nabi, Jul 2019.
2. [Recurrent Neural Networks Tutorial, Pt 1: Introduction to RNNs](https://dennybritz.com/posts/wildml/recurrent-neural-networks-tutorial-part-1/), Denny Britz, Sept 2015.
