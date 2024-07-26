---
layout: post
comments: true
title: "Implementation of Karpathy's "Neural Networks: Zero to Hero" Lecture Series"
excerpt: 
date: 2024-07-15
mathjax: true
---
This blog post contains my detailed implementation of Andrej Karpathy's [Neural Networks: Zero to Hero youtube](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) lecture series & exercises in Jupyter Notebook. The articles go into extreme details to ensure a proper, robust fundamental understanding of neural networks. This lecture series covers neural networks (NNs) and how to build them from scratch in code. The course starts with the basics of backpropagation, then proceeds to multi-layer perceptrons (MLPs), convolutional neural networks (CNNs), and finally builds up to modern deep neural networks like generative pre-trained transformers (GPTs). The course also introduces and covers diagnostic tools for understanding neural networks dynamics and performance. The field of focus in this course is language modeling (LM) because not only are language models a superb place to learn deep learning but also most of the skills learned here are immediately transferable to other fields of deep learning such as computer vision (CV). The full project can be found on [github](https://github.com/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero).

Three engines are built and leveraged in this lecture series: `micrograd`, `makemore` and `gpt`. The 1st two engines are not meant to be too heavyweight of libraries with a billion switches and knobs. They should exist as a single hackable file, and are mostly intended for educational purposes. Python and [PyTorch](https://pytorch.org) are the only requirements.
* `micrograd`: A tiny **autograd** (automatic gradient) engine that implements **backpropagation** (reverse-mode autodiff) over a dynamically built **DAG** (Directed Acyclic Graph) and a **small NNs library** on  top of it with a **PyTorch-like API**. It's a minimalistic, scalar-valued, auto-differentiation (**autodiff**) engine in python.
  
* `makemore`: *makemore* takes one text file as input, where each line is assumed to be one training thing, and generates more things like it. Under the hood, it is an **autoregressive character-level language model**, with a <u>wide choice of models from bigrams all the way to a Transformer (exactly as seen in GPT)</u>. For example, we can feed it a database of names, and makemore will generate cool baby name ideas that all sound name-like, but are not already existing names. Or if we feed it a database of company names then we can generate new ideas for a name of a company. Or we can just feed it valid scrabble words and generate english-like babble.
  ```
  "As the name suggests, makemore makes more."
  ```

* `gpt`: <u>Generative Pre-trained Transformer,</u> otherwise known as **GPT**, is a large language model (LLM) that is trained on a significant large size of text data to understand and generate human-like text sequentially. The "transformer" part refers to the model's architecture, which was introduced and inspired by the 2017 "Attention Is All You Need" paper. GPTs are based on the _**transformer**_ architecture, _**pre-trained**_ on large data sets of unlabelled text, and able to _**generate**_ novel human-like content.

---
The implementation of each lecture can be found below:
> Lecture 1: micrograd [notebook](https://nbviewer.org/github/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/001_micrograd/micrograd.ipynb)<br>
> Lecture 2: makemore 1 bigrams [notebook](https://nbviewer.org/github/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/002_makemore_Bigrams/makemore_Bigrams.ipynb)<br>
> Lecture 3: makemore 2 multi-layer perceptron [notebook](https://nbviewer.org/github/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/003_makemore_MLP/makemore_MLP.ipynb)<br>
> Lecture 4: makemore 3 batch normalization [notebook](https://nbviewer.org/github/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/004_makemore_BatchNorm/makemore_BatchNorm.ipynb)<br>
> Lecture 5: makemore 4 backpropagation on steroids [notebook](https://nbviewer.org/github/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/005_makemore_BackpropNinja/makemore_Backprop.ipynb)<br>
> Lecture 6: makemore 5 wavenet [notebook](https://github.com/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/006_makemore_WaveNet/makemore_WaveNet.ipynb)<br>
> Lecture 7: Generative Pretrained Transformers (GPT) [notebook](https://nbviewer.org/github/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/007_GPT/gpt.ipynb)<br>
