---
layout: post
comments: true
title: "Implementation of Karpathy's Neural Networks: Zero to Hero Lecture Series"
excerpt:  |
    This blog post presents my detailed implementation of Andrej Karpathy's Neural Networks: Zero to Hero YouTube lecture series and exercises in Jupyter Notebook. The articles go deeply from NNs to MLPs to CNNs to LLMs to ensure a thorough and robust understanding of neural networks.
date: 2024-07-15
mathjax: true
---

## Introduction
This blog post presents my detailed implementation of Andrej Karpathy's [Neural Networks: Zero to Hero YouTube](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) lecture series and exercises in Jupyter Notebook. The articles delve deeply into each topic to ensure a thorough and robust understanding of neural networks. The lecture series covers neural networks (NNs) and demonstrates how to build them from scratch in code. It begins with the basics of backpropagation, then moves on to multi-layer perceptrons (MLPs), convolutional neural networks (CNNs), and ultimately builds up to modern deep neural networks, such as Large Language Models (LLMs) like generative pre-trained transformers (GPTs), and LLM tokenization via Byte Pair Encoding (BPE). The course also introduces and explains diagnostic tools for understanding neural network dynamics and performance. The primary focus is on language modeling (LM), as language models provide an excellent foundation for learning deep learning concepts, and most of the skills acquired here are immediately transferable to other areas of deep learning, such as computer vision (CV). The full project can be found on [GitHub](https://github.com/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero).

Four engines are built and leveraged in this lecture series: `micrograd`, `makemore`, `gpt` and `minBPE`. The 1st two engines are not meant to be too heavyweight of libraries with a billion switches and knobs. They should exist as a single hackable file, and are mostly intended for educational purposes. Python and [PyTorch](https://pytorch.org) are the only requirements.

* `micrograd`: A tiny **autograd** (automatic gradient) engine that implements **backpropagation** (reverse-mode autodiff) over a dynamically built **DAG** (Directed Acyclic Graph) and a **small NNs library** on  top of it with a **PyTorch-like API**. It's a minimalistic, scalar-valued, auto-differentiation (**autodiff**) engine in python.

* `makemore`: `makemore` takes one text file as input, where each line is assumed to be one training thing, and generates more things like it. Under the hood, it is an **autoregressive character-level language model**, with a <u>wide choice of models from bigrams all the way to a Transformer (exactly as seen in GPT)</u>.  For example, we can feed it a database of names, and makemore will generate cool baby name ideas that all sound name-like, but are not already existing names. Or if we feed it a database of company names then we can generate new ideas for a name of a company. Or we can just feed it valid scrabble words and generate english-like babble.
  ```
  "As the name suggests, makemore makes more."
  ```

* `gpt`: <u>Generative Pre-trained Transformer,</u> otherwise known as `GPT`, is a large language model (LLM) that is trained on a significant large size of text data to understand and generate human-like text sequentially. The "transformer" part refers to the model's architecture, which was introduced and inspired by the 2017 ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper. GPTs are based on the **transformer** architecture, **pre-trained** on large data sets of unlabelled text, and able to **generate** novel human-like content.

* `minBPE`: A minimal, clean implementation of the Byte Pair Encoding (BPE) algorithm commonly used in **LLM tokenization**. The algorithm is based on the following paper: [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909). This tokenizer engine handles the crucial preprocessing step that converts raw text into tokens that language models can understand. BPE works at the **byte level**, processing **UTF-8-encoded strings** to efficiently handle a wide array of human languages and symbols. The `minBPE` tokenizer can train vocabulary and merges on text data, encode text to tokens, and decode tokens back to text - making it an essential component of the modern LLM pipeline used by models like `GPT`, `Llama`, and `Mistral`.

---
## Lecture Notebooks
The implementation of each lecture can be found below:
- **Lecture 1**: micrograd  
  [View Notebook →](https://nbviewer.org/github/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/001_micrograd/micrograd.ipynb)

- **Lecture 2**: makemore 1 (Bigram model)  
  [View Notebook →](https://nbviewer.org/github/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/002_makemore_Bigrams/makemore_Bigrams.ipynb)

- **Lecture 3**: makemore 2 (Multi-Layer Perceptron)  
  [View Notebook →](https://nbviewer.org/github/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/003_makemore_MLP/makemore_MLP.ipynb)

- **Lecture 4**: makemore 3 (Batch Normalization)  
  [View Notebook →](https://nbviewer.org/github/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/004_makemore_BatchNorm/makemore_BatchNorm.ipynb)

- **Lecture 5**: makemore 4 (Backprop Ninja)  
  [View Notebook →](https://nbviewer.org/github/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/005_makemore_BackpropNinja/makemore_Backprop.ipynb)

- **Lecture 6**: makemore 5 (WaveNet)  
  [View Notebook →](https://github.com/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/006_makemore_WaveNet/makemore_WaveNet.ipynb)

- **Lecture 7**: GPT (Transformer from scratch)  
  [View Notebook →](https://nbviewer.org/github/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/007_GPT/gpt.ipynb)

- **Lecture 8**: minBPE (GPT Tokenizer with Byte Pair Encoding [BPE])  
  [View Notebook →](https://nbviewer.org/github/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/008_minBPE/minbpe.ipynb)

- **Lecture 9**: GPT2 from scratch <br>
  [View Notebook →](https://nbviewer.org/github/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/009_GPT2/gpt-2-from-scratch.ipynb)

---


<!-- The implementation of each lecture can be found below:
> Lecture 1: micrograd [notebook](https://nbviewer.org/github/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/001_micrograd/micrograd.ipynb)<br>
> Lecture 2: makemore 1 bigrams [notebook](https://nbviewer.org/github/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/002_makemore_Bigrams/makemore_Bigrams.ipynb)<br>
> Lecture 3: makemore 2 multi-layer perceptron [notebook](https://nbviewer.org/github/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/003_makemore_MLP/makemore_MLP.ipynb)<br>
> Lecture 4: makemore 3 batch normalization [notebook](https://nbviewer.org/github/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/004_makemore_BatchNorm/makemore_BatchNorm.ipynb)<br>
> Lecture 5: makemore 4 backpropagation on steroids [notebook](https://nbviewer.org/github/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/005_makemore_BackpropNinja/makemore_Backprop.ipynb)<br>
> Lecture 6: makemore 5 wavenet [notebook](https://github.com/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/006_makemore_WaveNet/makemore_WaveNet.ipynb)<br>
> Lecture 7: Generative Pretrained Transformers (GPT) [notebook](https://nbviewer.org/github/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/007_GPT/gpt.ipynb)<br>
> Lecture 8: Byte Pair Encoding (BPE) [notebook](https://nbviewer.org/github/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/008_minBPE/minbpe.ipynb)<br>
> Lecture 9: GPT2 from scratch [notebook](https://nbviewer.org/github/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/009_GPT2/gpt2-from-scratch.ipynb)<br> -->
