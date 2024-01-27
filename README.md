# Transformer From Scratch

This is a PyTorch implementation of the Transformer model. I wrote this for my own understanding, but it is test-driven (see tests)
and is meant to be clean and efficient. This code is also documented and explained
in my blog post [here](http://localhost:8891/2023/12/23/transformer-from-scratch-in-pytorch-model/).

For this implementation, we mostly follow the original architecture 
from Attention is All You Need, although we follow Pre-Layer Normalization as 
it has been shown ([Lei Ba et. al.](https://arxiv.org/pdf/1607.06450.pdf)) to lead to better training than the originally 
proposed Post-Layer Normalization architecture. 

The main Transformer code is in [this](https://github.com/ltrujello/Transformer/blob/main/src/transformer/model.py) file.
