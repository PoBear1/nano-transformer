# Nano-transformer: re-implementation of the Transformer architecture in PyTorch
This is a re-implementation of the transformer architecture from the bottom - up using only PyTorch primitives, i.e. none of the attention modules or other pre-built modules later added to PyTorch. This also includes the training framework required to get a proper, working model. Core implementation based off the following two sources:
 - [Attention is all you need (2017), Vaswani et. al.](https://arxiv.org/abs/1706.03762)
 - [Probabilistic Machine Learning: An Introduction, Chapter 15 (2022), Murphy](https://probml.github.io/pml-book/book1.html)

The rest of the code for token generation is from Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat) repo.