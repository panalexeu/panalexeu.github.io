---
title: Reproducing "A Neural Probabilistic Language Model" (2003)
author: oleksii horchynskyi
---

Throughout my NLP career (already in the transformer era), the question that puzzled me the most was what kind of architectures preceded the transformer. In particular, what kind of iterative improvements, stacking on top of each other, led towards the autoregressive-decoder transformer architecture we have nowadays. Striving towards this knowledge, I discovered the Jurafsky and Martin NLP book [1], and while reading it, I decided to start a series of blog posts in which I discuss significant pre-transformer LM architectures for text generation.

I'd like to start this series with the architecture presented in a more than two decades old paper, "A Neural Probabilistic Language Model" (2003) [2]. I intentionally skip n-gram models as purely statistical LMs, though you can find my implementation of n-grams [here](https://github.com/panalexeu/nanongram). I believe the shortest and best way to describe the architecture presented in [2] is: a neural improvement on n-gram models. A neural probabilistic language model (NPLM) marks a discrete, observable shift from statistical LMs to neural LMs (fig. 1).

![nplm_pic](/assets/images/fig1_0.png)

Figure 1 - NPLM architecture figure taken from [2]

References:
1. Speech and Language Processing (3rd ed. draft), Dan Jurafsky and James H. Martin
2. "A Neural Probabilistic Language Model" (2003)