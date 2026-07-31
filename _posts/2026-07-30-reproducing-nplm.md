---
title: Reproducing "A Neural Probabilistic Language Model" (2003)
author: oleksii horchynskyi
---

Throughout my NLP career (already in the transformer era), the question that puzzled me the most was what kind of architectures preceded the transformer. In particular, what kind of iterative improvements, stacking on top of each other, led towards the autoregressive-decoder transformer architecture we have nowadays. Striving towards this knowledge, I discovered the Jurafsky and Martin NLP book [1], and while reading it, I decided to start a series of blog posts in which I discuss significant pre-transformer LM architectures for text generation.

I'd like to start this series with the architecture presented in a more than two decades old paper, "A Neural Probabilistic Language Model" (2003) [2]. I intentionally skip n-gram models as purely statistical LMs, though you can find my implementation of n-grams [here](https://github.com/panalexeu/nanongram). I believe the shortest and best way to describe the architecture presented in [2] is: a neural improvement on n-gram models. A neural probabilistic language model (NPLM) marks a discrete, observable shift from statistical LMs to neural LMs (fig. 1).

![nplm_pic](/assets/images/fig1_1.png)

Figure 1 - NPLM architecture figure taken from [2]

NPLM consists of three parts: an embedding matrix `C`, a hidden layer `H` (with tanh as nonlinearity), and an output projection layer `O` (which maps hidden layer activations back to the vocab).

NPLM scales linearly with the context size `n`, and the context is provided to the hidden layer as a single word feature vector (a concatenation of word feature activations from matrix `C`). Thus, positional information about words in a context is encoded purely through concatenation: a word's position corresponds directly to its position in the word feature vector (note: transformers, by contrast, encode positional information into the individual tokens *themselves*, which makes it even clearer why the transformer architecture is so parallelizable).

The paper also presents a version of the architecture with a skip connection: an additional matrix `W` is defined, mapping the concatenation of matrix `C` activations (the word feature vector) directly to the vocab, which is then summed with the activations of layer `O`, and a softmax is applied to the final sum. However, results presented in the paper suggest that skip connections *hurt* generalization:

```text
A reasonable interpretation is that direct input-to-output connections provide a bit more capacity and faster learning of the “linear” part of the mapping from word features to log probabilities. On the other hand, without those connections the hidden units form a tight bottleneck which might force better generalization.
```

Output probabilities of NPLM might also be mixed with output probabilities of a trained n-gram model. The paper states that such mixing always helps reduce perplexity.

Below is a PyTorch NPLM implementation with the skip connection matrix omitted (it hurts generalization, and compute power has increased so much over the past two decades that training for more epochs is no longer a problem):

```python
from typing import Self
from pathlib import Path

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from dataclasses import dataclass

@dataclass
class ModelConfig(): 
    n: int = 4
    embed: int = 32 
    hidden: int = 128
    vocab: int = 4096 
     
class Model(nn.Module): 
    def __init__(self, config: ModelConfig, vocab_table: dict = dict()):
        super().__init__()
        self.config = config 
        self.embed = nn.Embedding(
            num_embeddings=self.config.vocab, 
            embedding_dim=self.config.embed, 
        )
        self.down_proj = nn.Linear(
            in_features=self.config.embed * self.config.n, 
            out_features=self.config.hidden, 
            bias=True 
        ) 
        self.out_proj = nn.Linear(
            in_features=self.config.hidden, 
            out_features=self.config.vocab, 
            bias=True
        ) 
        self.vocab_table = vocab_table
        self.r_vocab_table = list(vocab_table.keys())

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module): 
        if isinstance(module, nn.Linear): 
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: 
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): 
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_num_params(self,): 
        return sum(p.numel() for p in self.parameters())
        
    def __call__(self, x: torch.Tensor, target: torch.Tensor | None): 
        assert x.size(0) == self.config.n, f"input size(0) should equal == {self.config.n}"

        x = self.embed(x)
        x = x.reshape(-1)
        x = F.tanh(self.down_proj(x)) 
        x = self.out_proj(x) 

        loss = None 
        if target is not None: 
            loss = F.cross_entropy(input=x, target=target)

        return x, loss

    def sample(
        self, 
        x: torch.Tensor,
        max_tokens: int, 
        stop_seq: list[int] = [], 
        t: float = 0.5, 
        stream: bool = True
    ): 
        sampled = []
        for _ in range(max_tokens): 
            y = self.__call__(x, target=None)[0]
            y /= t
            probs = F.softmax(y, dim=-1)
            sample = torch.multinomial(probs, num_samples=1).item()
            if sample in stop_seq: 
                break

            sampled.append(sample)
            x = torch.tensor(x[1:].tolist() + [sample])

            if stream: 
                print(self.r_vocab_table[sample] , end="")

        return sampled 

    def _load(self, ckpt_path: Path): 
        ckpt = torch.load(ckpt_path)
        self.load_state_dict(ckpt["model"])

    @classmethod
    def from_pretrained(cls, ckpt_path: Path) -> Self:
        ckpt = torch.load(ckpt_path)
        model = cls(
            config=ModelConfig(**ckpt['model_cfg']), 
            vocab_table=ckpt['vocab_table']
        )
        model.load_state_dict(ckpt['model'])

        return model 

    def configure_optimizer(self, lr: float, weight_decay: float): 
        """
        from paper: "...R is a weight decay penalty applied only to the weights 
        of the neural network and to the C matrix, not to the biases." 
        """
        decay, no_decay = [], [] 
        for name, param in self.named_parameters(): 
            if 'bias' in name: 
                no_decay.append(param)
            else: 
                decay.append(param)

        optimizer = torch.optim.SGD(params=[
            {'params': decay, 'weight_decay': weight_decay}, 
            {'params': no_decay, 'weight_decay': 0}
        ], lr=lr)

        return optimizer 
```

I'd like to finish this dense overview of the architecture by quoting a fragment of the proposed approach from the paper:

```text
In a nutshell, the idea of the proposed approach can be summarized as follows:

1. associate with each word in the vocabulary a distributed word feature vector (a real-
valued vector in Rm),
2. express the joint probability function of word sequences in terms of the feature vectors
of these words in the sequence, and
3. learn simultaneously the word feature vectors and the parameters of that probability
function.
```

And to build even more intuition, let me quote the paper's comparison of NPLM to n-grams from its introduction:

```text
First, it is not taking into account contexts farther than 1 or 2 words, second it is not taking into account the “similarity” between words. For example, having seen the sentence “The cat is walking in the bedroom” in the training corpus should help us generalize to make the sentence  “A dog was running in a room” almost as likely, simply because “dog” and “cat” (resp. “the” and “a”, “room” and “bedroom”, etc...) have similar semantic and grammatical roles.
```

The breakthrough of the paper at the time was an attempt to generalize better to unseen test sequences (e.g., n-gram models without interpolation simply output 0 probability for unseen sequences), by learning semantic word feature vector representations (matrix `C`), so that "The cat is walking in the bedroom" could naturally generalize to "A dog was running in a room."

You can find the NPLM implementation, with both training and inference scripts and a trained checkpoint, [here](https://github.com/panalexeu/nanonplm).

To give you a sense of what NPLM can achieve, with the following model configuration:

```python 
cfg = ModelConfig(
    n=5, 
    embed=30,
    hidden=100,
    vocab=13_986
)
```

trained using SGD for 1,000,000 steps on the first 297,832 tokens of tiny-shakespeare [3], it achieves a ppl of 136.44 on a test set of the last 29,783 unseen tokens.

Below is a generated sample for the input `to be or not to`, with temperature set to 0.7:

```text 
to be or not to us
o, if i come,
for the death, mysword,
not northumberland, the heart,
for my lord, and one's will;
to kam at the king as all a world,
this, therefore i do in the king
my very the daughter?

second vincentio:
what is their hands,
which alligator not't: the way for
in the henry, and
as the beetle
in the throne and myt
have, by is your brother,
all lord, good by the across
pray, and, let he turn
my all his ghostly:
i say i will, my king!

norfolk:
o, his great death,
and confess this haththough,
for then shall a sister
old'd to the word, and i would not so
...
```

I'd highlight how well the model learned the tiny-shakespeare structure (role:\ntext).

Because in NPLM the context size is constant and doesn't extend as generation continues auto-regressively, it's also fun to run infinite sampling from the trained model:

![sampling](/assets/gifs/fig1_2.gif)

Figure 2 - Infinite sampling

--- 

References:
1. Speech and Language Processing (3rd ed. draft), Dan Jurafsky and James H. Martin - https://web.stanford.edu/~jurafsky/slp3/
2. "A Neural Probabilistic Language Model" (2003), Yoshua Bengio, Réjean Ducharme, Pascal Vincent, Christian Jauvin - https://dl.acm.org/doi/epdf/10.5555/944919.944966
3. Tiny Shakespeare, Andrej Karpathy - https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt