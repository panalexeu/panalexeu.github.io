title: Reproducing "Backpropagation Applied to Handwritten Zip Code Recognition" (1989) [RETRO]
author: oleksii horchynskyi

---

I continue my series of reproduction blogposts, and this time my interest fell on convolutional neural networks. In fact, this is my first time actually reproducing a convolutional neural network. I decided to start with the 1989 paper "Backpropagation Applied to Handwritten Zip Code Recognition" by Y. LeCun et al. [1] because of its significance, and as Karpathy states, "...it is, to my knowledge, the earliest real-world application of a neural net trained end-to-end with backpropagation" [2]. Andrej Karpathy also reproduced the paper in 2022, so I used his implementation and blogpost as a cross-reference, which helps clarify some of the uncertainties in the paper (some points are not elaborated on, and the paper seems to be missing some characters).

You can find the diagram of the architecture presented in the paper below (fig. 1).

![cnn_pic](/assets/images/fig2_1.png)

Figure 1 - Network architecture taken from [1]

The network consists of 3 layers. The first layer H1 consists of 12 separate learned convolutional kernels of size 5x5 that produce 8x8 feature maps from 16x16 greyscale images. The second layer H2 also consists of 12 separate learned convolutional kernels of size 5x5 that produce 4x4 feature maps, though with a connection scheme that is a bit tricky to wrap your head around (at least it was for me). The H2 kernels are fed the 8x8 feature maps previously extracted by H1, and each output unit of H2 is a summation of local information extracted by the H2 kernels across the H1 feature maps. The H3 layer is a linear layer that takes the flattened H2 feature maps as input (12x4x4=192) and compresses them into 30 features. Then finally an output linear layer does 10-digit classification.

The dataset used in the paper consists of handwritten zip codes that appeared on U.S. mail. The zip codes were preprocessed by contractors who extracted the individual digits, and those digits were then downscaled from 40x60 into 16x16 greyscale images using some form of linear transformation that the paper does not specify. Note that the use of greyscale images allows more information to be preserved, since one pixel encodes a range of grey levels rather than a binary value. The greyscale images are normalized to the range [-1;1]. In my reproduction I use the [flwrlabs/usps](https://huggingface.co/datasets/flwrlabs/usps) dataset, which seems to be an exact reproduction of the dataset used in the paper.

Now let's discuss implementation details. Like Karpathy, I implemented the network in PyTorch, and I would like to highlight several features of my implementation:

1. The original implementation combines information in H2 from only 8 of the extracted feature maps from H1, but it does not specify how this H1 feature map selection works: "The eight maps in H1 on which a map in H2 takes its inputs are chosen according to a scheme that will not be described here". I skip this and combine information from all 12 feature maps of H1 in H2, though Karpathy has his own [interpretation](https://github.com/karpathy/lecun1989-repro/blob/master/repro.py#L78-L80) of how this might have worked.
2. The paper does not explicitly mention what kind of nonlinearity is used and where, so I just copied what Karpathy did and used a tanh nonlinearity after each layer (after the output projection too).
3. The network uses MSE for loss calculation. Since the output projection also passes through tanh, and we expect the network to assign the highest value to the correct target, and since tanh squashes values between -1 and 1, the target vector is encoded as -1s with a 1 for the correct label.
4. I use Karpathy's assumption that the weight initialization is 2.4 / sqrt(F_in), though the paper does not explicitly mention that a square root was used — potentially some symbols are missing.
5. Convolution biases are defined separately as parameters, since Conv2d in torch has only one bias per output map by default.
6. SGD is used as the optimizer, with the learning rate set to 0.03 (as in Karpathy's implementation). In the paper, though, some version of Newton's algorithm was used. The weights of the network are updated after each individual example, as specified in the paper: "The weights were updated according to the so-called stochastic gradient or 'on-line' procedure (updating after each presentation of a single pattern)".

Below is the code of the resulting PyTorch implementation:

```python
from typing import Self

import torch
import torch.nn as nn 

class Model(nn.Module): 
    def __init__(
        self, 
        h1_size: int = 12,
        h2_size: int = 12, 
        h3_size: int = 30,
    ): 
        super().__init__()

        self.init_factor = 2.4

        self.h1 = nn.Conv2d(
            in_channels=1, 
            out_channels=h1_size,
            kernel_size=5,
            stride=2, 
            bias=False 
        ) 
        h1_in = 5 ** 2 # 25
        torch.nn.init.uniform_(self.h1.weight, a=-self.init_factor / h1_in ** 0.5, b=self.init_factor / h1_in ** 0.5)
        # quoting the paper: Thus layer H1  comprises ... but only 
        # 1068 free parameters (768 biases plus 25 times 12 feature kernels)
        self.h1_bias = nn.Parameter(torch.zeros(h1_size, 8, 8))

        # for now this part is ignored and information is combined for all 12 feature 
        # maps from level H1: Each unit in H2 combines local information coming from 8
        #  of the 12 different feature maps in H1. 
        self.h2 = nn.Conv2d(
            in_channels=h2_size, 
            out_channels=h2_size, 
            kernel_size=5, 
            stride=2, 
            bias=False
        ) 
        h2_in = h1_size * h1_in # 12 * 5 * 5 = 300
        torch.nn.init.uniform_(self.h2.weight, a=-self.init_factor / h2_in ** 0.5, b=self.init_factor / h2_in ** 0.5)
        # quoting the paper: All these connections are controlled by only
        # 2592 free parameters (12 feature maps times 200 weights plus 192 biases). 
        self.h2_bias = nn.Parameter(torch.zeros(h2_size, 4, 4))

        h3_in = h2_size * 16 # 192
        self.h3 = nn.Linear(
            in_features=h3_in,  
            out_features=h3_size,
            bias=True
        )
        torch.nn.init.uniform_(self.h3.weight, a=-self.init_factor / h3_in ** 0.5, b=self.init_factor / h3_in ** 0.5)

        self.out = nn.Linear(
            in_features=h3_size, # 30
            out_features=10,
            bias=True
        ) 
        torch.nn.init.uniform_(self.out.weight, a=-self.init_factor / h3_size ** 0.5, b=self.init_factor / h3_size ** 0.5)

    def __call__(self, x: torch.Tensor, target: torch.Tensor | None = None): 
        # quoting the paper: Connections extending past the boundaries of the
        # input plane take their input from a virtual background plane whose state
        # is equal to a constant, pretedetermined background level, in our case -1.
        x = nn.functional.pad(x, (2, 2, 2, 2), mode='constant', value=-1)
        x = self.h1(x) + self.h1_bias
        x = nn.functional.tanh(x)
        x = nn.functional.pad(x, (2, 2, 2, 2), mode='constant', value=-1)
        x = self.h2(x) + self.h2_bias 
        x = nn.functional.tanh(x)
        x = x.reshape(-1)
        x = self.h3(x)
        x = nn.functional.tanh(x)
        x = self.out(x) 
        x = nn.functional.tanh(x)

        loss = None 
        if target is not None: 
            loss = nn.functional.mse_loss(x, target)

        return x, loss 

    def configure_optimizer(self, lr=0.03): 
        return torch.optim.SGD(self.parameters(), lr=lr)

    @classmethod
    def from_pretrained(cls, ckpt_path: str = './ckpt.pt') -> Self: 
        ckpt = torch.load(ckpt_path)
        model = cls()
        model.load_state_dict(ckpt)
        return model 
```

The paper reports the following results:

```
eval: split train. loss 2.5e-3. error 0.14%. misses: 10
eval: split test . loss 1.8e-2. error 5.00%. misses: 102
```

My reproduction achieves:

```
eval: split train. loss 2e-3 error 0.00% misses: 0
eval: split test . loss 3.7e-2 error 6.00% misses: 120
```

Karpathy's implementation achieves:

```
eval: split train. loss 4.073383e-03. error 0.62%. misses: 45
eval: split test . loss 2.838382e-02. error 4.09%. misses: 82
```

Though Karpathy's numbers are probably not really comparable, since he used a sampled version of MNIST.

It is also possible to infer from the results that using all 12 feature maps in H2 makes the network overfit the training data more.

As an experiment on truly OOD data, and on images hand-drawn by me, I also repeated the paper's data preprocessing pipeline by first drawing binarized digits in Aseprite on a 400x60 spritesheet, then downscaling the spritesheet to 160x16 in a greyscale colour palette, and classifying them with the neural network. You can see them in figures 2 and 3.

![numbers_40x60](/assets/images/numbers_40x60.png)

Figure 2 - Drawn numbers in a 400x60 Aseprite spritesheet

![numbers_16x16](/assets/images/numbers_16x16.png)

Figure 3 - Downscaled spritesheet, 160x16

Surprisingly, the network correctly classifies only 4 digits out of 10, specifically 2, 4, 5, and 6. I am not sure how to interpret these results.

---

References:
1. Backpropagation Applied to Handwritten Zip Code Recognition, Y. LeCun et al. - http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf
2. Deep Neural Nets: 33 years ago and 33 years from now, Andrej Karpathy - https://karpathy.github.io/2022/03/14/lecun1989/
