# Overview

See paper.

This model is based on ideas from UNet (https://arxiv.org/abs/1505.04597) and ResNet (https://arxiv.org/pdf/1512.03385.pdf)
among others (we use instance normalizations, layers similar to GLUs, and additional skip connections).

One of the more unique ideas in our model is that we don't complete the UNet (we have not seen this applied elsewhere,
but would not be surprised if it has been used). That is to say, we use 5 encoders (decreasing the width and height by a factor of two at each iteration),
and we only use 3 or 4 decoders (increasing the width and height by a factor of 2), so our final outputs 
are 2 or 4 times more coarse than the input layers.

There were various reasons to make this decision, one being that the training data, while very good, does not perfectly
align with the training images, so having a coarser resolution allows more wiggle room there. From a data perspective,
having data with 2 times less resolution requires roughly 4 times less storage, and 4 times less resolution requires 16 times less storage,
so there are benefits there as well. Lastly, the applications that we had in mind while making this model, namely 
identifying where waterways should be taken into account for infrastructure, don't require us to be super precises,
as we expect a human to actually investigate further, with this dataset as a guide of where you may have to look.


# Installation
Installation requires pytorch.
You may also want to install [WaterNet Training and Evaluation](https://github.com/Better-Planet-Laboratory/WaterNet_training_and_evaluation) and
[WaterNet Vectorize](https://github.com/Better-Planet-Laboratory/WaterNet_vectorize).

All code was prototyped using python 3.11.4 and pip 23.0.1.

A python environment and version handler such as [pyenv](https://github.com/pyenv/pyenv) should make those easy to obtain.

After getting your environment setup correctly, download this repository and use pip to install:

```
git clone https://github.com/Better-Planet-Laboratory/WaterNet.git
cd WaterNet
pip install .
```

or if you wish to edit to code

``pip install -e .``


# Usage Notes


* [waternet.model](src/waternet/model.py) contains the class for the waterway model.
* [waternet.model_layers](src/waternet/model_layers) contains all the layers used directly in
the waterway model.
* [waternet.basic_layers](src/waternet/basic_layers) has layers used to make the waternet.model_layers layers.

To make a new model:
```
from waternet.model import WaterNet

model = WaterNet(
    init_channels=10,
    num_encoders=5,
    num_decoders=3,
    num_channels=16,
    num_outputs=1,
    dtype=torch.bfloat16,
    device='cuda',
)
```
where
* init_channels is the number of input channels, in our case we had 10, which is the default value.
* num_encoders is the number of encoder layers to use. Each encoder decreases the height and width by a factor of 2, so for 5 encoders, your inputs must be at least (2^5, 2^5).
* num_decoders is the number of decoders to use. Each decoder increases the height and width by a factor of 2. So the final grid size will be (rows*2^(num_decoders-num_encoders), cols*2^(num_decoders-num_encoders)).
* num_channels is the number of channels after the initial global attention layer is preformed.
* num_outputs is the number of output channels


# Related Repositories

 * [WaterNet Training and Evaluation](https://github.com/Better-Planet-Laboratory/WaterNet_training_and_evaluation)
 * [WaterNet Vectorize](https://github.com/Better-Planet-Laboratory/WaterNet_vectorize)

