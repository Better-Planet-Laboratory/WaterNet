# Overview

This repository is associated with the forthcoming paper "Pierson, Matthew., and Mehrabi, Zia. 2024. Deep learning waterways for rural infrastructure development. arXiv." Please cite this paper and attribute the work if using the model or work.

The data outputs of this model (raster and vectorized versions) are stored and available under CC-BY-SA from the following source: Pierson, Matthew., Mehrabi. Zia. 2024, WaterNet Outputs and Code, https://doi.org/10.7910/DVN/YY2XMG, Harvard Dataverse.

This model is based on ideas from UNet (https://arxiv.org/abs/1505.04597) and ResNet (https://arxiv.org/pdf/1512.03385.pdf)
among others (we use instance normalizations, layers similar to GLUs, and additional skip connections). One of the unique aspects of this model is that we don't complete the UNet. That is to say, we use 5 encoders (decreasing the width and height of each image by a factor of two at each iteration), and we only use  4 decoders, optimizing storage while maintaining precision of raster outputs that are 20m globally.  These rasters are then vectorized by first connecting our waterways to the TDX-Hydro waterways using least cost pathing to connect disconnected segments, on top of which we employ a thinning and vectorization algorithm.

Notably, the model was trained across a diversity of hydrographic conditions using labels from the National Hydrography Dataset  (e.g. with an identifier for each water type, such as rivers, streams, lakes, ditches, intermittent, ephemeral, called the fcode),  in two steps, starting  with a larger training set of smaller context $1.5M grids (244 x 244 pixels),  and followed with a 10x decrease in training samples but  10x increase in context, $90K grids (832 x 832 pixels). We have found this two step approach to be a useful for making location predictions across a diversity of contexts and water way types, while at the same time minimizing evaluation time and maximizing speed and alignment of waterways network structures in the final product. We use a summed Binary Cross Entropy and Tanimoto loss weighted by waterway type. We effectively mask swamps, canals, intermittent lakes, ditches, and playas in training, with rivers and streams (intermittent, ephemeral and perennial) alongside perennial and perminant lakes, being our primary target (although we evaluate the model performance on all waterway types). Our input features includes 10 channels: the first four being transformed Sentinel-2 NRGB channels ($NRGB_t$), and the remaining 7 being $NDVI$, $NDWI$, Shifted Elevation ($E_S$), Elevation x-delta ($\Delta_x E$), Elevation y-delta ($\Delta_y E$), elevation gradient ($\nabla E$).


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
