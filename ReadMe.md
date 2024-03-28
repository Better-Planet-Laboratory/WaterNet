This model is based on ideas from UNet (https://arxiv.org/abs/1505.04597) and ResNet (https://arxiv.org/pdf/1512.03385.pdf)
among others (we use instance normalizations, layers similar to GLUs, and additional skip connections).

One of the more unique ideas in our model is that we don't complete the UNet (we have not seen this applied elsewhere,
but would not be surprised if it has been used). That is to say, we use 5 encoders (decreasing the width and height by a factor of two at each iteration),
and we only use 3 or 4 decoders (increasing the width and height by a factor of 2), so our final outputs 
are 2 or 4 times more coarse than the input layers.

There where various reasons to make this decision, one being that the training data, while very good, does not perfectly
align with the training images, so having a coarser resolution allows more wiggle room there. From a data perspective,
having data with 2 times less resolution requires roughly 4 times less storage, and 4 times less resolution requires 16 times less storage,
so there are benefits there as well. Lastly, the applications that we had in mind while making this model, namely 
identifying where waterways should be taken into account for infrastructure, don't require us to be super precises,
as we expect a human to actually investigate further, with this dataset as a guide of where you may have to look.



waternet.model contains the class for the waterway model. waternet.model_layers contains all the layers used directly in
the waterway model. waternet.basic_layers has layers used to make the waternet.model_layers layers.
