import torch
from torch import nn
from torch.nn import functional as F

"""
A fully connected layer that occurs in both the Encoder and Decoder layers (Middle is used to differentiate between this
layer and the fully connected layer used for the output
"""

class MiddleFullyConnected(nn.Module):
    def __init__(self,
                 num_channels,
                 out_channels,
                 num_layers=3,
                 decrease_percent=.25,
                 padding_mode='reflect',
                 dtype=torch.bfloat16,
                 device='cuda'
                 ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers - 1):
            middle_channels = max(int(num_channels * (1 - decrease_percent)), 1)
            self.layers.append(
                nn.Conv2d(
                    in_channels=num_channels, out_channels=middle_channels,
                    kernel_size=1, stride=1, padding=0, padding_mode=padding_mode,
                    dilation=1, bias=True, groups=1, dtype=dtype, device=device
                )
            )
            num_channels = middle_channels
        self.layers.append(
            nn.Conv2d(
                in_channels=num_channels, out_channels=out_channels,
                kernel_size=1, stride=1, padding=0, padding_mode=padding_mode,
                dilation=1, bias=True, groups=1, dtype=dtype, device=device
            )
        )

    def forward(self, x):
        for ind, layer in enumerate(self.layers):
            x = layer(x)
            if ind < self.num_layers - 1:
                x = F.leaky_relu(x)
        return x
