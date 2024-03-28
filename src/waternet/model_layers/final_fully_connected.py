import torch
from torch import nn
from torch.nn import functional as F


class FullyConnected(nn.Module):
    """
    The last layer in the network. Fully connected, applying sigmoid to the final ouput.
    """
    def __init__(self,
                 num_channels: int,
                 num_layers: int = 3,
                 num_outputs: int=1,
                 padding_mode: str = 'reflect',
                 dtype: torch.dtype = torch.bfloat16,
                 device: str = 'cuda'
                 ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers - 1):
            out_channels = 2*num_channels
            self.layers.append(
                nn.Conv2d(
                    in_channels=num_channels, out_channels=out_channels,
                    kernel_size=1, stride=1, padding=0, padding_mode=padding_mode,
                    dilation=1, bias=True, groups=1, dtype=dtype, device=device
                )
            )
            num_channels = out_channels
        self.layers.append(
            nn.Conv2d(
                in_channels=num_channels, out_channels=num_outputs, kernel_size=1,
                stride=1, padding=0, padding_mode=padding_mode, dilation=1,
                bias=False, groups=1, dtype=dtype, device=device
            )
        )

    def forward(self, x):
        for ind, layer in enumerate(self.layers):
            x = layer(x)
            if ind < self.num_layers - 1:
                x = F.leaky_relu(x)
            else:
                x = F.sigmoid(x)
        return x
