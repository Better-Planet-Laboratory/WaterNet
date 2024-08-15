import torch
from torch import nn
from ..basic_layers.res_blocks import MultiResBlock
from ..basic_layers.multiplication_block import MultiplicationBlock
from ..basic_layers.fully_connected import MiddleFullyConnected

class EncoderLayer(nn.Module):
    """
    Decreases the width and height by a factor of 2 with the initial convolution,
     increases the number of channels by a factor of 2.

    """
    def __init__(self,
                 num_channels=5,
                 max_kernel_size=7,
                 padding_mode='reflect',
                 dtype=torch.bfloat16, device='cuda',
                 track_running_stats=False, affine=True):
        super().__init__()
        self.decrease_conv = nn.Conv2d(
            in_channels=num_channels, out_channels=num_channels, kernel_size=2,
            padding=0, stride=2, dtype=dtype, device=device
        )
        self.normalization_layers1 = nn.InstanceNorm2d(
            num_features=num_channels, device=device, dtype=dtype,
            track_running_stats=track_running_stats, affine=affine
        )
        self.normalization_layers2 = nn.InstanceNorm2d(
            num_features=num_channels, device=device, dtype=dtype,
            track_running_stats=track_running_stats, affine=affine
        )
        self.normalization_layers3 = nn.InstanceNorm2d(
            num_features=num_channels, device=device, dtype=dtype,
            track_running_stats=track_running_stats, affine=affine
        )
        self.multiplication_blocks = MultiplicationBlock(
            num_channels=num_channels, num_conv_blocks=3,
            kernel_sizes=[
                min(7, max_kernel_size), min(5, max_kernel_size), min(3, max_kernel_size),
            ],
            padding_mode=padding_mode, dtype=dtype, device=device
        )
        self.multi_res_blocks = MultiResBlock(
            num_channels=num_channels, num_res_blocks=3, kernel_size=min(5, max_kernel_size),
            padding_mode=padding_mode, use_bias=False, dtype=dtype, device=device
        )
        self.middle_connected = MiddleFullyConnected(
            num_channels=3 * num_channels, out_channels=2 * num_channels,
            device=device, dtype=dtype, num_layers=3
        )
        num_channels += num_channels

    def forward(self, x):
        x = self.decrease_conv(x)
        x = self.normalization_layers1(x)
        x1 = self.multi_res_blocks(x)
        x1 = self.normalization_layers2(x1)
        x2 = self.multiplication_blocks(x)
        x2 = self.normalization_layers3(x2)
        x = torch.cat([x, x1, x2], dim=1)
        x = self.middle_connected(x)
        return x
