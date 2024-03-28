import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """
    Two convolutions with a leaky relu inbetween
    """
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 kernel_size=3,
                 kernel_size2=None,
                 use_bias=False,
                 padding_mode='reflect',
                 dtype=torch.float32,
                 device='cuda'):
        super(ConvBlock, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = mid_channels
        if kernel_size2 is None:
            kernel_size2 = kernel_size
        self.conv_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1, bias=use_bias,
            padding=kernel_size//2, padding_mode=padding_mode, dilation=1, groups=1, dtype=dtype, device=device
        )
        
        self.conv_2 = nn.Conv2d(
            in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size2, stride=1, bias=use_bias,
            padding=kernel_size2//2, padding_mode=padding_mode, dilation=1, groups=1, dtype=dtype, device=device
        )
    
    def forward(self, x):
        x = self.conv_1(x)
        x = F.leaky_relu(x)
        x = self.conv_2(x)
        return x


class MultiConvBlock(nn.Module):
    """
    n - ConvBlocks, with a leaky relu between each (except for the final)
    """
    def __init__(self,
                 num_channels,
                 num_conv_blocks=3,
                 kernel_sizes: int or list=3,
                 padding_mode='reflect',
                 use_bias=False,
                 dtype=torch.float32, device='cuda'):
        super().__init__()
        self.num_conv_blocks = num_conv_blocks
        if type(kernel_sizes) == int:
            kernel_sizes = [kernel_sizes for _ in range(self.num_conv_blocks)]
        self.conv_blocks = nn.ModuleList([
            ConvBlock(
                        in_channels=num_channels, use_bias=use_bias, kernel_size=kernel_sizes[i],
                        padding_mode=padding_mode, dtype=dtype, device=device
                ) for i in range(num_conv_blocks)
        ])
    
    def forward(self, x):
        for ind, conv_block in enumerate(self.conv_blocks):
            x = conv_block(x)
            if ind < self.num_conv_blocks - 1:
                x = F.leaky_relu(x)
        return x
