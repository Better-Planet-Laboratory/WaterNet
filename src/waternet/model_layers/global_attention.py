import torch
from torch import nn
from torch.nn import functional as F
from waternet.basic_layers.conv_blocks import ConvBlock

"""
The Attention mechanism uses channelwise means and standard deviations, similar to SRM in https://arxiv.org/abs/2111.07624
"""

class GlobalAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 padding_mode='reflect',
                 dtype=torch.bfloat16,
                 device='cuda'):
        super().__init__()
        self.conv_block1 = ConvBlock(
            in_channels=in_channels, mid_channels=25, out_channels=50, kernel_size=1,
            kernel_size2=1, padding_mode=padding_mode, device=device, dtype=dtype
        )
        self.conv_block2 = ConvBlock(
            in_channels=in_channels + 100, mid_channels=in_channels + 50, out_channels=in_channels,
            kernel_size=1, kernel_size2=1, padding_mode=padding_mode, device=device, dtype=dtype
        )
        self.normalization = nn.InstanceNorm2d(
            num_features=in_channels, device=device, dtype=dtype,
            track_running_stats=False, affine=False
        )
        self.channel_increase_conv = ConvBlock(
            in_channels=in_channels, mid_channels=max(out_channels//2, in_channels), out_channels=out_channels,
            kernel_size=3, device=device, dtype=dtype
        )
        self.device = device
        self.dtype = dtype
    
    def forward(self, x, x_out=None):
        if x_out is None:
            x_out = x
        x1 = self.conv_block1(x)
        x2 = x1.mean(dim=(2, 3), keepdims=True).repeat(1, 1, x.shape[-2], x.shape[-1])
        x3 = x1.std(dim=(2, 3), keepdims=True).repeat(1, 1, x.shape[-2], x.shape[-1])
        
        x1 = torch.concat([x, x2, x3], dim=1)
        x1 = F.sigmoid(self.conv_block2(x1))
        x = x1*x_out
        x = self.normalization(x)
        x = self.channel_increase_conv(x)
        return x
