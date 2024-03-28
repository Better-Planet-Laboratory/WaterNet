import torch
from torch import nn
from torch.nn import functional as F
from waternet.basic_layers.conv_blocks import MultiConvBlock


class MultiplicationBlock(nn.Module):
    """
    Much like a gated linear unit (GLU), applies a MultiConvBlock to the input, applies sigmoid to that output,
     and then does pixelwise multiplication with the original (or other input)
    """
    def __init__(self,
                 num_channels,
                 num_conv_blocks=3,
                 kernel_sizes:int or list=3,
                 padding_mode='reflect',
                 dtype=torch.float32,
                 device='cuda'):
        super().__init__()
        self.multi_conv_block = MultiConvBlock(
            num_channels=num_channels, num_conv_blocks=num_conv_blocks, kernel_sizes=kernel_sizes,
            padding_mode=padding_mode, use_bias=False, dtype=dtype, device=device
        )
    
    def forward(self, x, x_other=None):
        if x_other is None:
            x_other = x
        x = self.multi_conv_block(x)
        x = F.sigmoid(x)
        x = x*x_other
        return x
