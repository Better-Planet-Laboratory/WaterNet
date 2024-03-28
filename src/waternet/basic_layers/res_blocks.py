import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self,
                 num_channels,
                 kernel_size=3,
                 use_bias=False,
                 padding_mode='reflect',
                 dtype=torch.bfloat16,
                 device='cuda'):
        super(ResBlock, self).__init__()
        self.conv_1 = nn.Conv2d(
                in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size,
                stride=1, padding=kernel_size//2, padding_mode=padding_mode,
                dilation=1, bias=use_bias, groups=1, dtype=dtype, device=device
        )
        
        self.conv_2 = nn.Conv2d(
                in_channels=num_channels, out_channels=num_channels,
                kernel_size=kernel_size, stride=1, padding=kernel_size//2,
                padding_mode=padding_mode, dilation=1, bias=use_bias,
                groups=1, dtype=dtype, device=device
        )
        self.norm_2 = nn.InstanceNorm2d(
                num_features=num_channels, device=device, dtype=dtype,
                affine=True, track_running_stats=False
        )
        # self.norm_2.bias.data = self.norm_2.bias.data + 1.5
    
    def forward(self, x):
        x1 = self.conv_1(x)
        x1 = F.leaky_relu(x1)
        x1 = self.conv_2(x1)
        x1 = x1 + x
        x1 = self.norm_2(x1)
        return x1


class MultiResBlock(nn.Module):
    def __init__(self,
                 num_channels,
                 num_res_blocks=3,
                 kernel_size=3,
                 use_bias=False,
                 padding_mode='reflect',
                 dtype=torch.bfloat16, device='cuda'):
        super().__init__()
        self.num_blocks = num_res_blocks
        self.res_blocks = nn.ModuleList(
                [ResBlock(
                        num_channels=num_channels, use_bias=use_bias, kernel_size=kernel_size,
                        padding_mode=padding_mode, dtype=dtype, device=device
                ) for _ in range(num_res_blocks)]
        )
    
    def forward(self, x):
        for i, rb in enumerate(self.res_blocks):
            x = rb(x)
            # if i < self.num_blocks - 1:
            #     x = F.leaky_relu(x)
        return x