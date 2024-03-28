import torch
from torch import nn
from waternet.basic_layers.res_blocks import MultiResBlock
from waternet.basic_layers.multiplication_block import MultiplicationBlock
from waternet.basic_layers.conv_blocks import ConvBlock
from waternet.basic_layers.fully_connected import MiddleFullyConnected

class DecoderLayer(nn.Module):
    """
    Increase the width and height by a factor of 2 using a transposed convolution, decrease the number of channels
     by a factor of 2. Takes a skip connection input from the corresponding encoder layer.
    """
    def __init__(self,
                 init_channels,
                 max_kernel_size=5,
                 padding_mode='reflect',
                 dtype=torch.bfloat16, device='cuda'
                 ):
        super().__init__()
        num_channels = init_channels
        self.conv_trans = nn.ConvTranspose2d(
                in_channels=2*num_channels, out_channels=num_channels,
                kernel_size=2, stride=2, device=device, dtype=dtype
        )
        self.normalization_layers0 = nn.InstanceNorm2d(
            num_features=num_channels, device=device, dtype=dtype,
            track_running_stats=False, affine=True
        )
        self.normalization_layers1 = nn.InstanceNorm2d(
            num_features=2 * num_channels, device=device, dtype=dtype,
        )
        self.normalization_layers2 = nn.InstanceNorm2d(
            num_features=2 * num_channels, device=device, dtype=dtype,
        )
        self.normalization_layers3 = nn.InstanceNorm2d(
            num_features=2 * num_channels, device=device, dtype=dtype,
            track_running_stats=False, affine=True
        )
        self.normalization_layers4 = nn.InstanceNorm2d(
            num_features=num_channels, device=device, dtype=dtype,
            track_running_stats=False, affine=True
        )
        self.multiplication_blocks = MultiplicationBlock(
            num_channels=2 * num_channels, num_conv_blocks=3,
            kernel_sizes=[min(7, max_kernel_size),
                          min(5, max_kernel_size),
                          min(3, max_kernel_size),
                          ],
            padding_mode=padding_mode, dtype=dtype, device=device
        )
        self.multi_res_blocks = MultiResBlock(
            num_channels=2 * num_channels, num_res_blocks=3, kernel_size=min(5, max_kernel_size),
            padding_mode=padding_mode, use_bias=False, dtype=dtype, device=device
        )
        self.middle_connected = MiddleFullyConnected(
            num_channels=4 * num_channels, out_channels=2 * num_channels,
            device=device, dtype=dtype, num_layers=3
        )
        self.conv_blocks = ConvBlock(
            in_channels=2 * num_channels, out_channels=num_channels,
            kernel_size=min(5, max_kernel_size), device=device, dtype=dtype
        )


    def forward(self, x, x_skip):
        x = self.conv_trans(x)
        x = self.normalization_layers0(x)
        x = torch.concat([x, x_skip], dim=1)
        x1 = self.multi_res_blocks(x)
        x1 = self.normalization_layers1(x1)
        x2 = self.multiplication_blocks(x)
        x2 = self.normalization_layers2(x2)
        x = torch.concat([x1, x2], dim=1)
        x = self.middle_connected(x)
        x = self.normalization_layers3(x)
        x = self.conv_blocks(x)
        x = self.normalization_layers4(x)
        return x
