import torch
from torch import nn
from waternet.model_layers.global_attention import GlobalAttention
from waternet.model_layers.final_fully_connected import FullyConnected
from waternet.model_layers.encoder import EncoderLayer
from waternet.model_layers.decoder import DecoderLayer


class WaterwayModel(nn.Module):
    def __init__(self,
                 init_channels=10,
                 num_encoders=5,
                 num_decoders=3,
                 num_channels=16,
                 num_outputs=1,
                 padding_mode='reflect',
                 dtype=torch.bfloat16, device='cuda',
                 track_running_stats=False, affine=True):
        super().__init__()
        self.attention = GlobalAttention(
            in_channels=init_channels, out_channels=num_channels, padding_mode=padding_mode,
            device=device, dtype=dtype
        )
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(num_encoders):
            mks = 2 * (7 - i) + 1
            self.encoders.append(
                EncoderLayer(
                    num_channels=num_channels, padding_mode=padding_mode,
                    max_kernel_size=mks, device=device, dtype=dtype,
                    track_running_stats=track_running_stats, affine=affine
                )
            )
            num_channels *= 2

        for i in range(num_decoders):
            num_channels = num_channels // 2
            mks = 2 * (i + 1) + 1
            self.decoders.append(
                DecoderLayer(
                    init_channels=num_channels, max_kernel_size=mks, device=device, dtype=dtype,
                )
            )

        self.fully_connected = FullyConnected(
            num_channels=num_channels, num_layers=4, num_outputs=num_outputs, device=device, dtype=dtype
        )

    def forward(self, x):
        x = self.attention(x)
        encoder_out = []
        for encoder in self.encoders:
            encoder_out.append(x)
            x = encoder(x)
        for i, decoder in enumerate(self.decoders):
            xo = encoder_out[-1 - i]
            x = decoder(x, xo)
        x = self.fully_connected(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary
    device = 'cuda'
    dtype = torch.bfloat16
    num_channels = 10
    side_len = 832
    class_layer = WaterwayModel(
        init_channels=num_channels, num_encoders=5, num_channels=16,
        num_decoders=5, dtype=dtype, device=device
    )
    print(
        summary(
            model=class_layer, input_size=(1, num_channels, side_len, side_len), depth=10,
            device=device, dtypes=[dtype], col_names=['kernel_size', 'input_size', 'output_size', 'num_params'],
        )
    )
