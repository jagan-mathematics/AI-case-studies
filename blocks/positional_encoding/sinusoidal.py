import torch
from torch import nn

from model_configs.base import ModelConfigs


class PositionalEncoding(nn.Module):
    """
    Positional Encoding comprises sinusoidal waves which works based on the wavelength
    and frequency of the rotation across dimension.

    P_E(pos,2i) = sin(pos/10000^(2i/dmodel))
    P_E(pos,2i+1) = cos(pos/10000^(2i/dmodel))

    L => Seq_len
    D => dim
    B => Batch_size
    """

    def __init__(self, config: ModelConfigs):
        super(PositionalEncoding, self).__init__()

        depth = config.d_model / 2
        pe = torch.zeros(config.model_max_sequence, config.d_model)  # (L x D)
        position = torch.arange(0, config.model_max_sequence, dtype=torch.float).unsqueeze(1)  # (L x 1)
        depths = torch.arange(0, depth).unsqueeze(0) / depth  # (1 x D // 2)
        angle_rate = 1 / (10000 ** depths)  # angle rate is an monotonically increasing function (1 x D // 2)
        angle_rads = position * angle_rate  # (L x D // 2)

        pe[:, 0::2] = torch.sin(angle_rads)  # replace every position with sin wave
        pe[:, 1::2] = torch.cos(angle_rads)  # replace every position with cos wave
        self.register_buffer('pe', pe.unsqueeze(0))  # saved as buffer with dim => (1 x L x D)

    def forward(self, x):
        """model forward pass"""
        return x + self.pe[:, :x.size(1)]
