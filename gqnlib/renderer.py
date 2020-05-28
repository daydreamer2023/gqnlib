
"""Prior, Posterior, and Renderer for Consistent GQN."""

from typing import Tuple

import torch
from torch import nn, Tensor

from .generation import Conv2dLSTMCell


class LatentDistribution(nn.Module):
    """Latent Distribution p(z|r).

    * prior: p(z|r_c)
    * posterior: p(z|r_c, r_t)

    Args:
        r_channel (int): Number of channels in representation.
        e_channel (int): Number of channels in the conv. layer mapping input
            images to LSTM input (nf_enc).
        h_channel (int): Number of channels in LSTM layer (nf_to_hidden).
        z_channel (int): Number of channels in the stochastic latent in each
            DRAW step (nf_z).
    """

    def __init__(self, r_channel: int, e_channel: int, h_channel: int,
                 z_channel: int, stride: int):
        super().__init__()

        self.conv1 = nn.Conv2d(r_channel, e_channel, kernel_size=stride,
                               stride=stride)
        self.lstm_cell = Conv2dLSTMCell(e_channel + z_channel, h_channel,
                                        kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(h_channel, z_channel * 2, kernel_size=5,
                               stride=1, padding=2)

    def forward(self, r: Tensor, z: Tensor, h: Tensor, c: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Converts r -> z.

        Args:
            r (torch.Tensor): Representations, size `(b, r, 16, 16)`.
            z (torch.Tensor): Previous latents, size `(b, z, 8, 8)`.
            h (torch.Tensor): Previous hidden states, size `(b, h, 8, 8)`.
            c (torch.Tensor): Previous cell states, size `(b, h, 8, 8)`.

        Returns:
            h (torch.Tensor): Current hidden states, size `(b, h, 8, 8)`.
            c (torch.Tensor): Current cell states, size `(b, h, 8, 8)`.
            mu (torch.Tensor): Mean of `z` distribution, size `(b, z, 8, 8)`.
            logvar (torch.Tensor): Log variance of `z` distribution, size
                `(b, z, 8, 8)`.
        """

        lstm_input = self.conv1(r)
        h, c = self.lstm_cell(torch.cat([lstm_input, z], dim=1), (h, c))
        mu, logvar = torch.chunk(self.conv2(h), 2, dim=1)

        return h, c, mu, logvar
