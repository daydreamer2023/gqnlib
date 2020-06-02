
"""Attention Layer for Attetion GQN.

Ref)
D. Rosenbaum et al., "Learning models for visual 3D localization with implicit
mapping", http://arxiv.org/abs/1807.03149

S. Reed et al., "Few-shot Autoregressive Density Estimation: Towards Learning
to Learn Distributions", https://arxiv.org/abs/1710.10304
"""

from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DictionaryEncoder(nn.Module):
    """Dictionary encoder for representation (look-up table of attention).

    Args:
        x_channel (int, optional): Number of channels for images.
        v_dim (int, optional): Dimension size of viewpoints.
    """

    def __init__(self, x_channel: int = 3, v_dim: int = 7):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(x_channel, 32, kernel_size=2, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=1),
        )

    def forward(self, x: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward to calculate (key-value) pairs.

        1. Extract all image patches
        2. For each patch, calculate (key, value) pair.

        * keys: encoded images by conv. net.
        * values: concatenation of encoded images, viewpoints, position in
            image, and keys.

        Args:
            x (torch.Tensor): Images, size `(b, c, h, w)`.
            v (torch.Tensor): Viewpoints, size `(b, v)`.

        Returns:
            keys (torch.Tensor): Dict keys, size `(b*l, 64, 8, 8)`.
            values (torch.Tensor): Dict values, size `(b*l, c+v+2+64, 8, 8)`.
        """

        # Resize images: (64, 64) -> (32, 32)
        x = F.interpolate(x, (32, 32))
        _, c, *_ = x.size()

        # Extract 3x8x8 patches with an overlap of 4 pixels
        x = F.unfold(x, kernel_size=8, stride=4)
        x = x.permute(0, 2, 1)

        # Reshape: (b*l, c, 8, 8)
        b, l, _ = x.size()
        x = x.contiguous().view(b * l, c, 8, 8)

        # Keys: (b*l, 64, 8, 8)
        keys = self.conv(x)

        # Positions of each patch
        pos_x = torch.arange(7).repeat(7).float()
        pos_y = torch.arange(7).repeat_interleave(7).float()

        pos_x = pos_x.view(-1, 1, 1, 1).repeat(b, 1, 8, 8)
        pos_y = pos_y.view(-1, 1, 1, 1).repeat(b, 1, 8, 8)

        # Expand given viewpoints
        v = v.view(b, -1, 1, 1).repeat(l, 1, 8, 8)

        # Values: (b*l, x_channel + v_dim + 2 + key_channel, 8, 8)
        values = torch.cat([x, v, pos_x, pos_y, keys], dim=1)

        return keys, values
