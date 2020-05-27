
"""Representation network."""

import torch
from torch import nn, Tensor


class Pyramid(nn.Module):
    """Pyramid.

    Args:
        n_channel (int, optional): Number of channel of images.
        n_target (int, optional): Dimension of viewpoints.
    """

    def __init__(self, n_channel: int = 3, n_target: int = 7):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_channel + n_target, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=8, stride=8),
            nn.ReLU(),
        )

    def forward(self, x: Tensor, v: Tensor) -> Tensor:
        """Represents r given images `x` and viewpoints `v`.

        Args:
            x (torch.Tensor): Images tensor, size `(batch, c, h, w)`.
            v (torch.Tensor): Viewpoints tensor, size `(batch, t)`.

        Returns:
            r (torch.Tensor): Representation tensor, size `(batch, 256, 1, 1)`.
        """

        v = v.view(-1, 7, 1, 1).repeat(1, 1, 64, 64)
        r = self.conv(torch.cat([x, v], dim=1))

        return r
