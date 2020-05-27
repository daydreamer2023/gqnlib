
"""Generative Query Network."""

from typing import Dict, Tuple

import torch
from torch import nn, Tensor

from .generation import ConvolutionalDRAW, nll_normal
from .representation import Tower


class GenerativeQueryNetwork(nn.Module):
    """Generative Query Network class.

    Args:
        x_channel (int, optional): Number of channel in input images.
        v_dim (int, optional): Dimensions of viewpoints.
        r_dim (int, optional): Dimensions of representations.
        z_channel (int, optional): Number of channel in latent variable.
        h_channel (int, optional): Number of channel in hidden states.
        n_layer (int, optional): Number of recurrent layers.
    """

    def __init__(self, x_channel: int = 3, v_dim: int = 7, r_dim: int = 256,
                 z_channel: int = 64, h_channel: int = 128, n_layer: int = 8):
        super().__init__()

        self.r_dim = r_dim
        self.generator = ConvolutionalDRAW(x_channel, v_dim, r_dim, z_channel,
                                           h_channel, n_layer)
        self.representation = Tower(x_channel, v_dim)

    def forward(self, x_c: Tensor, v_c: Tensor, x_q: Tensor, v_q: Tensor
                ) -> Tensor:
        """Reconstructs queried images.

        Args:
            x_c (torch.Tensor): Context images, size `(b, m, c, h, w)`.
            v_c (torch.Tensor): Context viewpoints, size `(b, k)`.
            x_q (torch.Tensor): Query images, size `(b, m, c, h, w)`.
            v_q (torch.Tensor): Query viewpoints, size `(b, k)`.

        Returns:
            canvas (torch.Tensor): Reconstructed images, size `(b, c, h, w)`.
        """

        canvas, _ = self.inference(x_c, v_c, x_q, v_q)
        return canvas

    def loss_func(self, x_c: Tensor, v_c: Tensor, x_q: Tensor, v_q: Tensor,
                  var: float = 1.0) -> Dict[str, Tensor]:
        """ELBO loss.

        Args:
            x_c (torch.Tensor): Context images, size `(b, m, c, h, w)`.
            v_c (torch.Tensor): Context viewpoints, size `(b, k)`.
            x_q (torch.Tensor): Query images, size `(b, m, c, h, w)`.
            v_q (torch.Tensor): Query viewpoints, size `(b, k)`.
            var (float, optional): Variance of observations normal dist.

        Returns:
            loss_dict (dict of [str, torch.Tensor]): Calculated losses.
        """

        _, loss_dict = self.inference(x_c, v_c, x_q, v_q, var)
        return loss_dict

    def inference(self, x_c: Tensor, v_c: Tensor, x_q: Tensor, v_q: Tensor,
                  var: float = 1.0) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Inference.

        Args:
            x_c (torch.Tensor): Context images, size `(b, m, c, h, w)`.
            v_c (torch.Tensor): Context viewpoints, size `(b, k)`.
            x_q (torch.Tensor): Query images, size `(b, m, c, h, w)`.
            v_q (torch.Tensor): Query viewpoints, size `(b, k)`.
            var (float, optional): Variance of observations normal dist.

        Returns:
            canvas (torch.Tensor): Reconstructed images, size `(b, c, h, w)`.
            loss_dict (dict of [str, torch.Tensor]): Calculated losses.
        """

        # Reshape: (b, m, c, h, w) -> (b*m, c, h, w)
        b, m, *x_dims = x_c.size()
        _, _, *v_dims = v_c.size()

        x_c = x_c.view(-1, *x_dims)
        v_c = v_c.view(-1, *v_dims)

        # Representation generated from context.
        r = self.representation(x_c, v_c)
        _, *r_dims = r.size()
        r = r.view(b, m, *r_dims)

        # Sum over representations: (b, c, h, w)
        r = r.sum(1)

        # Query images by v_q, i.e. reconstruct
        canvas, kl_loss = self.generator(x_q, v_q, r)

        # Reconstruction loss
        nll_loss = nll_normal(x_q, canvas, torch.tensor([var]), reduce=False)
        nll_loss = nll_loss.sum([1, 2, 3]).mean()

        # Returned loss
        loss_dict = {"loss": nll_loss + kl_loss, "nll_loss": nll_loss,
                     "kl_loss": kl_loss}

        return canvas, loss_dict

    def sample(self, x_c: Tensor, v_c: Tensor, v_q: Tensor) -> Tensor:
        """Inference.

        Args:
            x_c (torch.Tensor): Context images, size `(b, m, c, h, w)`.
            v_c (torch.Tensor): Context viewpoints, size `(b, k)`.
            v_q (torch.Tensor): Query viewpoints, size `(b, k)`.

        Returns:
            canvas (torch.Tensor): Reconstructed images, size `(b, c, h, w)`.
        """

        # Reshape: (b, m, c, h, w) -> (b*m, c, h, w)
        b, m, *x_dims = x_c.size()
        _, _, *v_dims = v_c.size()

        x_c = x_c.view(-1, *x_dims)
        v_c = v_c.view(-1, *v_dims)

        # Representation generated from context.
        r = self.representation(x_c, v_c)
        _, *r_dims = r.size()
        r = r.view(b, m, *r_dims)

        # Sum over representations: (b, c, h, w)
        r = r.sum(1)

        # Sample query images
        canvas = self.generator.sample(v_q, r)

        return canvas
