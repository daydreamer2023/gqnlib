
"""SLIM (Spatial Language Integrating Model)."""

from typing import Tuple, Dict

from torch import Tensor

from .base import BaseGQN
from .embedding import RepresentationNetwork
from .slim_generator import SlimGenerator
from .utils import nll_normal


class CaptionedGQN(BaseGQN):
    """Captioned Generative Query Network.

    Implementation of SLIM (Spatial Language Integrating Model).
    """

    def __init__(self, vocab_dim: int, representation_params: dict = {},
                 generator_params: dict = {}):
        super().__init__()

        self.representation = RepresentationNetwork(
            vocab_dim=vocab_dim, **representation_params)
        self.generator = SlimGenerator(**generator_params)

    def inference(self, d_c: Tensor, v_c: Tensor, x_q: Tensor, v_q: Tensor,
                  var: float = 1.0
                  ) -> Tuple[Tuple[Tensor, ...], Dict[str, Tensor]]:
        """Inference.

        Args:
            d_c (torch.Tensor): Context captions, size `(b, m, l)`.
            v_c (torch.Tensor): Context viewpoints, size `(b, m, k)`.
            x_q (torch.Tensor): Query images, size `(b, n, c, h, w)`.
            v_q (torch.Tensor): Query viewpoints, size `(b, n, k)`.
            var (float, optional): Variance of observations normal dist.

        Returns:
            canvas (torch.Tensor): Reconstructed images, size `(b, c, h, w)`.
            r_c (torch.Tensor): Representations of context, size
                `(b, r, h, w)`.
            r_q (torch.Tensor): Representations of query, size `(b, r, h, w)`.
            loss_dict (dict of [str, torch.Tensor]): Dict of calculated losses
                with size `(b, n)`.
        """

        # Reshape: (b, m, c, h, w) -> (b*m, c, h, w)
        b, m, *d_dims = d_c.size()
        _, _, *v_dims = v_c.size()
        _, n, *x_dims = x_q.size()

        d_c = d_c.view(-1, *d_dims)
        v_c = v_c.view(-1, *v_dims)

        x_q = x_q.view(-1, *x_dims)
        v_q = v_q.view(-1, *v_dims)

        # Representation generated from context
        r_c = self.representation(d_c, v_c)
        _, *r_dims = r_c.size()
        r_c = r_c.view(b, m, *r_dims)
        r_c = r_c.sum(1)

        # Copy representations for query
        r_c = r_c.repeat_interleave(n, dim=0)

        # Query images by v_q, i.e. reconstruct
        canvas, kl_loss = self.generator(x_q, v_q, r_c)

        # Reconstruction loss
        nll_loss = nll_normal(x_q, canvas, x_q.new_ones((1)) * var,
                              reduce=False)
        nll_loss = nll_loss.sum([1, 2, 3])

        # Returned loss
        nll_loss = nll_loss.view(b, n)
        kl_loss = kl_loss.view(b, n)
        loss_dict = {"loss": nll_loss + kl_loss, "nll_loss": nll_loss,
                     "kl_loss": kl_loss}

        # Restore original shape
        canvas = canvas.view(b, n, *x_dims)
        r_c = r_c.view(b, n, *r_dims)

        return (canvas, r_c), loss_dict