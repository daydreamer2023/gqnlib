
"""Consistent Generative Query Network (a.k.a. JUMP).

(Reference)

A. Kumar et al., "Consistent Generative Query Network".
http://arxiv.org/abs/1807.02033
"""

from typing import Dict, Tuple

from torch import Tensor

from .base import BaseGQN
from .renderer import DRAWRenderer
from .representation import Simple
from .utils import nll_normal


class ConsistentGQN(BaseGQN):
    """Consistent Generative Query Network (a.k.a. JUMP).

    Args:
        representation_params (dict, optional): Parameters of representation
            network.
        generator_params (dict, optional): Parameters of generator network.
    """

    def __init__(self, representation_params: dict = {},
                 generator_params: dict = {}):
        super().__init__()

        self.representation = Simple(**representation_params)
        self.generator = DRAWRenderer(**generator_params)

    def inference(self, x_c: Tensor, v_c: Tensor, x_q: Tensor, v_q: Tensor,
                  var: float = 1.0
                  ) -> Tuple[Tuple[Tensor, ...], Dict[str, Tensor]]:
        """Inference.

        Args:
            x_c (torch.Tensor): Context images, size `(b, m, c, h, w)`.
            v_c (torch.Tensor): Context viewpoints, size `(b, m, k)`.
            x_q (torch.Tensor): Query images, size `(b, n, c, h, w)`.
            v_q (torch.Tensor): Query viewpoints, size `(b, n, k)`.
            var (float, optional): Variance of observations normal dist.

        Returns:
            canvas (torch.Tensor): Reconstructed images, size `(b, c, h, w)`.
            r_c (torch.Tensor): Representations of context, size
                `(b, r, x, y)`.
            r_q (torch.Tensor): Representations of query, size `(b, r, x, y)`.
            loss_dict (dict of [str, torch.Tensor]): Dict of calculated losses
                with size `(b, n)`.
        """

        # Reshape: (b, m, c, h, w) -> (b*m, c, h, w)
        b, m, *x_dims = x_c.size()
        _, _, *v_dims = v_c.size()
        n = x_q.size(1)

        x_c = x_c.view(-1, *x_dims)
        v_c = v_c.view(-1, *v_dims)

        x_q = x_q.view(-1, *x_dims)
        v_q = v_q.view(-1, *v_dims)

        # Representation generated from context
        r_c = self.representation(x_c, v_c)
        _, *r_dims = r_c.size()
        r_c = r_c.view(b, m, *r_dims)
        r_c = r_c.sum(1)

        # Representation generated from query
        r_q = self.representation(x_q, v_q)
        r_q = r_q.view(b, n, *r_dims)
        r_q = r_q.sum(1)

        # Copy representations for query
        r_c = r_c.repeat_interleave(n, dim=0)
        r_q = r_q.repeat_interleave(n, dim=0)

        # Query images by v_q, i.e. reconstruct
        canvas, kl_loss = self.generator(x_q, v_q, r_c, r_q)

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
        r_q = r_q.view(b, n, *r_dims)

        # Squash images to [0, 1]
        canvas = canvas.clamp(0.0, 1.0)

        return (canvas, r_c, r_q), loss_dict

    def sample(self, x_c: Tensor, v_c: Tensor, v_q: Tensor) -> Tensor:
        """Samples images `x_q` by context pair `(x, v)` and query viewpoint
        `v_q`.

        Args:
            x_c (torch.Tensor): Context images, size `(b, m, c, h, w)`.
            v_c (torch.Tensor): Context viewpoints, size `(b, m, k)`.
            v_q (torch.Tensor): Query viewpoints, size `(b, n, k)`.

        Returns:
            canvas (torch.Tensor): Reconstructed images, size
                `(b, n, c, h, w)`.
        """

        # Reshape: (b, m, c, h, w) -> (b*m, c, h, w)
        b, m, *x_dims = x_c.size()
        _, _, *v_dims = v_c.size()

        x_c = x_c.view(-1, *x_dims)
        v_c = v_c.view(-1, *v_dims)

        n = v_q.size(1)
        v_q = v_q.view(-1, *v_dims)

        # Representation generated from context.
        r = self.representation(x_c, v_c)
        _, *r_dims = r.size()
        r = r.view(b, m, *r_dims)

        # Sum over representations: (b, c, x, y)
        r = r.sum(1)
        r = r.repeat_interleave(n, dim=0)

        # Sample query images
        canvas = self.generator.sample(v_q, r)

        # Restore origina shape
        canvas = canvas.view(b, n, *x_dims)

        # Squash images to [0, 1]
        canvas = canvas.clamp(0.0, 1.0)

        return canvas

    def query(self, v_q: Tensor, r_c: Tensor) -> Tensor:
        """Query images with context representation.

        Args:
            v_q (torch.Tensor): Query viewpoints, size `(b, n, k)`.
            r_c (torch.Tensor): Representations of context, size
                `(b, n, r, x, y)`.

        Returns:
            canvas (torch.Tensor): Reconstructed images, size
                `(b, n, c, h, w)`.
        """

        # Squeeze data: (b, n, k) -> (b*n, k)
        b, n, v_dim = v_q.size()
        v_q = v_q.view(-1, v_dim)

        _, _, *r_dims = r_c.size()
        r_c = r_c.view(-1, *r_dims)

        # Sample data
        canvas = self.generator.sample(v_q, r_c)

        # Restore the original shape: (b*n, c, h, w) -> (b, n, c, h, w)
        _, *x_dims = canvas.size()
        canvas = canvas.view(b, n, *x_dims)

        # Squash images to [0, 1]
        canvas = canvas.clamp(0.0, 1.0)

        return canvas
