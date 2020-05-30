
"""Base class for Generative Query Network."""

from typing import Tuple, Dict

from torch import nn, Tensor


class BaseGQN(nn.Module):
    """Base class for GQN models."""

    def forward(self, x_c: Tensor, v_c: Tensor, x_q: Tensor, v_q: Tensor,
                var: float = 1.0) -> Tensor:
        """ELBO loss.

        Args:
            x_c (torch.Tensor): Context images, size `(b, m, c, h, w)`.
            v_c (torch.Tensor): Context viewpoints, size `(b, m, k)`.
            x_q (torch.Tensor): Query images, size `(b, n, c, h, w)`.
            v_q (torch.Tensor): Query viewpoints, size `(b, n, k)`.
            var (float, optional): Variance of observations normal dist.

        Returns:
            loss (torch.Tensor): ELBO loss.
        """

        _, loss_dict = self.inference(x_c, v_c, x_q, v_q, var)
        return loss_dict["loss"]

    def loss_func(self, x_c: Tensor, v_c: Tensor, x_q: Tensor, v_q: Tensor,
                  var: float = 1.0) -> Dict[str, Tensor]:
        """ELBO loss with separated nll and kl loss in dict.

        Args:
            x_c (torch.Tensor): Context images, size `(b, m, c, h, w)`.
            v_c (torch.Tensor): Context viewpoints, size `(b, m, k)`.
            x_q (torch.Tensor): Query images, size `(b, n, c, h, w)`.
            v_q (torch.Tensor): Query viewpoints, size `(b, n, k)`.
            var (float, optional): Variance of observations normal dist.

        Returns:
            loss_dict (dict of [str, torch.Tensor]): Calculated losses.
        """

        _, loss_dict = self.inference(x_c, v_c, x_q, v_q, var)
        return loss_dict

    def reconstruct(self, x_c: Tensor, v_c: Tensor, x_q: Tensor, v_q: Tensor
                    ) -> Tuple[Tensor, ...]:
        """Reconstruct given query images.

        Args:
            x_c (torch.Tensor): Context images, size `(b, m, c, h, w)`.
            v_c (torch.Tensor): Context viewpoints, size `(b, m, k)`.
            x_q (torch.Tensor): Query images, size `(b, n, c, h, w)`.
            v_q (torch.Tensor): Query viewpoints, size `(b, n, k)`.

        Returns:
            losses (tuple of torch.Tensor): Tuple of losses.
        """

        data, _ = self.inference(x_c, v_c, x_q, v_q)
        return data

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
            data (torch.Tensor): Tuple of inferenced data. Size of each tensor
                is `(b, n, c, h, w)`.
            loss_dict (dict of [str, torch.Tensor]): Calculated losses.
        """

        raise NotImplementedError
