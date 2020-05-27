
"""Generator network.

Generator network is similar to Convolutional DRAW [K. Gregor et al., 2016],
[K. Gregor et al., 2016]. The original GQN paper [S. M. Ali Eslami et al.,
2018] does not mention it, but the successive paper [A. Kumar et al., (2018)]
explicitly uses Convolutional DRAW.

(Reference)

* A. Kumar et al., "Consistent Generative Query Networks" (2018).
  http://arxiv.org/abs/1807.02033

* K. Gregor et al., "DRAW: A Recurrent Neural Network For Image Generation"
  (2015)
  http://arxiv.org/abs/1502.04623

* K. Gregor et al., "Towards conceptual compression" (2016).
  http://arxiv.org/abs/1604.08772

(Reference code)

https://github.com/wohlert/generative-query-network-pytorch/blob/master/draw/draw.py
"""

from typing import Tuple

import math

import torch
from torch import nn, Tensor


class Conv2dLSTMCell(nn.Module):
    """2D Convolutional long short-term memory (LSTM) cell.

    Args:
        in_channels (int): Number of input channel.
        out_channels (int): Number of output channel.
        kernel_size (int): Size of image kernel.
        stride (int): Length of kernel stride.
        padding (int): Number of pixels to pad with.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, padding: int):
        super().__init__()

        kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding)

        self.forget = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.input = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.output = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.state = nn.Conv2d(in_channels, out_channels, **kwargs)

        self.transform = nn.Conv2d(out_channels, in_channels, **kwargs)

    def forward(self, x: Tensor, states: Tuple[Tensor, Tensor]
                ) -> Tuple[Tensor, Tensor]:
        """Forward through cell.

        Args:
            x (torch.Tensor): Input to send through.
            states (tuple of torch.Tensor): (hidden, cell) pair of internal
                state.

        Returns:
            next_states (tuple of torch.Tensor): (hidden, cell) pair of
                internal next state.
        """

        hidden, cell = states
        x = x + self.transform(hidden)

        forget_gate = torch.sigmoid(self.forget(x))
        input_gate = torch.sigmoid(self.input(x))
        output_gate = torch.sigmoid(self.output(x))
        state_gate = torch.tanh(self.state(x))

        # Update internal cell state
        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * torch.tanh(cell)

        return hidden, cell


class ConvolutionalDRAW(nn.Module):
    """Convolutional DRAW (Deep Recurrent Attentive Writer).

    Args:
        x_channel (int, optional): Number of channel in input images.
        v_dim (int, optional): Dimensions of viewpoints.
        r_dim (int, optional): Dimensions of representations.
        z_channel (int, optional): Number of channel in latent variable.
        h_channel (int, optional): Number of channel in hidden states.
        n_layer (int, optional): Number of recurrent layers.
        scale (int, optional): Scale of image generation process.
    """

    def __init__(self, x_channel: int = 3, v_dim: int = 7, r_dim: int = 256,
                 z_channel: int = 64, h_channel: int = 128, n_layer: int = 8,
                 scale: int = 4):
        super().__init__()

        self.x_channel = x_channel
        self.h_channel = h_channel
        self.z_channel = z_channel
        self.n_layer = n_layer
        self.scale = scale

        # Distributions (variational posterior / prior)
        kwargs = dict(kernel_size=5, stride=1, padding=2)
        self.posterior = nn.Conv2d(h_channel, z_channel * 2, **kwargs)
        self.prior = nn.Conv2d(h_channel, z_channel * 2, **kwargs)

        # Top layer
        kwargs = dict(kernel_size=scale, stride=scale, padding=0, bias=False)
        self.read_head = nn.Conv2d(x_channel, x_channel, **kwargs)
        self.write_head = nn.ConvTranspose2d(h_channel, h_channel, **kwargs)

        # Recurrent encoder/decoder models
        kwargs = dict(kernel_size=5, stride=1, padding=2)
        self.encoder = Conv2dLSTMCell(v_dim + r_dim + x_channel + h_channel,
                                      h_channel, **kwargs)
        self.decoder = Conv2dLSTMCell(v_dim + r_dim + z_channel, h_channel,
                                      **kwargs)

        # Final layer to convert u -> canvas
        kwargs = dict(kernel_size=1, stride=1, padding=0)
        self.observation = nn.Conv2d(h_channel, x_channel, **kwargs)

    def forward(self, x: Tensor, v: Tensor, r: Tensor
                ) -> Tuple[Tensor, Tensor]:
        """Inferences given query pair (x, v) and representation r.

        Args:
            x (torch.Tensor): True queried iamges `x_q`, size `(b, c, h, w)`.
            v (torch.Tensor): Query of viewpoints `v_q`, size `(b, v)`.
            r (torch.Tensor): Representation of context, size `(b, c, h, w)`.

        Returns:
            canvas (torch.Tensor): Reconstructed images, size `(b, c, h, w)`.
            kl_loss (torch.Tensor): Calculated KL loss, size `(1)`.
        """

        # Data size
        batch_size, _, h, w = x.size()
        h_scale = h // self.scale
        w_scale = w // self.scale

        # Generator initial state
        h_enc = x.new_zeros((batch_size, self.h_channel, h_scale, w_scale))
        c_dec = x.new_zeros((batch_size, self.h_channel, h_scale, w_scale))

        # Inference initial state
        h_dec = x.new_zeros((batch_size, self.h_channel, h_scale, w_scale))
        c_enc = x.new_zeros((batch_size, self.h_channel, h_scale, w_scale))

        # Canvas that data is drawn on
        u = x.new_zeros((batch_size, self.h_channel, h, w))

        # KL loss value
        kl_loss = 0

        # Reshape: Downsample x, upsample v and r
        x = self.read_head(x)
        v = v.view(batch_size, -1, 1, 1).repeat(1, 1, h_scale, w_scale)
        if r.size(2) != h_scale:
            r = r.repeat(1, 1, h_scale, w_scale)

        for _ in range(self.n_layer):
            # Prior factor (eta_pi)
            p_mu, p_logvar = torch.chunk(self.prior(h_dec), 2, dim=1)

            # Inference state update
            h_enc, c_enc = self.encoder(torch.cat([h_dec, x, v, r], dim=1),
                                        (h_enc, c_enc))

            # Posterior factor (eta_e)
            q_mu, q_logvar = torch.chunk(self.posterior(h_enc), 2, dim=1)

            # Posterior sample
            z = q_mu + (0.5 * q_logvar).exp() * torch.randn_like(q_logvar)

            # Generator state update
            h_dec, c_dec = self.decoder(torch.cat([z, v, r], dim=1),
                                        (h_dec, c_dec))

            # Draw canvas
            u = u + self.write_head(h_dec)

            # Calculate loss
            _kl_tmp = kl_divergence_normal(q_mu, q_logvar.exp(), p_mu,
                                           p_logvar.exp(), reduce=False)
            kl_loss += _kl_tmp.sum([1, 2, 3]).mean()

        # Returned values
        canvas = torch.sigmoid(self.observation(u))

        return canvas, kl_loss

    def sample(self, v: Tensor, r: Tensor, x_shape: Tuple[int, int] = (64, 64)
               ) -> Tensor:
        """Samples images from the prior given viewpoint and representation.

        Args:
            v (torch.Tensor): Query of viewpoints `v_q`, size `(b, v)`.
            r (torch.Tensor): Representation of context, size `(b, c, h, w)`.
            x_shape (tuple of int, optional): Sampled x shape.

        Returns:
            canvas (torch.Tensor): Sampled data, size `(b, c, h, w)`.
        """

        batch_size = v.size(0)
        h, w = x_shape
        h_scale = h // self.scale
        w_scale = w // self.scale

        # Hidden states
        h_dec = v.new_zeros((batch_size, self.h_channel, h_scale, w_scale))
        c_dec = v.new_zeros((batch_size, self.h_channel, h_scale, w_scale))

        # Canvas that data is drawn on
        u = v.new_zeros((batch_size, self.h_channel, h, w))

        # Upsample v and r
        v = v.view(batch_size, -1, 1, 1).repeat(1, 1, h_scale, w_scale)
        if r.size(2) != h_scale:
            r = r.repeat(1, 1, h_scale, w_scale)

        for _ in range(self.n_layer):
            # Sample prior
            p_mu, p_logvar = torch.chunk(self.prior(h_dec), 2, dim=1)
            z = p_mu + (0.5 * p_logvar).exp() * torch.randn_like(p_logvar)

            # Decode
            h_dec, c_dec = self.decoder(torch.cat([z, v, r], dim=1),
                                        (h_dec, c_dec))

            # Draw canvas
            u = u + self.write_head(h_dec)

        canvas = self.observation(u)

        return canvas


def nll_normal(x: Tensor, mu: Tensor, var: Tensor, reduce: bool = True
               ) -> Tensor:
    """Negative log likelihood for 1-D Normal distribution.

    Args:
        mu (torch.Tensor): Mean vector.
        var (torch.Tensor): Variance vector.
        reduce (bool, optional): If `True`, sum calculated loss for each
            data point.

    Returns:
        nll (torch.Tensor): Calculated nll for each data.
    """

    nll = 0.5 * ((2 * math.pi * var).log() + (x - mu) ** 2 / var)

    if reduce:
        return nll.sum(-1)
    return nll


def kl_divergence_normal(mu0: Tensor, var0: Tensor, mu1: Tensor, var1: Tensor,
                         reduce: bool = True) -> Tensor:
    """Kullback Leibler divergence for 1-D Normal distributions.

    p = N(mu0, var0)
    q = N(mu1, var1)
    KL(p||q) = 1/2 * (var0/var1 + (mu1-mu0)^2/var1 - 1 + log(var1/var0))

    Args:
        mu0 (torch.Tensor): Mean vector of p.
        var0 (torch.Tensor): Diagonal variance of p.
        mu1 (torch.Tensor): Mean vector of q.
        var1 (torch.Tensor): Diagonal variance of q.
        reduce (bool, optional): If `True`, sum calculated loss for each
            data point.

    Returns:
        kl (torch.Tensor): Calculated kl divergence for each data.
    """

    diff = mu1 - mu0
    kl = (var0 / var1 + diff ** 2 / var1 - 1 + (var1 / var0).log()) * 0.5

    if reduce:
        return kl.sum(-1)
    return kl
