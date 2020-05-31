
"""Embedding representation.

ref) Sequence-to-Sequence Modeling with nn.Transformer and TorchText
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""

import math

import torch
from torch import Tensor, nn


class EmbeddingEncoder(nn.Module):
    """Embedding encoder class.

    This class is a part of Transformer.

    Args:
        vocab_dim (int): Vocabulary size.
        embed_dim (int): Dimension of embedding layer.
        n_head (int, optional): Number of heads in multi-head attention models.
        h_dim (int, optional): Dimension of feed-forward network of
            `TransformerEncoderLayer` class.
        n_layer (int, optional): Number of sub-encoder-layers in the encoder.
        dropout (float, optional): Dropout rate (default=0.1).
        max_len (int, optional): Max length of input strings.
    """

    def __init__(self, vocab_dim: int, embed_dim: int, n_head: int = 2,
                 h_dim: int = 200, n_layer: int = 2, dropout: float = 0.1,
                 max_len: int = 500):
        super().__init__()

        # Size
        self.embed_dim = embed_dim

        # Embedding layer
        self.embedding = nn.Embedding(vocab_dim, embed_dim)

        # Attention layer
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, n_head,
                                                   h_dim, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layer)

        # Positional encoder
        self.pos_encoder = PositionalEncoder(embed_dim, dropout, max_len)

        # Mask for input sequence
        self.src_mask = None

    def forward(self, src: Tensor) -> Tensor:
        """Forward through encoder.

        Args:
            src (int): Source tensor, size `(b, l)`.

        Returns:
            encoded (torch.Tensor): Encoded source, size `(b, l, embed_dim)`.
        """

        # Check mask size
        if (self.src_mask is None) or (self.src_mask.size(0) != len(src)):
            self._generate_square_subsequent_mask(src)

        src = self.embedding(src) * math.sqrt(self.embed_dim)
        src = self.pos_encoder(src)
        output = self.encoder(src, self.src_mask)
        return output

    def _generate_square_subsequent_mask(self, src: Tensor) -> None:
        """Generates a mask for the sequence.

        The masked positions are filled with float('-inf'). Unmasked positions
        are filled with 0.

        Args:
            src (torch.Tensor): Source tensor.
        """

        size = src.size(0)
        mask = torch.triu(torch.ones(size, size)).T
        mask = mask.masked_fill(
            mask == 0, float("-inf")).masked_fill(mask == 1, 0.)

        self.src_mask = mask.to(device=src.device)


class PositionalEncoder(nn.Module):
    """Postional encoder for Transformer.

    Args:
        d_model (int): Dimension size of model.
        dropout (float, optional): Dropout rate.
        max_len (int, optional): Max length of given sequences.
    """

    def __init__(self, d_model: int, dropout: float = 0.1,
                 max_len: int = 5000):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.exp(torch.arange(0, d_model, 2).float())
                    * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Forward through positional emcode.

        Args:
            x (torch.Tensor): Input tensor, size `(l, n, d)`, where length `l`
                <= max_len.

        Returns:
            encoded (torch.Tensor): Output tensor, size `(l, n, d)`.
        """

        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class RepresentationNetwork(nn.Module):
    """Representation Network r = f(v, c).

    Args:
        vocab_dim (int): Vocabulary size.
        embed_dim (int): Dimension of embedding vectors.
        v_dim (int, optional): Dimension of viewpoints.
        r_dim (int, optional): Dimension of representations.
        embed_params (dict, optional): Parameters for embedding encoder.
    """

    def __init__(self, vocab_dim: int, embed_dim: int = 64, v_dim: int = 4,
                 r_dim: int = 256, embed_params: dict = {}):
        super().__init__()

        self.viewpoint_encoder = nn.Linear(v_dim, 32)
        self.embedding_encoder = EmbeddingEncoder(
            vocab_dim, embed_dim, **embed_params)
        self.fc = nn.Sequential(
            nn.Linear(32 + embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, r_dim),
        )

    def forward(self, v: Tensor, c: Tensor) -> Tensor:
        """Forward: r = f(v, c)

        Args:
            v (torch.Tensor): Viewpoints, size `(b, v)`.
            c (torch.Tensor): Captions, size `(b, c)`.

        Returns:
            r (torch.Tensor): Representations, size `(b, r)`.
        """

        # Encode viewpoints
        v = self.viewpoint_encoder(v)

        # Encode captions
        c = self.embedding_encoder(c)
        c = c.sum(1)

        r = self.fc(torch.cat([v, c], dim=1))

        return r