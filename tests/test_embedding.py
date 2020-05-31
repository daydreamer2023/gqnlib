
import unittest

import torch

import gqnlib


class TestEmbeddingEncoder(unittest.TestCase):

    def test_forward(self):
        vocab_dim = 10
        embed_dim = 8
        n_head = 2
        h_dim = 20
        n_layer = 2
        model = gqnlib.EmbeddingEncoder(
            vocab_dim, embed_dim, n_head, h_dim, n_layer)

        x = torch.arange(vocab_dim).repeat(2).unsqueeze(0)
        x = x.repeat(9, 1)
        batch, length = x.size()
        d = model(x)

        self.assertTupleEqual(d.size(), (batch, length, embed_dim))


if __name__ == "__main__":
    unittest.main()
