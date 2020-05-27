
import unittest

import torch

import gqnlib


class TestPyramid(unittest.TestCase):

    def test_pyramid(self):
        batch_size = 10
        x = torch.empty(batch_size, 3, 64, 64)
        v = torch.empty(batch_size, 7)

        model = gqnlib.Pyramid()
        r = model(x, v)

        self.assertTupleEqual(r.size(), (batch_size, 256, 1, 1))


if __name__ == "__main__":
    unittest.main()
