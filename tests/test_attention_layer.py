
import unittest

import torch

import gqnlib


class TestDictionaryEncoder(unittest.TestCase):

    def test_foward(self):

        model = gqnlib.DictionaryEncoder()

        x = torch.randn(10, 3, 64, 64)
        v = torch.randn(10, 7)
        keys, values = model(x, v)

        self.assertTupleEqual(keys.size(), (490, 64, 8, 8))
        self.assertTupleEqual(values.size(), (490, 76, 8, 8))


if __name__ == "__main__":
    unittest.main()
