
import unittest

import torch

import gqnlib


class TestGenerativeQueryNetwork(unittest.TestCase):

    def setUp(self):
        self.model = gqnlib.GenerativeQueryNetwork()

    def test_forwad(self):
        x_c = torch.randn(4, 15, 3, 64, 64)
        v_c = torch.randn(4, 15, 7)
        x_q = torch.randn(4, 3, 64, 64)
        v_q = torch.randn(4, 7)

        canvas, loss_dict = self.model.inference(x_c, v_c, x_q, v_q)

        self.assertTupleEqual(canvas.size(), (4, 3, 64, 64))
        self.assertGreater(loss_dict["loss"], 0)
        self.assertGreater(loss_dict["nll_loss"], 0)
        self.assertGreater(loss_dict["kl_loss"], 0)

    def test_forward(self):
        x_c = torch.randn(4, 15, 3, 64, 64)
        v_c = torch.randn(4, 15, 7)
        x_q = torch.randn(4, 3, 64, 64)
        v_q = torch.randn(4, 7)

        canvas = self.model(x_c, v_c, x_q, v_q)
        self.assertTupleEqual(canvas.size(), (4, 3, 64, 64))

    def test_loss_func(self):
        x_c = torch.randn(4, 15, 3, 64, 64)
        v_c = torch.randn(4, 15, 7)
        x_q = torch.randn(4, 3, 64, 64)
        v_q = torch.randn(4, 7)

        loss_dict = self.model.loss_func(x_c, v_c, x_q, v_q)
        self.assertGreater(loss_dict["loss"], 0)
        self.assertGreater(loss_dict["nll_loss"], 0)
        self.assertGreater(loss_dict["kl_loss"], 0)

    def test_sample(self):
        x_c = torch.randn(4, 15, 3, 64, 64)
        v_c = torch.randn(4, 15, 7)
        v_q = torch.randn(4, 7)

        canvas = self.model.sample(x_c, v_c, v_q)
        self.assertTupleEqual(canvas.size(), (4, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()
