
import unittest

import torch

import gqnlib


class TestLatentDistribution(unittest.TestCase):

    def test_prior(self):
        r_channel = 32
        e_channel = 128
        h_channel = 64
        z_channel = 3
        stride = 2
        model = gqnlib.LatentDistribution(
            r_channel, e_channel, h_channel, z_channel, stride)

        r = torch.randn(4, r_channel, 16, 16)
        z = torch.randn(4, z_channel, 8, 8)
        h = torch.randn(4, h_channel, 8, 8)
        c = torch.randn(4, h_channel, 8, 8)

        mu, logvar, h_n, c_n = model(r, z, h, c)

        self.assertTupleEqual(mu.size(), z.size())
        self.assertTupleEqual(logvar.size(), z.size())
        self.assertTupleEqual(h_n.size(), h.size())
        self.assertTupleEqual(c_n.size(), c.size())

    def test_posterior(self):
        r_channel = 32
        e_channel = 128
        h_channel = 64
        z_channel = 3
        stride = 2
        model = gqnlib.LatentDistribution(
            r_channel * 2, e_channel, h_channel, z_channel, stride)

        r = torch.randn(4, r_channel, 16, 16)
        z = torch.randn(4, z_channel, 8, 8)
        h = torch.randn(4, h_channel, 8, 8)
        c = torch.randn(4, h_channel, 8, 8)

        mu, logvar, h_n, c_n = model(torch.cat([r, r], dim=1), z, h, c)

        self.assertTupleEqual(mu.size(), z.size())
        self.assertTupleEqual(logvar.size(), z.size())
        self.assertTupleEqual(h_n.size(), h.size())
        self.assertTupleEqual(c_n.size(), c.size())


class TestRenderer(unittest.TestCase):

    def test_forward(self):
        h_channel = 64
        d_channel = 64
        z_channel = 3
        u_channel = 128
        v_dim = 7
        stride = 2
        model = gqnlib.Renderer(
            h_channel, d_channel, z_channel, u_channel, v_dim, stride)

        z = torch.randn(4, z_channel, 8, 8)
        v = torch.randn(4, 7)
        u = torch.randn(4, u_channel, 16, 16)
        h = torch.randn(4, h_channel, 8, 8)
        c = torch.randn(4, h_channel, 8, 8)

        u_n, h_n, c_n = model(z, v, u, h, c)

        self.assertTupleEqual(u_n.size(), u.size())
        self.assertTupleEqual(h_n.size(), h.size())
        self.assertTupleEqual(c_n.size(), c.size())


class TestDRAWRenderer(unittest.TestCase):

    def test_forward(self):
        x_channel = 3
        u_channel = 128
        r_channel = 32
        e_channel = 128
        d_channel = 128
        h_channel = 64
        z_channel = 3
        stride = 2
        v_dim = 7
        model = gqnlib.DRAWRenderer(
            x_channel, u_channel, r_channel, e_channel, d_channel, h_channel,
            z_channel, stride, v_dim)

        x = torch.randn(4, x_channel, 64, 64)
        v = torch.randn(4, v_dim)
        r_c = torch.randn(4, r_channel, 16, 16)
        r_q = torch.randn(4, r_channel, 16, 16)

        canvas, kl_loss = model(x, v, r_c, r_q)

        self.assertTupleEqual(canvas.size(), x.size())
        self.assertGreater(kl_loss, 0)


if __name__ == "__main__":
    unittest.main()
