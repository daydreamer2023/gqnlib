
import unittest

import gzip
import pathlib
import tempfile

import torch

import gqnlib


class TestSceneDataset(unittest.TestCase):

    def test_len(self):
        dataset = gqnlib.SceneDataset(".")
        self.assertGreaterEqual(len(dataset), 0)

    def test_getitem(self):
        # Dummy data
        imgs = torch.empty(15, 64, 64, 3)
        tgts = torch.empty(15, 5)
        data = [(imgs.numpy(), tgts.numpy())] * 20

        with tempfile.TemporaryDirectory() as root:
            path = str(pathlib.Path(root, "1.pt.gz"))
            with gzip.open(path, "wb") as f:
                torch.save(data, f)

            # Access data
            dataset = gqnlib.SceneDataset(root)
            frames, cameras = dataset[0]

        self.assertTupleEqual(frames.size(), (20, 15, 3, 64, 64))
        self.assertTupleEqual(cameras.size(), (20, 15, 7))

    def test_multi_getitem(self):
        # Dummy data
        imgs = torch.empty(15, 64, 64, 3)
        tgts = torch.empty(15, 5)
        data = [(imgs.numpy(), tgts.numpy())] * 20

        with tempfile.TemporaryDirectory() as root:
            for i in range(10):
                path = str(pathlib.Path(root, f"{i}.pt.gz"))
                with gzip.open(path, "wb") as f:
                    torch.save(data, f)

            # Access data
            dataset = gqnlib.SceneDataset(root)
            loader = torch.utils.data.DataLoader(dataset, batch_size=5)
            for frames, cameras in loader:
                self.assertTupleEqual(frames.size(), (5, 20, 15, 3, 64, 64))
                self.assertTupleEqual(cameras.size(), (5, 20, 15, 7))

    def test_partition(self):
        # Dummy data
        imgs = torch.empty(15, 64, 64, 3)
        tgts = torch.empty(15, 5)
        data = [(imgs.numpy(), tgts.numpy())] * 20

        with tempfile.TemporaryDirectory() as root:
            for i in range(5):
                path = str(pathlib.Path(root, f"{i}.pt.gz"))
                with gzip.open(path, "wb") as f:
                    torch.save(data, f)

            # Access data
            dataset = gqnlib.SceneDataset(root)
            loader = torch.utils.data.DataLoader(dataset, batch_size=5)
            images, viewpoints = next(iter(loader))

        x_c, v_c, x_q, v_q = gqnlib.partition(images, viewpoints)

        # x_c
        self.assertEqual(x_c.size(0), 5 * 20)
        self.assertLess(x_c.size(1), 15)
        self.assertEqual(x_c.size(2), 3)
        self.assertEqual(x_c.size(3), 64)
        self.assertEqual(x_c.size(4), 64)

        # v_c
        self.assertEqual(v_c.size(0), 5 * 20)
        self.assertLess(v_c.size(1), 15)
        self.assertEqual(v_c.size(2), 7)

        # x_q
        self.assertEqual(x_q.size(0), 5 * 20)
        self.assertEqual(x_q.size(1), 1)
        self.assertEqual(x_q.size(2), 3)
        self.assertEqual(x_q.size(3), 64)
        self.assertEqual(x_q.size(4), 64)

        # v_q
        self.assertEqual(v_q.size(0), 5 * 20)
        self.assertEqual(v_q.size(1), 1)
        self.assertEqual(v_q.size(2), 7)


if __name__ == "__main__":
    unittest.main()
