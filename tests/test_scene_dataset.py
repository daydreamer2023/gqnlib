
import unittest

import gzip
import pathlib
import tempfile

import torch

import gqnlib


class TestSceneDataset(unittest.TestCase):

    def test_len(self):
        self.dataset = gqnlib.SceneDataset(".")
        self.assertGreaterEqual(len(self.dataset), 0)

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
            self.dataset = gqnlib.SceneDataset(root)
            frames, cameras = self.dataset[1]

        self.assertTupleEqual(frames.size(), (20, 15, 3, 64, 64))
        self.assertTupleEqual(cameras.size(), (20, 15, 7))


if __name__ == "__main__":
    unittest.main()
