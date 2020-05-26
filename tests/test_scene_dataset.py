
import unittest

import pathlib

import gqnlib


class TestSceneDataset(unittest.TestCase):

    def setUp(self):
        root = pathlib.Path(__file__).parent.parent
        root /= "data/shepard_metzler_5_parts_torch/train/"
        self.dataset = gqnlib.SceneDataset(root)

    def test_len(self):
        self.assertGreaterEqual(len(self.dataset), 0)

    def test_getitem(self):
        frames, cameras = self.dataset[1]
        self.assertTupleEqual(frames.size(), (20, 15, 3, 64, 64))
        self.assertTupleEqual(cameras.size(), (20, 15, 7))


if __name__ == "__main__":
    unittest.main()
