
"""Dataset class for GQN."""

import gzip
import pathlib

import torch


class SceneDataset(torch.nn.utils.Dataset):
    """SceneDataset class for GQN.

    SceneDataset class loads data files at each time accessed by index.

    Args:
        root_dir (str): Path to root directory
    """

    def __init__(self, root_dir: str):
        super().__init__()

        self.root_dir = pathlib.Path(root_dir)

    def __len__(self) -> int:
        """Returns number of files and directories in root dir.

        Returns:
            len (int): Number of objects in root dir.
        """

        return len(list(self.root_dir.iterdir()))

    def __getitem__(self, index):
        """Load and get data with specified index.

        This method read `<index>.pt.gz` file, which includes list of tuples,
        `(frames, cameras)`.

        Args:
            index (int): Index number
        """

        with gzip.open(str(self.root_dir), "rb") as f:
            dataset = f.read()

        frames = torch.stack([torch.from_numpy(data[0]) for data in dataset])
        cameras = torch.stack([torch.from_numpy(data[1]) for data in dataset])

        return frames, cameras
