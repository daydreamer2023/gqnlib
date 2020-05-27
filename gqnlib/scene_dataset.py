
"""Dataset class for GQN."""

from typing import Tuple

import gzip
import pathlib
import random

import torch
from torch import Tensor


class SceneDataset(torch.utils.data.Dataset):
    """SceneDataset class for GQN.

    SceneDataset class loads data files at each time accessed by index.

    Args:
        root_dir (str): Path to root directory

    Attributes:
        record_list (list of pathlib.Path): List of path to data files.
    """

    def __init__(self, root_dir: str):
        super().__init__()

        root_dir = pathlib.Path(root_dir)
        self.record_list = sorted(root_dir.glob("*.pt.gz"))

    def __len__(self) -> int:
        """Returns number of files and directories in root dir.

        Returns:
            len (int): Number of objects in root dir.
        """

        return len(self.record_list)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Loads data file and returns data with specified index.

        This method reads `<index>.pt.gz` file, which includes list of tuples
        `(images, viewpoints)`.

        Args:
            index (int): Index number.

        Returns:
            images (torch.Tensor): Image tensor, size
                `(observations, num_points, 3, 64, 64)`.
            viewpoints (torch.Tensor): View points, size
                `(observations, num_points, 7)`.
        """

        with gzip.open(self.record_list[index], "rb") as f:
            dataset = torch.load(f)

        images = torch.stack([torch.from_numpy(x[0]) for x in dataset])
        viewpoints = torch.stack([torch.from_numpy(x[1]) for x in dataset])

        # Convert data size: BNHWC -> BNCHW
        images = images.permute(0, 1, 4, 2, 3)

        # Transform viewpoints
        viewpoints = transform_viewpoint(viewpoints)

        return images, viewpoints


def transform_viewpoint(viewpoints: Tensor) -> Tensor:
    """Transforms viewpoints.

    (x, y, z, yaw, pitch)
        -> (x, y, z, cos(yaw), sin(yaw), cos(pitch), sin(pitch))

    Args:
        viewpoints (torch.Tensor): Input viewpoints, size `(*, 5)`.

    Returns:
        converted (torch.Tensor): Transformed viewpoints, size `(*, 7)`.
    """

    pos, tmp = torch.split(viewpoints, 3, dim=-1)
    y, p = torch.split(tmp, 1, dim=-1)

    view = [pos, torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)]
    view = torch.cat(view, dim=-1)
    return view


def partition(images: Tensor, viewpoints: Tensor
              ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Partitions given data into context and query sets.

    * Number of context is randomly sampled.
    * Number of query is 1.

    Args:
        images (torch.Tensor): Image tensor, size
            `(batch, observations, num_points, c, h, w)`.
        viewpoints (torch.Tensor): Viewpoints tensor, size
            `(batch, observations, num_points, target)`.

    Returns:
        x_c (torch.Tensor): Context images, size `(b*m, num_context, c, h, w)`.
        v_c (torch.Tensor): Context viewpoints, size `(b*m, num_context, t)`.
        x_q (torch.Tensor): Query images, size `(b*m, c, h, w)`.
        v_q (torch.Tensor): Query viewpoints, size `(b*m, t)`.
    """

    # Maximum number of context
    _, _, num, *x_dims = images.size()
    _, _, num, *v_dims = viewpoints.size()

    # Squeeze dataset
    images = images.view(-1, num, *x_dims)
    viewpoints = viewpoints.view(-1, num, *v_dims)

    # Sample randum number of data
    n_data = random.randint(2, num - 1)
    indices = random.sample(range(num), n_data)

    # Partition into context and query
    context_idx = indices[:-1]
    query_idx = indices[-1]

    x_c = images[:, context_idx]
    v_c = viewpoints[:, context_idx]

    x_q = images[:, query_idx]
    v_q = viewpoints[:, query_idx]

    return x_c, v_c, x_q, v_q
