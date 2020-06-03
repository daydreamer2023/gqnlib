
"""Dataset class for GQN."""

from typing import Tuple, List

import gzip
import pathlib
import random

import torch
from torch import Tensor


class SceneDataset(torch.utils.data.Dataset):
    """SceneDataset class for GQN.

    SceneDataset class loads data files at each time accessed by index.

    * This class reads `<index>.pt.gz` file, which includes a list of tuples
      `(images, viewpoints)`; images size = `(m, h, w, c)`, viewpoints size
      `(m, v)`, where `m` means sequence length of data.

    * Original data include too many image-viewpoints pairs, so this class
      splits the list of tuples to minibatches. Therefore, returned value
      is list of tuples `(images, viewpoints)`; images size =
      `(batch_size, m, c, h, w)`, viewpoints size = `(batch_size, m, v)`.

    Args:
        root_dir (str): Path to root directory.
        batch_size (int): Mini-batch size.

    Attributes:
        record_list (list of pathlib.Path): List of path to data files.
    """

    def __init__(self, root_dir: str, batch_size: int):
        super().__init__()

        root_dir = pathlib.Path(root_dir)
        self.record_list = sorted(root_dir.glob("*.pt.gz"))
        self.batch_size = batch_size

    def __len__(self) -> int:
        """Returns number of files and directories in root dir.

        Returns:
            len (int): Number of objects in root dir.
        """

        return len(self.record_list)

    def __getitem__(self, index: int) -> List[Tuple[Tensor]]:
        """Loads data file and returns data with specified index.

        * Images: `(batch_size, m, c, h, w)`
        * Viewpoints: `(batch_size, m, v)`

        Args:
            index (int): Index number.

        Returns:
            data_list (torch.Tensor): List of tuples of tensors
                `(images, viewpoints)`. Length of list is
                `data_num // batch_size`.
        """

        with gzip.open(self.record_list[index], "rb") as f:
            dataset = torch.load(f)

        # Read list of tuples
        images = []
        viewpoints = []
        for img, view in dataset:
            images.append(torch.from_numpy(img))
            viewpoints.append(torch.from_numpy(view))

        images = torch.stack(images)
        viewpoints = torch.stack(viewpoints)

        # Convert data size: NMHWC -> NMCHW
        images = images.permute(0, 1, 4, 2, 3)

        # Transform viewpoints
        viewpoints = transform_viewpoint(viewpoints)

        # Trim off extra elements
        batch_num = images.size(0) // self.batch_size
        images = images[:self.batch_size * batch_num]
        viewpoints = viewpoints[:self.batch_size * batch_num]

        _, *i_dims = images.size()
        _, *v_dims = viewpoints.size()

        # Resize: (n, m, *) -> (a, b, m, *)
        images = images.contiguous().view(
            batch_num, self.batch_size, *i_dims)
        viewpoints = viewpoints.contiguous().view(
            batch_num, self.batch_size, *v_dims)

        data_list = []
        for i in range(batch_num):
            data_list.append((images[i], viewpoints[i]))

        return data_list


def transform_viewpoint(viewpoints: Tensor) -> Tensor:
    """Transforms viewpoints for single batch.

    (x, y, z, yaw, pitch)
        -> (x, y, z, cos(yaw), sin(yaw), cos(pitch), sin(pitch))

    Args:
        viewpoints (torch.Tensor): Input viewpoints, size `(num, 5)`.

    Returns:
        converted (torch.Tensor): Transformed viewpoints, size `(num, 7)`.
    """

    pos, tmp = torch.split(viewpoints, 3, dim=-1)
    y, p = torch.split(tmp, 1, dim=-1)

    view = [pos, torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)]
    view = torch.cat(view, dim=-1)
    return view


def partition_scene(images: Tensor, viewpoints: Tensor, num_query: int = 1
                    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Partitions given data into context and query sets.

    Number of context is randomly sampled.

    Args:
        images (torch.Tensor): Image tensor, size
            `(batch, b, num_points, c, h, w)`.
        viewpoints (torch.Tensor): Viewpoints tensor, size
            `(batch, b, num_points, target)`.
        num_query (int, optional): Number of queries.

    Returns:
        x_c (torch.Tensor): Context images, size `(b, num_context, c, h, w)`.
        v_c (torch.Tensor): Context viewpoints, size `(b, num_context, t)`.
        x_q (torch.Tensor): Query images, size `(b, num_query, c, h, w)`.
        v_q (torch.Tensor): Query viewpoints, size `(b, num_query, t)`.

    Raises:
        ValueError: If `num_query` is equal or greater than `num_points`.
    """

    # Maximum number of context
    _, batch, num, *x_dims = images.size()
    _, _, _, *v_dims = viewpoints.size()

    if num_query >= num:
        raise ValueError(f"Number of queries (n={num_query}) must be less "
                         f"than -total data (n={num}).")

    # Squeeze dataset
    images = images.view(batch, num, *x_dims)
    viewpoints = viewpoints.view(batch, num, *v_dims)

    # Sample randum number of data
    n_data = random.randint(num_query + 1, num)
    indices = random.sample(range(num), n_data)

    # Partition into context and query
    context_idx = indices[:-num_query]
    query_idx = indices[-num_query:]

    x_c = images[:, context_idx]
    v_c = viewpoints[:, context_idx]

    x_q = images[:, query_idx]
    v_q = viewpoints[:, query_idx]

    return x_c, v_c, x_q, v_q
