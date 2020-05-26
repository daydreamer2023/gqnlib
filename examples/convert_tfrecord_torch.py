
"""Convert tfrecord to torch data.

Original dataset provided by DeepMind is saved as tfrecord format. This script
convert these records to torch format.

Refrence)

https://github.com/deepmind/gqn-datasets

https://github.com/deepmind/gqn-datasets/blob/master/data_reader.py

https://github.com/iShohei220/torch-gqn/blob/master/dataset/convert2torch.py

https://github.com/wohlert/generative-query-network-pytorch/blob/master/scripts/tfrecord-converter.py
"""

import argparse
import collections
import gzip
import multiprocessing as mp
import pathlib

import tensorflow as tf
import torch


DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['basepath', 'train_size', 'test_size', 'frame_size', 'sequence_size']
)
Scene = collections.namedtuple('Scene', ['frames', 'cameras'])


_DATASETS = dict(
    jaco=DatasetInfo(
        basepath='jaco',
        train_size=3600,
        test_size=400,
        frame_size=64,
        sequence_size=11),

    mazes=DatasetInfo(
        basepath='mazes',
        train_size=1080,
        test_size=120,
        frame_size=84,
        sequence_size=300),

    rooms_free_camera_with_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_with_object_rotations',
        train_size=2034,
        test_size=226,
        frame_size=128,
        sequence_size=10),

    rooms_ring_camera=DatasetInfo(
        basepath='rooms_ring_camera',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    rooms_free_camera_no_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_no_object_rotations',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    shepard_metzler_5_parts=DatasetInfo(
        basepath='shepard_metzler_5_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15),

    shepard_metzler_7_parts=DatasetInfo(
        basepath='shepard_metzler_7_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15)
)
_NUM_CHANNELS = 3
_NUM_RAW_CAMERA_PARAMS = 5


def run_loop(dataset_name: str, org_dir: pathlib.Path, save_dir: pathlib.Path
             ) -> None:
    """Runs process in for-loop.

    Args:
        dataset_name (str): Name of dataset.
        org_dir (pathlib.Path): Path to original data.
        save_dir (pathlib.Path): Path to saved data.
    """

    # Dataset info
    dataset_info = _DATASETS[dataset_name]

    for path in sorted(org_dir.glob("*.tfrecord")):
        # Saved path
        base_name = path.stem.split("-")[0]
        save_path = save_dir / f"{base_name}.pt.gz"

        # Read tf record
        raw_data = tf.data.TFRecordDataset(str(path))

        # Run in multiprocess
        p = mp.Process(target=convert_raw_to_torch,
                       args=(dataset_info, raw_data, str(save_path)))
        p.start()
        p.join()


def convert_raw_to_torch(dataset_info: collections.namedtuple,
                         raw_data: tf.Tensor,
                         path: str) -> None:
    """Converts raw data to tensor and saves into torch gziped file.

    Args:
        dataset_info (collections.namedtuple): Information tuple.
        raw_data (tf.Tensor): Tensor of original data.
        path (str): Path to saved file.
    """

    feature_map = {
        'frames': tf.io.FixedLenFeature(
            shape=dataset_info.sequence_size, dtype=tf.string),
        'cameras': tf.io.FixedLenFeature(
            shape=[dataset_info.sequence_size * _NUM_RAW_CAMERA_PARAMS],
            dtype=tf.float32),
    }
    example = tf.io.parse_example(raw_data, feature_map)
    frames = _preprocess_frames(dataset_info, example)
    cameras = _preprocess_cameras(dataset_info, example)
    scene = Scene(frames=frames.numpy(), cameras=cameras.numpy())
    with gzip.open(path, "wb") as f:
        torch.save(scene, f)


def _convert_frame_data(jpeg_data):
    decoded_frames = tf.image.decode_jpeg(jpeg_data)
    return tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)


def _preprocess_frames(dataset_info, example):
    frames = tf.concat(example["frames"], axis=0)
    frames = tf.map_fn(_convert_frame_data, tf.reshape(frames, [-1]),
                       dtype=tf.float32, back_prop=False)
    dataset_image_dimensions = tuple(
        [dataset_info.frame_size] * 2 + [_NUM_CHANNELS])
    frames = tf.reshape(
        frames, (-1, dataset_info.sequence_size) + dataset_image_dimensions)

    # Squeeze images to 64x64
    if dataset_info.frame_size != 64:
        frames = tf.reshape(frames, (-1,) + dataset_image_dimensions)
        new_frame_dimensions = (64,) * 2 + (_NUM_CHANNELS,)
        frames = tf.image.resize(
            frames, new_frame_dimensions[:2], align_corners=True)
        frames = tf.reshape(
            frames, (-1, dataset_info.sequence_size) + new_frame_dimensions)

    return frames


def _preprocess_cameras(dataset_info, example):
    raw_pose_params = example["cameras"]
    raw_pose_params = tf.reshape(
        raw_pose_params,
        [-1, dataset_info.sequence_size, _NUM_RAW_CAMERA_PARAMS])
    pos = raw_pose_params[:, :, 0:3]
    yaw = raw_pose_params[:, :, 3:4]
    pitch = raw_pose_params[:, :, 4:5]
    cameras = tf.concat(
        [pos, tf.sin(yaw), tf.cos(yaw), tf.sin(pitch), tf.cos(pitch)], axis=2)
    return cameras


def main():
    # Specify dataset name
    parser = argparse.ArgumentParser(description="Convert tfrecord to torch")
    parser.add_argument("--dataset", type=str,
                        default="shepard_metzler_5_parts",
                        help="Dataset name.")
    args = parser.parse_args()

    if args.dataset not in _DATASETS:
        raise ValueError(f"Unrecognized dataset name {args.dataset}. ",
                         f"Available datasets are {_DATASETS.keys()}.")

    # Path
    root = pathlib.Path("./data/")
    tf_train_dir = root / f"{args.dataset}/train/"
    tf_test_dir = root / f"{args.dataset}/test/"
    torch_train_dir = root / f"{args.dataset}_torch/train/"
    torch_test_dir = root / f"{args.dataset}_torch/test/"

    torch_train_dir.mkdir(parents=True, exist_ok=True)
    torch_test_dir.mkdir(parents=True, exist_ok=True)

    if not tf_train_dir.exists() or not tf_test_dir.exists():
        raise FileNotFoundError("TFRecord path does not exists. ",
                                f"train: {tf_train_dir}, test: {tf_test_dir}.")

    # Process
    run_loop(args.dataset, tf_train_dir, torch_train_dir)
    run_loop(args.dataset, tf_test_dir, torch_test_dir)


if __name__ == "__main__":
    main()
