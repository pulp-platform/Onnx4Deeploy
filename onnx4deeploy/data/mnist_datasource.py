# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""MNIST DataSource — loads real MNIST images as training data."""

import gzip
import os
import struct
import urllib.request
from typing import List, Optional, Tuple

import numpy as np

from .base_datasource import DataSource

# Official MNIST mirror
_MNIST_URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}


def _load_mnist_numpy(data_path: str, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MNIST from local IDX binary files (downloaded if missing).

    Returns:
        images: float32 array of shape (N, 784), values in [0, 1].
        labels: int64 array of shape (N,).
    """
    prefix = "train" if split == "train" else "test"
    img_key = f"{prefix}_images"
    lbl_key = f"{prefix}_labels"

    img_file = os.path.join(data_path, f"{prefix}-images-idx3-ubyte")
    lbl_file = os.path.join(data_path, f"{prefix}-labels-idx1-ubyte")

    os.makedirs(data_path, exist_ok=True)

    for local_path, url_key in [(img_file, img_key), (lbl_file, lbl_key)]:
        if not os.path.exists(local_path):
            gz_path = local_path + ".gz"
            print(f"   Downloading MNIST {url_key} → {gz_path} ...")
            urllib.request.urlretrieve(_MNIST_URLS[url_key], gz_path)
            with gzip.open(gz_path, "rb") as f_in, open(local_path, "wb") as f_out:
                f_out.write(f_in.read())
            os.remove(gz_path)

    # Read images (IDX3 format)
    with open(img_file, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
    images = images.astype(np.float32) / 255.0  # normalise to [0, 1]

    # Read labels (IDX1 format)
    with open(lbl_file, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)

    return images, labels


def _resize_flat(flat_img: np.ndarray, target_size: int) -> np.ndarray:
    """
    Linearly resample a flat image vector to target_size elements.
    Used when the model input_size differs from 784 (e.g. 64 for 8×8 MLP).
    """
    if flat_img.shape[0] == target_size:
        return flat_img
    x_old = np.linspace(0.0, 1.0, flat_img.shape[0])
    x_new = np.linspace(0.0, 1.0, target_size)
    return np.interp(x_new, x_old, flat_img).astype(np.float32)


class MNISTDataSource(DataSource):
    """
    Loads real MNIST images as training mini-batches.

    Files are downloaded automatically to `data_path` if not present.
    Images are normalised to [0, 1] and flattened (or resampled) to match
    the model's input_shape.

    Args:
        data_path: Directory for storing / reading MNIST binary files.
                   Defaults to ``/tmp/mnist``.
        split:     ``"train"`` (60 000 samples) or ``"test"`` (10 000 samples).
    """

    def __init__(self, data_path: Optional[str] = None, split: str = "train"):
        self.data_path = data_path or "/tmp/mnist"
        self.split = split
        self._images: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None

    def _ensure_loaded(self) -> None:
        if self._images is None:
            print(f"   Loading MNIST ({self.split}) from {self.data_path} ...")
            self._images, self._labels = _load_mnist_numpy(self.data_path, self.split)
            print(f"   MNIST loaded: {self._images.shape[0]} samples")

    def load_batches(
        self,
        n_batches: int,
        input_shape: Tuple[int, ...],
        num_classes: int,
        seed: int = 42,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        self._ensure_loaded()

        batch_size = input_shape[0]
        flat_size = int(np.prod(input_shape[1:]))  # e.g. 784 for (1, 784)

        np.random.seed(seed)
        total_needed = n_batches * batch_size
        indices = np.random.randint(0, len(self._images), total_needed)

        inputs_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []

        for b in range(n_batches):
            idx = indices[b * batch_size : (b + 1) * batch_size]
            imgs = np.stack(
                [_resize_flat(self._images[i], flat_size) for i in idx], axis=0
            )  # (batch_size, flat_size)
            imgs = imgs.reshape(input_shape)
            lbls = self._labels[idx].astype(np.int64)
            inputs_list.append(imgs)
            labels_list.append(lbls)

        return inputs_list, labels_list
