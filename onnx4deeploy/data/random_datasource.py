# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Random DataSource — preserves the original behaviour (numpy random arrays)."""

from typing import List, Tuple

import numpy as np

from .base_datasource import DataSource


class RandomDataSource(DataSource):
    """Generates random Gaussian inputs and uniform-random integer labels."""

    def load_batches(
        self,
        n_batches: int,
        input_shape: Tuple[int, ...],
        num_classes: int,
        seed: int = 42,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        np.random.seed(seed)
        batch_size = input_shape[0]
        inputs = [np.random.randn(*input_shape).astype(np.float32) for _ in range(n_batches)]
        labels = [
            np.random.randint(0, num_classes, size=(batch_size,)).astype(np.int64)
            for _ in range(n_batches)
        ]
        return inputs, labels
