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
        # Use a local RandomState so we do NOT touch the global numpy seed.
        rng = np.random.RandomState(seed)
        batch_size = input_shape[0]
        inputs = [rng.randn(*input_shape).astype(np.float32) for _ in range(n_batches)]
        labels = [
            rng.randint(0, num_classes, size=(batch_size,)).astype(np.int64)
            for _ in range(n_batches)
        ]
        return inputs, labels
