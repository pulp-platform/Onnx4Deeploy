# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Base DataSource abstraction for training test data generation."""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class DataSource(ABC):
    """
    Abstract base class for training data sources.

    Implementations return (inputs, labels) lists for n mini-batches,
    decoupling data loading from the gradient-accumulation / SGD simulation
    logic in create_training_test_data().
    """

    @abstractmethod
    def load_batches(
        self,
        n_batches: int,
        input_shape: Tuple[int, ...],
        num_classes: int,
        seed: int = 42,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Return n_batches of (input, label) pairs.

        Args:
            n_batches:   Number of mini-batches to produce.
            input_shape: Shape of each input tensor, e.g. (batch_size, 784).
            num_classes: Number of output classes (used for label range).
            seed:        Random seed for reproducibility.

        Returns:
            inputs_list: List of n_batches float32 arrays, each shape == input_shape.
            labels_list: List of n_batches int64 arrays, each shape == (batch_size,).
        """
