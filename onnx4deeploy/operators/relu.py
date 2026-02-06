# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""ReLU operator test implementation."""

from typing import Dict, Tuple

import numpy as np

from .base_operator import SimpleElementwiseOperator


class ReluOperatorTest(SimpleElementwiseOperator):
    """Test generator for ONNX Relu operator."""

    def get_operator_name(self) -> str:
        return "Relu"

    def get_config_key(self) -> str:
        return "relu"

    def generate_single_input(self, name: str, shape: Tuple[int, ...]) -> np.ndarray:
        """Generate input with both positive and negative values."""
        return np.random.uniform(-1, 1, shape).astype(np.float32)

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute expected output using NumPy."""
        input_data = inputs["input"]
        return {"output": np.maximum(0, input_data)}
