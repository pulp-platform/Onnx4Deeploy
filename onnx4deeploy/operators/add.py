# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Add operator test implementation."""

from typing import Any, Dict

import numpy as np

from .base_operator import SimpleElementwiseOperator


class AddOperatorTest(SimpleElementwiseOperator):
    """Test generator for ONNX Add operator."""

    def get_operator_name(self) -> str:
        return "Add"

    def get_config_key(self) -> str:
        return "adder"

    def load_config(self) -> Dict[str, Any]:
        """Load configuration and set up for two inputs."""
        config = super().load_config()

        # Override: Add always has two inputs of the same shape
        op_config = config.get(self.get_config_key(), {})

        if "input_shape" in op_config:
            shape = tuple(op_config["input_shape"])
            self.input_shapes = [shape, shape]
            self.input_names = ["input_a", "input_b"]

        return config

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute expected output using NumPy."""
        input_a = inputs["input_a"]
        input_b = inputs["input_b"]
        return {"output": input_a + input_b}
