# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Sub operator test implementation."""

from typing import Any, Dict

import numpy as np

from .base_operator import SimpleElementwiseOperator


class SubOperatorTest(SimpleElementwiseOperator):
    """Test generator for ONNX Sub operator.

    ONNX Specification:
    - Description: Performs element-wise binary subtraction (A - B) with Numpy-style broadcasting
    - Inputs:
        * A (T): First operand
        * B (T): Second operand
    - Outputs: C (T): Result with same element type as inputs
    - Attributes: None
    - Type Constraints: T: tensor(float16), tensor(float), tensor(double),
                        tensor(int8), tensor(int16), tensor(int32), tensor(int64),
                        tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64),
                        tensor(bfloat16)
    - Opset Version: 14 (current)
    - Reference: https://onnx.ai/onnx/operators/onnx__Sub.html
    """

    def get_operator_name(self) -> str:
        """Return ONNX operator name."""
        return "Sub"

    def get_config_key(self) -> str:
        """Return config key for YAML."""
        return "sub"

    def load_config(self) -> Dict[str, Any]:
        """Load configuration and set up for two inputs."""
        config = super().load_config()

        # Override: Sub always has two inputs of the same shape
        op_config = config.get(self.get_config_key(), {})

        if "input_shape" in op_config:
            shape = tuple(op_config["input_shape"])
            self.input_shapes = [shape, shape]
            self.input_names = ["input_a", "input_b"]

        return config

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute expected output using NumPy.

        Args:
            inputs: Dictionary of input tensors

        Returns:
            Dictionary of output tensors
        """
        input_a = inputs["input_a"]
        input_b = inputs["input_b"]
        return {"output": input_a - input_b}
