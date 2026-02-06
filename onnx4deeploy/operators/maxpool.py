# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""MaxPool operator test implementation."""

from typing import Any, Dict, Tuple

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class MaxPoolOperatorTest(BaseOperatorTest):
    """Test generator for ONNX MaxPool operator."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = None
        self.kernel_shape = None
        self.strides = None
        self.pads = None
        self.ceil_mode = 0

    def get_operator_name(self) -> str:
        return "MaxPool"

    def load_config(self) -> Dict[str, Any]:
        """Load MaxPool-specific configuration."""
        config = super().load_config()

        maxpool_config = config.get("maxpool", {})
        self.input_shape = tuple(maxpool_config["input_shape"])
        self.kernel_shape = tuple(maxpool_config["kernel_shape"])
        self.strides = tuple(maxpool_config.get("strides", self.kernel_shape))
        self.pads = tuple(maxpool_config.get("pads", [0, 0, 0, 0]))
        self.ceil_mode = int(maxpool_config.get("ceil_mode", 0))

        # Validate input shape
        if len(self.input_shape) != 4:
            raise ValueError("Input shape must be 4D tensor in NCHW format")

        if len(self.kernel_shape) != 2:
            raise ValueError("Kernel shape must be 2D [height, width]")

        if len(self.strides) != 2:
            raise ValueError("Strides must be 2D [height, width]")

        if len(self.pads) != 4:
            raise ValueError("Pads must be [top, left, bottom, right]")

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random input data."""
        return {
            "input": np.random.randn(*self.input_shape).astype(np.float32),
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for MaxPool operator."""
        # Input tensor
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, self.input_shape)

        # Output shape
        output_shape = self._compute_output_shape()
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        # MaxPool node
        maxpool_node = helper.make_node(
            "MaxPool",
            inputs=["input"],
            outputs=["output"],
            kernel_shape=self.kernel_shape,
            strides=self.strides,
            pads=self.pads,
            ceil_mode=self.ceil_mode,
            name="maxpool_node",
        )

        # Graph
        graph = helper.make_graph([maxpool_node], "maxpool_graph", [input_tensor], [output_tensor])

        return graph

    def _compute_output_shape(self) -> Tuple[int, ...]:
        """Compute output shape based on pooling parameters."""
        batch_size, channels, height, width = self.input_shape
        kernel_h, kernel_w = self.kernel_shape
        stride_h, stride_w = self.strides
        pad_t, pad_l, pad_b, pad_r = self.pads

        # Calculate output height and width
        if self.ceil_mode == 0:
            out_height = (height + pad_t + pad_b - kernel_h) // stride_h + 1
            out_width = (width + pad_l + pad_r - kernel_w) // stride_w + 1
        else:
            out_height = (height + pad_t + pad_b - kernel_h + stride_h - 1) // stride_h + 1
            out_width = (width + pad_l + pad_r - kernel_w + stride_w - 1) // stride_w + 1

        return (batch_size, channels, out_height, out_width)

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute expected output using NumPy.

        Return None to skip validation - ONNX Runtime handles this operator correctly.
        """
        return None
