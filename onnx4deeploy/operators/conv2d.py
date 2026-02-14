# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Conv2D operator test implementation."""

from typing import Any, Dict

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class Conv2DOperatorTest(BaseOperatorTest):
    """Test generator for ONNX Conv operator."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = None
        self.kernel_size = 3  # Can be int or list [h, w]
        self.stride = 1  # Can be int or list [h, w]
        self.padding = 0  # Can be int or list [top, left, bottom, right]
        self.out_channels = 16
        self.use_bias = False
        self.group = 1
        self.dilation = 1  # Can be int or list [h, w]

    def get_operator_name(self) -> str:
        return "Conv"

    def load_config(self) -> Dict[str, Any]:
        """Load Conv2D-specific configuration."""
        config = super().load_config()

        conv_config = config.get("conv2d", {})
        self.input_shape = tuple(conv_config["input_shape"])
        self.kernel_size = conv_config.get("kernel_size", 3)
        self.stride = conv_config.get("stride", 1)
        self.padding = conv_config.get("padding", 0)
        self.out_channels = conv_config.get("out_channels", 16)
        self.use_bias = conv_config.get("use_bias", False)
        self.group = conv_config.get("group", 1)
        self.dilation = conv_config.get("dilation", 1)

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random input data."""
        return {
            "input": np.random.randn(*self.input_shape).astype(np.float32),
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for Conv operator."""
        batch_size, in_channels, height, width = self.input_shape

        # Handle kernel_size: can be int or list [h, w]
        if isinstance(self.kernel_size, (list, tuple)):
            kernel_h, kernel_w = self.kernel_size
        else:
            kernel_h = kernel_w = self.kernel_size

        # Handle stride: can be int or list [h, w]
        if isinstance(self.stride, (list, tuple)):
            stride_h, stride_w = self.stride
        else:
            stride_h = stride_w = self.stride

        # Handle dilation: can be int or list [h, w]
        if isinstance(self.dilation, (list, tuple)):
            dilation_h, dilation_w = self.dilation
        else:
            dilation_h = dilation_w = self.dilation

        # Handle padding: can be int or list [top, left, bottom, right]
        if isinstance(self.padding, (list, tuple)):
            pad_top, pad_left, pad_bottom, pad_right = self.padding
        else:
            pad_top = pad_left = pad_bottom = pad_right = self.padding

        # Compute output shape
        effective_kernel_h = (kernel_h - 1) * dilation_h + 1
        effective_kernel_w = (kernel_w - 1) * dilation_w + 1
        output_height = (height + pad_top + pad_bottom - effective_kernel_h) // stride_h + 1
        output_width = (width + pad_left + pad_right - effective_kernel_w) // stride_w + 1
        output_shape = (batch_size, self.out_channels, output_height, output_width)

        # Create weight and bias
        weight_shape = (
            self.out_channels,
            in_channels // self.group,
            kernel_h,
            kernel_w,
        )
        weight = np.random.randn(*weight_shape).astype(np.float32)

        # Input tensor
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, self.input_shape)

        # Output tensor
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        # Weight as initializer
        weight_tensor = helper.make_tensor(
            "weight", TensorProto.FLOAT, weight_shape, weight.flatten()
        )

        initializers = [weight_tensor]

        # Include bias if enabled
        if self.use_bias:
            bias = np.random.randn(self.out_channels).astype(np.float32)
            bias_tensor = helper.make_tensor("bias", TensorProto.FLOAT, bias.shape, bias.flatten())
            initializers.append(bias_tensor)
            conv_inputs = ["input", "weight", "bias"]
        else:
            conv_inputs = ["input", "weight"]

        # Conv node - padding format is [top, left, bottom, right]
        if isinstance(self.padding, (list, tuple)):
            pads_list = list(self.padding)
        else:
            pads_list = [self.padding, self.padding, self.padding, self.padding]

        conv_node = helper.make_node(
            "Conv",
            inputs=conv_inputs,
            outputs=["output"],
            name="conv_node",
            kernel_shape=[kernel_h, kernel_w],
            strides=[stride_h, stride_w],
            pads=pads_list,
            group=self.group,
            dilations=[dilation_h, dilation_w],
        )

        # Graph
        graph = helper.make_graph(
            [conv_node], "conv_graph", [input_tensor], [output_tensor], initializer=initializers
        )

        return graph

    def create_model(self, graph, opset_version: int = 21):
        """Create ONNX model for Conv (use opset 21 like the legacy test)."""
        model = helper.make_model(
            graph,
            producer_name=f"{self.get_operator_name().lower()}_test",
            opset_imports=[helper.make_opsetid("", opset_version)],
        )

        return model

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Return None to skip validation - we use ONNX Runtime as ground truth.
        """
        return None
