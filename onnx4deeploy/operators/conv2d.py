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
        self.kernel_size = 3
        self.stride = 1
        self.padding = 0
        self.out_channels = 16
        self.use_bias = True
        self.group = 1
        self.dilation = 1

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
        self.use_bias = conv_config.get("use_bias", True)
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

        # Compute output shape
        effective_kernel = (self.kernel_size - 1) * self.dilation + 1
        output_height = (height + 2 * self.padding - effective_kernel) // self.stride + 1
        output_width = (width + 2 * self.padding - effective_kernel) // self.stride + 1
        output_shape = (batch_size, self.out_channels, output_height, output_width)

        # Create weight and bias
        weight_shape = (
            self.out_channels,
            in_channels // self.group,
            self.kernel_size,
            self.kernel_size,
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

        # Conv node
        conv_node = helper.make_node(
            "Conv",
            inputs=conv_inputs,
            outputs=["output"],
            name="conv_node",
            kernel_shape=[self.kernel_size, self.kernel_size],
            strides=[self.stride, self.stride],
            pads=[self.padding, self.padding, self.padding, self.padding],
            group=self.group,
            dilations=[self.dilation, self.dilation],
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
