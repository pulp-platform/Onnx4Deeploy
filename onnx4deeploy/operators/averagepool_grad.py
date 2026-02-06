"""AveragePoolGrad operator test implementation."""

from typing import Any, Dict

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class AveragePoolGradOperatorTest(BaseOperatorTest):
    """Test generator for ONNX AveragePoolGrad operator (custom/training op)."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_grad_shape = None
        self.original_input_shape = None
        self.kernel_shape = None
        self.strides = None
        self.pads = None
        self.count_include_pad = 0
        self.ceil_mode = 0

    def get_operator_name(self) -> str:
        return "AveragePoolGrad"

    def load_config(self) -> Dict[str, Any]:
        """Load AveragePoolGrad-specific configuration."""
        config = super().load_config()

        averagepoolgrad_config = config.get("averagepoolgrad", {})
        self.input_grad_shape = tuple(averagepoolgrad_config["input_grad_shape"])
        self.original_input_shape = tuple(averagepoolgrad_config["original_input_shape"])
        self.kernel_shape = tuple(averagepoolgrad_config.get("kernel_shape", [2, 2]))
        self.strides = tuple(averagepoolgrad_config.get("strides", [2, 2]))
        self.pads = tuple(averagepoolgrad_config.get("pads", [0, 0, 0, 0]))
        self.count_include_pad = int(averagepoolgrad_config.get("count_include_pad", 0))
        self.ceil_mode = int(averagepoolgrad_config.get("ceil_mode", 0))

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random gradient data."""
        return {
            "input_grad": np.random.randn(*self.input_grad_shape).astype(np.float32),
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for AveragePoolGrad operator."""
        # Input tensor (gradient from downstream)
        input_grad_tensor = helper.make_tensor_value_info(
            "input_grad", TensorProto.FLOAT, self.input_grad_shape
        )

        # Output tensor (gradient to upstream, same shape as original input)
        output_grad_tensor = helper.make_tensor_value_info(
            "output_grad", TensorProto.FLOAT, self.original_input_shape
        )

        # AveragePoolGrad node
        averagepoolgrad_node = helper.make_node(
            "AveragePoolGrad",
            inputs=["input_grad"],
            outputs=["output_grad"],
            kernel_shape=self.kernel_shape,
            strides=self.strides,
            pads=self.pads,
            count_include_pad=self.count_include_pad,
            ceil_mode=self.ceil_mode,
            name="averagepoolgrad_node",
        )

        # Graph
        graph = helper.make_graph(
            [averagepoolgrad_node],
            "averagepoolgrad_graph",
            [input_grad_tensor],
            [output_grad_tensor],
        )

        return graph

    def create_model(self, graph, opset_version: int = 13):
        """Create ONNX model for AveragePoolGrad."""
        model = helper.make_model(
            graph,
            producer_name=f"{self.get_operator_name().lower()}_test",
            opset_imports=[helper.make_opsetid("", opset_version)],
        )

        # Manually set output shape for custom operators
        output_tensor = model.graph.output[0]
        # Clear existing dimensions
        del output_tensor.type.tensor_type.shape.dim[:]
        # Add dimensions with correct values
        for dim in self.original_input_shape:
            output_tensor.type.tensor_type.shape.dim.add().dim_value = dim

        return model

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute expected output using NumPy.

        Return None to skip validation - this is a custom operator.
        """
        return None
