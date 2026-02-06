"""Split operator test implementation."""

from typing import Any, Dict, List, Tuple

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class SplitOperatorTest(BaseOperatorTest):
    """Test generator for ONNX Split operator."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = None
        self.axis = 0
        self.num_outputs = 2

    def get_operator_name(self) -> str:
        return "Split"

    def load_config(self) -> Dict[str, Any]:
        """Load Split-specific configuration."""
        config = super().load_config()

        split_config = config.get("split", {})
        self.input_shape = tuple(split_config["input_shape"])
        self.axis = split_config.get("axis", 0)
        self.num_outputs = split_config.get("num_outputs", 2)

        # Validate that the dimension is divisible by num_outputs
        if self.input_shape[self.axis] % self.num_outputs != 0:
            raise ValueError(
                f"Dimension {self.input_shape[self.axis]} at axis {self.axis} "
                f"is not evenly divisible by num_outputs={self.num_outputs}"
            )

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random input data."""
        return {
            "input": np.random.randn(*self.input_shape).astype(np.float32) * 0.1,
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for Split operator."""
        # Input tensor
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, self.input_shape)

        # Output shapes (equal splits)
        output_shapes = self._compute_output_shapes()
        output_names = [f"output_{i}" for i in range(self.num_outputs)]

        output_tensors = [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            for name, shape in zip(output_names, output_shapes)
        ]

        # Split node (for equal splits, no 'split' attribute needed in opset 13)
        split_node = helper.make_node(
            "Split", inputs=["input"], outputs=output_names, axis=self.axis, name="split_node"
        )

        # Graph
        graph = helper.make_graph([split_node], "split_graph", [input_tensor], output_tensors)

        return graph

    def _compute_output_shapes(self) -> List[Tuple[int, ...]]:
        """Compute output shapes for each split."""
        split_size = self.input_shape[self.axis] // self.num_outputs
        output_shapes = []

        for i in range(self.num_outputs):
            output_shape = list(self.input_shape)
            output_shape[self.axis] = split_size
            output_shapes.append(tuple(output_shape))

        return output_shapes

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute expected output using NumPy."""
        input_data = inputs["input"]

        # Split along the specified axis
        splits = np.split(input_data, self.num_outputs, axis=self.axis)

        # Create output dictionary
        output_dict = {f"output_{i}": split for i, split in enumerate(splits)}

        return output_dict
