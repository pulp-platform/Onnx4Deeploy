# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Concat operator test implementation."""

from typing import Any, Dict, Tuple

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class ConcatOperatorTest(BaseOperatorTest):
    """Test generator for ONNX Concat operator."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shapes = []
        self.axis = 0
        self.num_inputs = 3

    def get_operator_name(self) -> str:
        return "Concat"

    def load_config(self) -> Dict[str, Any]:
        """Load Concat-specific configuration."""
        config = super().load_config()

        concat_config = config.get("concat", {})
        self.axis = concat_config.get("axis", 0)
        self.num_inputs = concat_config.get("num_inputs", 3)

        # Load input shapes
        # Format: input_shapes: [[shape1], [shape2], [shape3]]
        if "input_shapes" in concat_config:
            self.input_shapes = [tuple(shape) for shape in concat_config["input_shapes"]]
        else:
            # If not specified, use base_shape and infer shapes
            base_shape = tuple(concat_config.get("base_shape", [1, 4, 8, 8]))
            self.input_shapes = [base_shape] * self.num_inputs

        # Validate that all shapes match except along the concat axis
        if not self._validate_shapes():
            raise ValueError(
                f"Input shapes must match in all dimensions except axis {self.axis}. "
                f"Got shapes: {self.input_shapes}"
            )

        return config

    def _validate_shapes(self) -> bool:
        """Validate that input shapes can be concatenated along the specified axis."""
        if len(self.input_shapes) < 2:
            return False

        reference_shape = list(self.input_shapes[0])
        for shape in self.input_shapes[1:]:
            if len(shape) != len(reference_shape):
                return False
            for i, (dim1, dim2) in enumerate(zip(reference_shape, shape)):
                if i != self.axis and dim1 != dim2:
                    return False
        return True

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random input data."""
        inputs = {}
        for i, shape in enumerate(self.input_shapes):
            input_name = f"input_{i}"
            inputs[input_name] = np.random.randn(*shape).astype(np.float32) * 0.1
        return inputs

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for Concat operator."""
        # Input tensors
        input_tensors = []
        input_names = []
        for i, shape in enumerate(self.input_shapes):
            input_name = f"input_{i}"
            input_names.append(input_name)
            input_tensors.append(
                helper.make_tensor_value_info(input_name, TensorProto.FLOAT, shape)
            )

        # Output shape (concatenate along axis)
        output_shape = self._compute_output_shape()
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        # Concat node
        concat_node = helper.make_node(
            "Concat", inputs=input_names, outputs=["output"], axis=self.axis, name="concat_node"
        )

        # Graph
        graph = helper.make_graph([concat_node], "concat_graph", input_tensors, [output_tensor])

        return graph

    def _compute_output_shape(self) -> Tuple[int, ...]:
        """Compute output shape after concatenation."""
        output_shape = list(self.input_shapes[0])
        # Sum the dimensions along the concat axis
        concat_dim = sum(shape[self.axis] for shape in self.input_shapes)
        output_shape[self.axis] = concat_dim
        return tuple(output_shape)

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute expected output using NumPy."""
        # Sort inputs by name to ensure correct order (input_0, input_1, input_2, ...)
        sorted_inputs = [inputs[f"input_{i}"] for i in range(len(inputs))]

        # Concatenate along the specified axis
        output = np.concatenate(sorted_inputs, axis=self.axis)

        return {"output": output}
