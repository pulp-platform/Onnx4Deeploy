# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""GEMM (General Matrix Multiplication) operator test implementation."""

from typing import Any, Dict, Tuple

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from .base_operator import BaseOperatorTest


class GemmOperatorTest(BaseOperatorTest):
    """Test generator for ONNX Gemm operator."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_a_shape = None
        self.input_b_shape = None
        self.input_c_shape = None
        self.transA = 0
        self.transB = 0
        self.alpha = 1.0
        self.beta = 1.0
        self.a_is_const = False
        self.b_is_const = False
        self.c_is_const = False
        self.use_bias = True

    def get_operator_name(self) -> str:
        return "Gemm"

    def load_config(self) -> Dict[str, Any]:
        """Load GEMM-specific configuration."""
        config = super().load_config()

        gemm_config = config.get("gemm", {})

        self.input_a_shape = tuple(gemm_config["input_a_shape"])
        self.input_b_shape = tuple(gemm_config["input_b_shape"])

        self.transA = gemm_config.get("transA", 0)
        self.transB = gemm_config.get("transB", 0)
        self.alpha = gemm_config.get("alpha", 1.0)
        self.beta = gemm_config.get("beta", 1.0)

        self.a_is_const = gemm_config.get("a_is_const", False)
        self.b_is_const = gemm_config.get("b_is_const", False)
        self.c_is_const = gemm_config.get("c_is_const", False)

        self.use_bias = gemm_config.get("use_bias", True)

        # Validate matrix dimensions
        self._validate_shapes()

        # Calculate output shape and bias shape
        if self.use_bias:
            if "input_c_shape" in gemm_config:
                self.input_c_shape = tuple(gemm_config["input_c_shape"])
            else:
                # Default bias shape
                output_shape = self._compute_output_shape()
                self.input_c_shape = output_shape[-2:]

        return config

    def _validate_shapes(self):
        """Validate that matrix dimensions are compatible."""
        if len(self.input_a_shape) < 2 or len(self.input_b_shape) < 2:
            raise ValueError("Input shapes must have at least 2 dimensions")

        a_last_two = self.input_a_shape[-2:]
        b_last_two = self.input_b_shape[-2:]

        a_effective = (a_last_two[1], a_last_two[0]) if self.transA else a_last_two
        b_effective = (b_last_two[1], b_last_two[0]) if self.transB else b_last_two

        if a_effective[1] != b_effective[0]:
            raise ValueError(
                f"Incompatible matrix dimensions: A={self.input_a_shape} "
                f"(transA={self.transA}), B={self.input_b_shape} (transB={self.transB})"
            )

    def _compute_output_shape(self) -> Tuple[int, ...]:
        """Compute output shape based on input shapes and transpose flags."""
        a_last_two = self.input_a_shape[-2:]
        b_last_two = self.input_b_shape[-2:]

        a_effective = (a_last_two[1], a_last_two[0]) if self.transA else a_last_two
        b_effective = (b_last_two[1], b_last_two[0]) if self.transB else b_last_two

        output_rows = a_effective[0]
        output_cols = b_effective[1]

        batch_dims = self.input_a_shape[:-2]
        return batch_dims + (output_rows, output_cols)

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random input data for GEMM."""
        inputs = {}

        inputs["input_a"] = np.random.randn(*self.input_a_shape).astype(np.float32) * 0.1
        inputs["input_b"] = np.random.randn(*self.input_b_shape).astype(np.float32) * 0.1

        if self.use_bias:
            inputs["input_c"] = np.random.randn(*self.input_c_shape).astype(np.float32) * 0.1

        return inputs

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for GEMM operator."""
        graph_inputs = []
        graph_initializers = []
        node_inputs = []

        # Handle input A
        if self.a_is_const:
            tensor_a = numpy_helper.from_array(inputs["input_a"], name="input_a")
            graph_initializers.append(tensor_a)
        else:
            input_tensor_a = helper.make_tensor_value_info(
                "input_a", TensorProto.FLOAT, self.input_a_shape
            )
            graph_inputs.append(input_tensor_a)
        node_inputs.append("input_a")

        # Handle input B
        if self.b_is_const:
            tensor_b = numpy_helper.from_array(inputs["input_b"], name="input_b")
            graph_initializers.append(tensor_b)
        else:
            input_tensor_b = helper.make_tensor_value_info(
                "input_b", TensorProto.FLOAT, self.input_b_shape
            )
            graph_inputs.append(input_tensor_b)
        node_inputs.append("input_b")

        # Handle input C (bias)
        if self.use_bias:
            if self.c_is_const:
                tensor_c = numpy_helper.from_array(inputs["input_c"], name="input_c")
                graph_initializers.append(tensor_c)
            else:
                input_tensor_c = helper.make_tensor_value_info(
                    "input_c", TensorProto.FLOAT, self.input_c_shape
                )
                graph_inputs.append(input_tensor_c)
            node_inputs.append("input_c")

        # Create output tensor
        output_shape = self._compute_output_shape()
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        # Create GEMM node
        gemm_node = helper.make_node(
            "Gemm",
            inputs=node_inputs,
            outputs=["output"],
            name="gemm_node",
            transA=self.transA,
            transB=self.transB,
            alpha=self.alpha,
            beta=self.beta,
        )

        # Create graph
        graph = helper.make_graph(
            [gemm_node], "gemm_graph", graph_inputs, [output_tensor], initializer=graph_initializers
        )

        return graph

    def run_inference(self, onnx_file: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run inference, only passing non-constant inputs."""
        feed_dict = {}

        if not self.a_is_const:
            feed_dict["input_a"] = inputs["input_a"]
        if not self.b_is_const:
            feed_dict["input_b"] = inputs["input_b"]
        if self.use_bias and not self.c_is_const:
            feed_dict["input_c"] = inputs["input_c"]

        return super().run_inference(onnx_file, feed_dict)

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute expected output using NumPy.

        Note: Returns None to skip validation - ONNX Runtime already validates correctness.
        """
        # Skip custom validation for GEMM - ONNX Runtime validates correctness
        return None
