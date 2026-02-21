# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""PerturbEggroll operator test implementation."""

from typing import Any, Dict, Tuple

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class PerturbEggrollOperatorTest(BaseOperatorTest):
    """Test generator for ONNX PerturbEggroll operator (custom/training op)."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = None
        self.num_classes = None
        self.batch_size = None

    def get_operator_name(self) -> str:
        return "PerturbEggroll"

    def load_config(self) -> Dict[str, Any]:
        """Load PerturbEggroll-specific configuration."""
        config = super().load_config()

        pn_config = config.get("perturbeggroll", {})
        self.input_shape = tuple(pn_config["input_shape"])
        return config


    def generate_inputs(self) -> np.ndarray:
        """Generate input with both positive and negative values."""
        return {"x": np.random.randn(*self.input_shape).astype(np.float32)}

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for PerturbEggroll operator."""
        # Input tensors (without loss_grad for the final model)
        x_tensor = helper.make_tensor_value_info(
            "x", TensorProto.FLOAT, self.input_shape
        )
        # Output tensor
        perturbed_x_tensor = helper.make_tensor_value_info(
            "perturbed_x", TensorProto.FLOAT, self.input_shape
        )

        epsilon = 0.01  # Example scaling factor for the perturbation

        # Shape annotation for intermediate outputs
        a_shape = [self.input_shape[0], 1]
        b_shape = [int(np.prod(self.input_shape[1:])), 1]

        a_tensor = helper.make_tensor_value_info(
            "a", TensorProto.FLOAT, a_shape
        )
        b_tensor = helper.make_tensor_value_info(
            "b", TensorProto.FLOAT, b_shape
        )

        # Eggroll noise node (without loss_grad input)
        noise_node = helper.make_node(
            "GenerateEggrollNoise",
            inputs=["x"],
            outputs=["a", "b"],
            name="generate_eggroll_noise_node",
            domain="com.microsoft"
        )

        gemm_node = helper.make_node(
            "Gemm",
            inputs=["a", "b", "x"],
            outputs=["perturbed_x"],
            name="gemm_node",
            transA=0,
            transB=1,
            alpha=epsilon,
            beta=0
        )

        # Graph
        graph = helper.make_graph(
            [noise_node, gemm_node],
            "perturb_eggroll_graph",
            [x_tensor],
            [perturbed_x_tensor],
            value_info=[a_tensor, b_tensor]  # <-- shape annotation here
        )

        for vi in graph.value_info:
            print(vi.name, [d.dim_value for d in vi.type.tensor_type.shape.dim])
        return graph

    def create_model(self, graph, opset_version: int = 13):
        """Create ONNX model for PerturbEggroll with custom domain."""
        model = helper.make_model(
            graph,
            producer_name=f"{self.get_operator_name().lower()}_test",
            opset_imports=[
                helper.make_opsetid("", opset_version),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )

        return model

    def run_inference(self, onnx_file: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference using custom emulation
        """
        # perturbation is built from 2 low rank tensors
        a = np.random.randn(*self.input_shape[:-1]).astype(np.float32)
        b = np.random.randn(self.input_shape[-1]).astype(np.float32)
        perturbation = np.outer(a, b)  # shape: input_shape
        perturbed_x = inputs["x"] + perturbation

        return {"perturbed_x": perturbed_x}

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Return None to skip validation - this is a custom operator.
        """
        return None
