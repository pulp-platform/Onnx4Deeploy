# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""PerturbRademacher operator test implementation."""

from typing import Any, Dict, Tuple

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class PerturbRademacherOperatorTest(BaseOperatorTest):
    """Test generator for ONNX PerturbRademacher operator (custom/training op)."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = None
        self.num_classes = None
        self.batch_size = None

    def get_operator_name(self) -> str:
        return "PerturbRademacher"

    def load_config(self) -> Dict[str, Any]:
        """Load PerturbRademacher-specific configuration."""
        config = super().load_config()

        pn_config = config.get("perturbrademacher", {})
        self.input_shape = tuple(pn_config["input_shape"])
        return config

    
    def generate_inputs(self) -> np.ndarray:
        """Generate input with both positive and negative values."""
        return {"x": np.random.randn(*self.input_shape).astype(np.float32)}
    
    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for PerturbRademacher operator."""
        # Input tensors (without loss_grad for the final model)
        x_tensor = helper.make_tensor_value_info(
            "x", TensorProto.FLOAT, self.input_shape
        )
        # Output tensor
        perturbed_x_tensor = helper.make_tensor_value_info(
            "perturbed_x", TensorProto.FLOAT, self.input_shape
        )

        # PerturbRademacher node (without loss_grad input)
        perturb_node = helper.make_node(
            "PerturbRademacher",
            inputs=["x"],
            outputs=["perturbed_x"],
            seed=42,
            eps=0.01,
            idx=0,
            name="perturb_rademacher_node",
            domain="com.microsoft"
        )

        # Graph
        graph = helper.make_graph(
            [perturb_node],
            "perturb_rademacher_graph",
            [x_tensor],
            [perturbed_x_tensor],
        )

        return graph

    def create_model(self, graph, opset_version: int = 13):
        """Create ONNX model for PerturbRademacher with custom domain."""
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
        # perturbation is built from -1's and 1's
        perturbation = np.random.choice([-1, 1], size=self.input_shape).astype(np.float32)
        perturbed_x = inputs["x"] + perturbation
        
        return {"perturbed_x": perturbed_x}

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Return None to skip validation - this is a custom operator.
        """
        return None
