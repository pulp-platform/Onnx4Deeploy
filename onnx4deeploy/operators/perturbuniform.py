# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""PerturbUniform operator test implementation."""

from typing import Any, Dict, Tuple

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class PerturbUniformOperatorTest(BaseOperatorTest):
    """Test generator for ONNX PerturbUniform operator (custom/training op)."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = None
        self.num_classes = None
        self.batch_size = None

    def get_operator_name(self) -> str:
        return "PerturbUniform"

    def load_config(self) -> Dict[str, Any]:
        """Load PerturbUniform-specific configuration."""
        config = super().load_config()

        pn_config = config.get("perturbuniform", {})
        self.input_shape = tuple(pn_config["input_shape"])
        return config

    
    def generate_inputs(self) -> np.ndarray:
        """Generate input with both positive and negative values."""
        return {"x": np.random.randn(*self.input_shape).astype(np.float32)}
    
    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for PerturbUniform operator."""
        # Input tensors (without loss_grad for the final model)
        x_tensor = helper.make_tensor_value_info(
            "x", TensorProto.FLOAT, self.input_shape
        )
        # Output tensor
        perturbed_x_tensor = helper.make_tensor_value_info(
            "perturbed_x", TensorProto.FLOAT, self.input_shape
        )

        # PerturbUniform node (without loss_grad input)
        perturb_node = helper.make_node(
            "PerturbUniform",
            inputs=["x"],
            outputs=["perturbed_x"],
            name="perturb_uniform_node",
            domain="com.microsoft",
            reduction="mean",
        )

        # Graph
        graph = helper.make_graph(
            [perturb_node],
            "perturb_uniform_graph",
            [x_tensor],
            [perturbed_x_tensor],
        )

        return graph

    def create_model(self, graph, opset_version: int = 13):
        """Create ONNX model for PerturbUniform with custom domain."""
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
        Run inference using ONNX Runtime.

        For this custom op, we build a separate model that implements the
        PerturbUniform functionality using standard ONNX ops (RandomUniform + Add)
        and run inference on that to get the output.
        """
        # --- Create the "Execution" Graph ---
        # This graph implements the behavior of PerturbUniform for testing.

        # Input tensor info
        x_tensor = helper.make_tensor_value_info(
            "x", TensorProto.FLOAT, self.input_shape
        )
        # Output tensor info
        perturbed_x_tensor = helper.make_tensor_value_info(
            "perturbed_x", TensorProto.FLOAT, self.input_shape
        )

        # Intermediate tensor for the random noise
        noise_tensor_name = "random_noise"

        # 1. RandomUniform node to generate noise
        # The shape of the noise must match the input shape.
        random_node = helper.make_node(
            "RandomUniform",
            inputs=[],  # RandomUniform has no inputs
            outputs=[noise_tensor_name],
            name="random_uniform_for_perturb",
            shape=self.input_shape,
            dtype=TensorProto.FLOAT,
            low = -np.sqrt(3),
            high = np.sqrt(3)
        )

        # 2. Add node to add the noise to the input
        add_node = helper.make_node(
            "Add",
            inputs=["x", noise_tensor_name],
            outputs=["perturbed_x"],
            name="add_perturbation",
        )

        # Create the graph that implements the custom op's logic
        execution_graph = helper.make_graph(
            [random_node, add_node],
            "perturb_uniform_execution_graph",
            [x_tensor],
            [perturbed_x_tensor],
        )

        # Create the ONNX model for execution
        execution_model = self.create_model(execution_graph)

        # Run inference on the execution model
        sess_options = ort.SessionOptions()
        # Disable all optimizations to ensure nodes are not fused or altered
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        session = ort.InferenceSession(execution_model.SerializeToString(), sess_options)
        # The output name is "perturbed_x"
        output_names = ["perturbed_x"]
        outputs = session.run(output_names, inputs)

        # Return the output in the expected dictionary format
        return {"perturbed_x": outputs[0]}

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Return None to skip validation - this is a custom operator.
        """
        return None
