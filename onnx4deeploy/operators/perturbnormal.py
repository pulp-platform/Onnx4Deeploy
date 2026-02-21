# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""PerturbNormal operator test implementation."""

from typing import Any, Dict, Tuple


import torch
from torch.autograd import Function
import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class Xorshift32:
    def __init__(self, seed: int = 0):
        self.state = seed if seed != 0 else 1  # Avoid zero state

    def next(self) -> int:
        # Xorshift32 algorithm
        self.state ^= (self.state << 13) & 0xFFFFFFFF
        self.state ^= (self.state >> 17) & 0xFFFFFFFF
        self.state ^= (self.state << 5) & 0xFFFFFFFF
        return self.state
    
class Ziggurat():
    def __init__(self, seed: int = 0):
        self.seed = seed if seed != 0 else 1  # Avoid zero state
        # Precompute the Ziggurat tables
        self.N = 256  # Number of layers
        self.R = 3.442619855899  # Right tail boundary
        self.x = np.zeros(self.N + 1)
        self.y = np.zeros(self.N)
        self.x[0] = self.R
        self.x[self.N] = 0
        for i in range(1, self.N):
            self.x[i] = np.sqrt(-2.0 * np.log(np.exp(-0.5 * self.x[i-1]**2)))
        for i in range(self.N):
            self.y[i] = np.exp(-0.5 * self.x[i]**2)
        self.rng = Xorshift32(self.seed)

    def next(self) -> float:
        while True:
            # Generate random layer index
            k = self.rng.next() % self.N
            # Generate uniform random number
            u = self.rng.next() / 0xFFFFFFFF
            x = u * (self.x[k] - self.x[k+1]) + self.x[k+1]
            # Accept or reject
            if u < self.y[k] / self.y[k+1]:
                return x
            if x < self.R:
                y = np.exp(-0.5 * x * x)
                if u * (self.y[k+1] - self.y[k]) < (y - self.y[k]):
                    return x

class PerturbNormalFunction(Function):
    @staticmethod
    def forward(ctx, x, seed=42, epsilon=0.01):
        # generate noise using Xorshift.
        rng = Ziggurat(seed)
        for _ in range(x.numel()):
            noise = rng.next() * epsilon
        perturbed_x = x + noise
        return perturbed_x
    
    @staticmethod
    def symbolic(g, x):
        return g.op("ai.zo::PerturbNormal", x, outputs=1)

class PerturbNormalOperatorTest(BaseOperatorTest):
    """Test generator for ONNX PerturbNormal operator (custom/training op)."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = None
        self.num_classes = None
        self.batch_size = None

    def get_operator_name(self) -> str:
        return "PerturbNormal"

    def load_config(self) -> Dict[str, Any]:
        """Load PerturbNormal-specific configuration."""
        config = super().load_config()

        pn_config = config.get("perturbnormal", {})
        self.input_shape = tuple(pn_config["input_shape"])
        return config

    
    def generate_inputs(self) -> np.ndarray:
        """Generate input with both positive and negative values."""
        return {"x": np.random.randn(*self.input_shape).astype(np.float32)}
    
    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for PerturbNormal operator."""
        # Input tensors (without loss_grad for the final model)
        x_tensor = helper.make_tensor_value_info(
            "x", TensorProto.FLOAT, self.input_shape
        )
        # Output tensor
        perturbed_x_tensor = helper.make_tensor_value_info(
            "perturbed_x", TensorProto.FLOAT, self.input_shape
        )

        # PerturbNormal node (without loss_grad input)
        perturb_node = helper.make_node(
            "PerturbNormal",
            inputs=["x"],
            outputs=["perturbed_x"],
            name="perturb_normal_node",
            seed=42,
            eps=0.01,
            idx=0,
            # dtype=dtype,
            doc_string="y = x + epsilon * RandomNormal(x, seed)",
            domain="com.microsoft"
        )

        # Graph
        graph = helper.make_graph(
            [perturb_node],
            "perturb_normal_graph",
            [x_tensor],
            [perturbed_x_tensor],
        )

        return graph

    def create_model(self, graph, opset_version: int = 13):
        """Create ONNX model for PerturbNormal with custom domain."""
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
        PerturbNormal functionality using standard ONNX ops (RandomNormal + Add)
        and run inference on that to get the output.
        """
        # --- Create the "Execution" Graph ---
        # This graph implements the behavior of PerturbNormal for testing.

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

        # 1. RandomNormal node to generate noise
        # The shape of the noise must match the input shape.
        random_node = helper.make_node(
            "RandomNormal",
            inputs=[],  # RandomNormal has no inputs
            outputs=[noise_tensor_name],
            name="random_normal_for_perturb",
            shape=self.input_shape,
            dtype=TensorProto.FLOAT,
            mean=0.0,
            scale=1.0,  # Standard normal distribution
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
            "perturb_normal_execution_graph",
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
