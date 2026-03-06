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
   
        normal_epsilon = 0.01
        uniform_epsilon = 0.01 * np.sqrt(3)
        rademacher_epsilon = 0.01 
        

        # Shape annotation for intermediate outputs
        a_shape = [self.input_shape[0], 1]
        b_shape = [int(np.prod(self.input_shape[1:])), 1]

        a_tensor = helper.make_tensor_value_info(
            "a", TensorProto.FLOAT, a_shape
        )
        b_tensor = helper.make_tensor_value_info(
            "b", TensorProto.FLOAT, b_shape
        )
        
        shape_a_tensor = helper.make_tensor(name=f"shape_a", data_type=TensorProto.INT64, dims=[len(a_shape)],
                                                vals=np.array(a_shape, dtype=np.int64))
        shape_b_tensor = helper.make_tensor(name=f"shape_b", data_type=TensorProto.INT64, dims=[len(b_shape)],
                                                vals=np.array(b_shape, dtype=np.int64))

        shape_input_name = helper.make_tensor(name=f"shape_x", data_type=TensorProto.INT64, dims=[len(self.input_shape)],
                                                vals=np.array(self.input_shape, dtype=np.int64))
        
        if len(self.input_shape) > 2:

            shape_flat_name = helper.make_tensor(name=f"shape_x_flat", data_type=TensorProto.INT64, dims=[2],
                                                    vals=np.array([a_shape[0], b_shape[0]], dtype=np.int64))
            # insert flattening nodes
            flatten_node = helper.make_node(
                "Reshape",
                inputs=["x", f"shape_x_flat"],
                outputs=[f"flattened_x"],
                name=f"flatten_x"
            )
            
            flattened_tensor_name = helper.make_tensor_value_info(
                f"flattened_x", TensorProto.FLOAT,[a_shape[0], b_shape[0]]
            )

            unflatten_node = helper.make_node(
                "Reshape",
                inputs=["flattened_perturbed_x", "shape_x"],
                outputs=["perturbed_x"],
                name="unflatten_perturbed_x"
            )
            flattened_perturbed_tensor_name = helper.make_tensor_value_info(
                "flattened_perturbed_x", TensorProto.FLOAT, [a_shape[0], b_shape[0]]
            )
            eggroll_input = "flattened_x"
            eggroll_output = "flattened_perturbed_x"
        else:
            eggroll_input = "x"
            eggroll_output = "perturbed_x"

        # Eggroll noise node (without loss_grad input)
        noise_node_a = helper.make_node(
            "PerturbEggroll",
            inputs=["shape_a"],
            outputs=["a"],
            name=f"gen_eggroll_noise_a",
            seed=13,
            idx=0,
            domain="com.microsoft",
            doc_string="a = RandomRademacher(x[0], seed)"
        )
        
        noise_node_b = helper.make_node(
            "PerturbEggroll",
            inputs=["shape_b"],
            outputs=["b"],
            name=f"gen_eggroll_noise_b",
            seed=14,
            idx=1,
            domain="com.microsoft",
            doc_string="b = RandomRademacher(x[1:], seed)"
        )

        gemm_node = helper.make_node(
            "Gemm",
            inputs=["a", "b", eggroll_input],
            outputs=[eggroll_output],
            name=f"eggroll_gemm_perturb_x",
            transA=0,
            transB=1,
            alpha=uniform_epsilon,
            beta=0
        )
        # Graph
        if len(self.input_shape) > 2:
            graph = helper.make_graph(
                [flatten_node, noise_node_a, noise_node_b, gemm_node, unflatten_node],
                "perturb_eggroll_graph",
                [x_tensor],
                [perturbed_x_tensor],
                [shape_input_name, 
                 shape_flat_name,
                 shape_a_tensor,
                 shape_b_tensor],  # <-- shape annotations here
                value_info=[a_tensor, b_tensor, 
                            flattened_tensor_name,
                            flattened_perturbed_tensor_name]  # <-- shape annotation here
            )
        else:
            graph = helper.make_graph(
                [noise_node_a, noise_node_b, gemm_node],
                "perturb_eggroll_graph",
                [x_tensor],
                [perturbed_x_tensor],
                [shape_a_tensor, shape_b_tensor],
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
        if len(self.input_shape) > 2:
            perturbation = perturbation.reshape(self.input_shape)
        perturbed_x = inputs["x"] + perturbation

        return {"perturbed_x": perturbed_x}

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Return None to skip validation - this is a custom operator.
        """
        return None
