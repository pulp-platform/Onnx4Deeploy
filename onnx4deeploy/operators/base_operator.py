"""
Base class for ONNX operator test generation and validation.

This module provides a unified framework for creating operator-level tests
that generate ONNX models, test data, and validate correctness.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import yaml
from onnx import TensorProto, helper


class BaseOperatorTest(ABC):
    """
    Abstract base class for ONNX operator tests.

    Provides common functionality for:
    - Loading configuration from YAML
    - Generating test input data
    - Creating ONNX graphs
    - Running ONNX Runtime inference
    - Validating outputs
    - Saving models and data
    """

    def __init__(self, config_path: Optional[str] = None, save_path: Optional[str] = None):
        """
        Initialize the operator test.

        Args:
            config_path: Path to config.yaml file. If None, uses config.yaml in same directory as the test.
            save_path: Directory to save ONNX model and test data. If None, uses 'onnx' subdirectory.
        """
        self.config_path = config_path
        self.save_path = save_path
        self.config = None

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self.config_path is None:
            # Try to find config.yaml relative to the calling script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.config_path = os.path.join(script_dir, "config.yaml")

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        return self.config

    def get_save_paths(self) -> Tuple[str, str, str]:
        """
        Get standardized file paths for ONNX model and data.

        Returns:
            Tuple of (onnx_file, input_file, output_file)
        """
        if self.save_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.save_path = os.path.join(script_dir, "onnx")

        os.makedirs(self.save_path, exist_ok=True)

        onnx_file = os.path.join(self.save_path, "network.onnx")
        input_file = os.path.join(self.save_path, "inputs.npz")
        output_file = os.path.join(self.save_path, "outputs.npz")

        return onnx_file, input_file, output_file

    @abstractmethod
    def get_operator_name(self) -> str:
        """Return the ONNX operator name (e.g., 'Add', 'Relu', 'Gemm')."""

    @abstractmethod
    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """
        Generate test input data based on configuration.

        Returns:
            Dictionary mapping input names to numpy arrays
        """

    @abstractmethod
    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]) -> onnx.GraphProto:
        """
        Create ONNX computation graph.

        Args:
            inputs: Dictionary of input tensors

        Returns:
            ONNX GraphProto
        """

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute expected output using NumPy (for validation).

        Args:
            inputs: Dictionary of input tensors

        Returns:
            Dictionary mapping output names to numpy arrays

        Note:
            Override this method to provide custom validation logic.
            Default implementation uses ONNX Runtime as ground truth.
        """
        return None

    def create_model(self, graph: onnx.GraphProto, opset_version: int = 13) -> onnx.ModelProto:
        """
        Create ONNX model from graph.

        Args:
            graph: ONNX GraphProto
            opset_version: ONNX opset version

        Returns:
            ONNX ModelProto
        """
        model = helper.make_model(
            graph,
            producer_name=f"{self.get_operator_name().lower()}_test",
            opset_imports=[helper.make_opsetid("", opset_version)],
        )
        return model

    def run_inference(self, onnx_file: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference using ONNX Runtime.

        Args:
            onnx_file: Path to ONNX model file
            inputs: Dictionary of input tensors

        Returns:
            Dictionary mapping output names to numpy arrays
        """
        session = ort.InferenceSession(onnx_file)
        output_names = [output.name for output in session.get_outputs()]

        # Run inference
        outputs = session.run(output_names, inputs)

        # Create output dictionary
        output_dict = {name: data for name, data in zip(output_names, outputs)}

        return output_dict

    def validate_outputs(
        self,
        onnx_outputs: Dict[str, np.ndarray],
        expected_outputs: Optional[Dict[str, np.ndarray]] = None,
        tolerance: float = 1e-5,
    ) -> bool:
        """
        Validate ONNX Runtime outputs against expected outputs.

        Args:
            onnx_outputs: Outputs from ONNX Runtime
            expected_outputs: Expected outputs (if None, validation is skipped)
            tolerance: Maximum allowed difference

        Returns:
            True if validation passes
        """
        if expected_outputs is None:
            print("⚠️ No expected outputs provided, skipping validation")
            return True

        for name, expected in expected_outputs.items():
            if name not in onnx_outputs:
                print(f"❌ Output '{name}' not found in ONNX outputs")
                return False

            actual = onnx_outputs[name]
            max_diff = np.max(np.abs(actual - expected))

            print(f"Output '{name}': max_diff = {max_diff:.2e}")

            if max_diff > tolerance:
                print(
                    f"❌ Validation failed for '{name}': max_diff = {max_diff:.2e} > {tolerance:.2e}"
                )
                return False

        print("✅ Validation passed")
        return True

    def save_data(
        self,
        inputs: Dict[str, np.ndarray],
        outputs: Dict[str, np.ndarray],
        input_file: str,
        output_file: str,
    ):
        """
        Save input and output data to NPZ files.

        Args:
            inputs: Dictionary of input tensors
            outputs: Dictionary of output tensors
            input_file: Path to save inputs
            output_file: Path to save outputs
        """
        np.savez(input_file, **inputs)
        print(f"✅ Input data saved to {input_file}")

        np.savez(output_file, **outputs)
        print(f"✅ Output data saved to {output_file}")

    def generate(self) -> Tuple[str, str, str]:
        """
        Complete pipeline: generate model, run inference, and save data.

        Returns:
            Tuple of (onnx_file, input_file, output_file)
        """
        # Load configuration
        self.load_config()

        # Generate inputs
        inputs = self.generate_inputs()

        # Create ONNX graph
        graph = self.create_onnx_graph(inputs)

        # Create model
        model = self.create_model(graph)

        # Get save paths
        onnx_file, input_file, output_file = self.get_save_paths()

        # Save ONNX model
        onnx.save(model, onnx_file)
        print(f"✅ ONNX model saved to {onnx_file}")

        # Run inference
        outputs = self.run_inference(onnx_file, inputs)

        # Validate outputs (if expected outputs are provided)
        expected_outputs = self.compute_expected_output(inputs)
        if expected_outputs is not None:
            self.validate_outputs(outputs, expected_outputs)

        # Save data
        self.save_data(inputs, outputs, input_file, output_file)

        return onnx_file, input_file, output_file


class SimpleElementwiseOperator(BaseOperatorTest):
    """
    Base class for simple elementwise operators (Add, Relu, etc.).

    Simplifies creation of operators with:
    - One or more inputs of the same shape
    - One output of the same shape
    - No additional attributes
    """

    def __init__(self, config_path: Optional[str] = None, save_path: Optional[str] = None):
        super().__init__(config_path, save_path)
        self.input_shapes = []
        self.input_names = []

    @abstractmethod
    def get_config_key(self) -> str:
        """Return the key in config.yaml for this operator."""

    def load_config(self) -> Dict[str, Any]:
        """Load configuration and extract input shapes."""
        # Try to load config, use default if not found
        try:
            config = super().load_config()
        except FileNotFoundError:
            # Use default configuration
            config = {self.get_config_key(): self.get_default_operator_config()}

        # Get operator-specific config
        op_config = config.get(self.get_config_key(), {})

        # If no config found, use default
        if not op_config:
            op_config = self.get_default_operator_config()

        # Extract input shape(s)
        if "input_shape" in op_config:
            # Single input
            self.input_shapes = [tuple(op_config["input_shape"])]
            self.input_names = ["input"]
        elif "input_a_shape" in op_config:
            # Multiple inputs (e.g., Add)
            self.input_shapes = [
                tuple(op_config.get(f"input_{name}_shape"))
                for name in ["a", "b", "c", "d", "e"]
                if f"input_{name}_shape" in op_config
            ]
            self.input_names = [
                f"input_{name}"
                for name in ["a", "b", "c", "d", "e"]
                if f"input_{name}_shape" in op_config
            ]
        else:
            # No input shape found, use default
            default_config = self.get_default_operator_config()
            if "input_shape" in default_config:
                self.input_shapes = [tuple(default_config["input_shape"])]
                self.input_names = ["input"]

        return config

    def get_default_operator_config(self) -> Dict[str, Any]:
        """
        Get default configuration for this operator.

        Subclasses can override to provide custom defaults.

        Returns:
            Default operator configuration
        """
        return {
            "input_shape": [1, 64, 32, 32],  # Default NCHW shape
        }

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random input data."""
        inputs = {}
        for name, shape in zip(self.input_names, self.input_shapes):
            inputs[name] = self.generate_single_input(name, shape)
        return inputs

    def generate_single_input(self, name: str, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Generate a single input tensor.

        Override this method to customize input generation (e.g., for Relu, use uniform(-1, 1)).
        """
        return np.random.randn(*shape).astype(np.float32) * 0.1

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]) -> onnx.GraphProto:
        """Create ONNX graph for elementwise operator."""
        # Create input tensor infos
        input_tensors = [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, data.shape)
            for name, data in inputs.items()
        ]

        # Output has same shape as first input
        output_shape = list(inputs.values())[0].shape
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        # Create operator node
        node = self.create_operator_node(list(inputs.keys()))

        # Create graph
        graph = helper.make_graph(
            [node], f"{self.get_operator_name().lower()}_graph", input_tensors, [output_tensor]
        )

        return graph

    def create_operator_node(self, input_names: List[str]) -> onnx.NodeProto:
        """
        Create the operator node.

        Override this method to add custom attributes.
        """
        return helper.make_node(
            self.get_operator_name(),
            inputs=input_names,
            outputs=["output"],
            name=f"{self.get_operator_name().lower()}_node",
        )
