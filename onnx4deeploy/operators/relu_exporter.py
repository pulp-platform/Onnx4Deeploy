"""
ReLU Operator Exporter using BaseONNXExporter.

This is a refactored version of Tests/Operators/testFloatRelu/testgenerate.py
demonstrating how to use the BaseONNXExporter abstraction for simple operators.
"""

import os
from typing import Any, Dict, Tuple

import numpy as np
import onnxruntime as ort
import yaml
from onnx import TensorProto, helper

# Note: BaseONNXExporter is designed for PyTorch models
# For simple operators, we override the necessary methods
from onnx4deeploy.core.base_exporter import BaseONNXExporter


class ReluExporter(BaseONNXExporter):
    """
    ONNX exporter for ReLU operator.

    This is a simple operator that doesn't require PyTorch.
    It demonstrates using BaseONNXExporter for operator testing.

    Usage:
        exporter = ReluExporter(config_file="path/to/config.yaml")
        exporter.export_operator(save_path="./onnx")
    """

    def load_config(self) -> Dict[str, Any]:
        """
        Load ReLU operator configuration.

        Returns:
            Dictionary with input_shape and other settings
        """
        # Resolve config.yaml relative to current directory or provided path
        if os.path.isabs(self.config_file):
            config_path = self.config_file
        else:
            config_path = os.path.join(os.getcwd(), self.config_file)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        model_config = config.get("relu", {})

        return {
            "input_shape": tuple(model_config.get("input_shape", [1, 64, 32, 32])),
            "opset_version": model_config.get("opset_version", 13),
            "batch_size": model_config.get("input_shape", [1, 64, 32, 32])[0],
        }

    def create_model(self):
        """
        ReLU doesn't need a PyTorch model - we create ONNX directly.

        Returns:
            None (we override _export_to_onnx instead)
        """
        return None

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get input shape for ReLU operator.

        Returns:
            Input tensor shape
        """
        return self.config["input_shape"]

    def create_onnx_graph(self):
        """
        Create ONNX graph for ReLU operator.

        Returns:
            ONNX ModelProto
        """
        input_shape = self.get_input_shape()
        opset_version = self.config.get("opset_version", 13)

        # Define ONNX tensors
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shape)

        # Create ONNX computation graph
        relu_node = helper.make_node("Relu", inputs=["input"], outputs=["output"], name="relu_node")

        graph_def = helper.make_graph([relu_node], "relu_graph", [input_tensor], [output_tensor])

        model_def = helper.make_model(
            graph_def,
            producer_name="relu_model",
            opset_imports=[helper.make_opsetid("", opset_version)],
        )

        return model_def

    def generate_test_data(self, base_path: str):
        """
        Generate test input and output data for ReLU.

        Args:
            base_path: Directory to save test data
        """
        input_shape = self.get_input_shape()

        # Generate random input data (with some negative values)
        input_data = np.random.uniform(-1, 1, input_shape).astype(np.float32)

        # Save input
        input_file = os.path.join(base_path, "inputs.npz")
        np.savez(input_file, input=input_data)
        print(f"✅ Input data saved to {input_file}")

        # Run inference using ONNX Runtime
        onnx_file = os.path.join(base_path, "network.onnx")
        if os.path.exists(onnx_file):
            ort_session = ort.InferenceSession(onnx_file)
            output_data = ort_session.run(None, {"input": input_data})[0]

            # Save output
            output_file = os.path.join(base_path, "outputs.npz")
            np.savez(output_file, output=output_data)
            print(f"✅ Output data saved to {output_file}")

    def export_operator(self, save_path: str = None) -> str:
        """
        Export ReLU operator to ONNX with test data.

        This is a simplified export for operators (no training mode).

        Args:
            save_path: Optional custom save path

        Returns:
            Path to the exported ONNX file
        """
        if save_path:
            self.save_path = save_path

        # Load configuration
        self.config = self.load_config()

        # Setup paths
        if self.save_path:
            output_dir = self.save_path
        else:
            output_dir = os.path.join(os.getcwd(), "onnx")

        os.makedirs(output_dir, exist_ok=True)
        self.paths = {"output_dir": output_dir, "network": os.path.join(output_dir, "network.onnx")}

        print(f"\n{'='*60}")
        print("🚀 Exporting ReLU Operator to ONNX")
        print(f"{'='*60}\n")
        print(f"   Input shape: {self.config['input_shape']}")
        print(f"   Opset version: {self.config['opset_version']}")

        # Create ONNX model directly
        print("\n📤 Creating ONNX graph...")
        import onnx

        onnx_model = self.create_onnx_graph()

        # Save ONNX model
        onnx.save(onnx_model, self.paths["network"])
        print(f"✅ ONNX model saved to {self.paths['network']}")

        # Generate test data
        print("\n🧪 Generating test data...")
        self.generate_test_data(self.paths["output_dir"])

        print(f"\n{'='*60}")
        print("✅ Export Complete!")
        print(f"   ONNX file: {self.paths['network']}")
        print(f"{'='*60}\n")

        return self.paths["network"]


def generate_relu_onnx_and_data(save_path=None, config_file="config.yaml"):
    """
    Convenience function for backward compatibility.

    Generates ReLU ONNX model and test data using the new ReluExporter.

    Args:
        save_path: Optional path to save ONNX and data files
        config_file: Path to configuration YAML file

    Returns:
        Path to the generated ONNX file
    """
    exporter = ReluExporter(save_path=save_path, config_file=config_file)
    return exporter.export_operator()
