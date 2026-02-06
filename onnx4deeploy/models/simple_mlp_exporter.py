# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Simple MLP Model Exporter."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..core.base_exporter import BaseONNXExporter

# Import SimpleMLP PyTorch model from new location
from .pytorch_models.simple_mlp import SimpleMLP


class SimpleMlpExporter(BaseONNXExporter):
    """ONNX exporter for Simple MLP model."""

    def __init__(self, save_path: str = None, config_file: str = "config.yaml"):
        """
        Initialize Simple MLP exporter.

        Args:
            save_path: Optional custom path to save ONNX files
            config_file: Path to configuration YAML file
        """
        super().__init__(save_path, config_file)
        self.model_config = {}

    def load_config(self) -> Dict[str, Any]:
        """
        Load Simple MLP configuration.

        Returns:
            Dictionary containing Simple MLP configuration parameters
        """
        # Default Simple MLP configuration
        config = {
            "batch_size": 1,
            "input_height": 28,
            "input_width": 28,
            "hidden_size": 128,
            "num_classes": 10,
            "opset_version": 17,
            "dropout": 0.0,  # No dropout for inference
            # Training configuration
            "training_strategy": "full",  # Options: "full", "last_layer", "custom"
            "custom_trainable_params": [],
        }

        self.model_config = config
        return config

    def create_model(self) -> torch.nn.Module:
        """
        Create Simple MLP PyTorch model.

        Returns:
            Simple MLP model ready for export
        """
        input_size = self.model_config["input_height"] * self.model_config["input_width"]

        model = SimpleMLP(
            input_size=input_size,
            hidden_size=self.model_config["hidden_size"],
            num_classes=self.model_config["num_classes"],
            dropout=self.model_config["dropout"],
        )

        return model

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the input tensor shape for Simple MLP.

        Returns:
            Tuple representing input shape (batch_size, 1, height, width)
        """
        batch_size = self.config["batch_size"]
        height = self.config["input_height"]
        width = self.config["input_width"]
        return (batch_size, 1, height, width)

    def get_trainable_params(self, all_param_names: List[str]) -> List[str]:
        """
        Get list of trainable parameter names for Simple MLP.

        Supports multiple training strategies:
        - "full": Train all parameters (default)
        - "last_layer": Only train the final classification layer
        - "custom": Use custom_trainable_params from config

        Args:
            all_param_names: List of all parameter names in the model

        Returns:
            List of parameter names that should be trainable
        """
        strategy = self.config.get("training_strategy", "full")

        # Define training strategies
        strategy_params = {
            "full": all_param_names,  # Train everything
            "last_layer": [
                "fc3.weight",
                "fc3.bias",
            ],
            "custom": self.config.get("custom_trainable_params", []),
        }

        # Get trainable params based on strategy
        if strategy not in strategy_params:
            print(f"⚠️  Unknown training strategy '{strategy}', using 'full' as fallback")
            strategy = "full"

        trainable_params = strategy_params[strategy]

        # Filter to only include params that exist in the model
        requires_grad = [name for name in all_param_names if name in trainable_params]

        # Print strategy info
        print(f"\n🎯 Training Strategy: '{strategy}'")
        print(f"   Total params in model: {len(all_param_names)}")
        print(f"   Params to train: {len(requires_grad)}")
        print(f"   Frozen params: {len(all_param_names) - len(requires_grad)}")

        return requires_grad

    def _get_config_string(self) -> str:
        """
        Get configuration string for folder naming.

        Returns:
            Configuration string like "_28x28_128_10"
        """
        return f"_{self.config['input_height']}x{self.config['input_width']}_{self.config['hidden_size']}_{self.config['num_classes']}"

    def save_test_data(self, model: torch.nn.Module, save_dir: str):
        """
        Save test input/output data for validation.

        Uses PyTorch model to generate reference output for validating ONNX correctness.

        Args:
            model: PyTorch model to run inference with
            save_dir: Directory to save test data
        """
        print("💾 Saving test input/output data...")

        # Create test input
        input_shape = self.get_input_shape()
        test_input = np.random.randn(*input_shape).astype(np.float32)

        # Get PyTorch output (reference for validating ONNX)
        was_training = model.training
        model.eval()

        with torch.no_grad():
            input_tensor = torch.from_numpy(test_input)
            output_tensor = model(input_tensor)
            test_output = output_tensor.numpy()

        # Restore training mode if needed
        if was_training:
            model.train()

        # Save as .npz files
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        np.savez(save_path / "inputs.npz", input=test_input)
        np.savez(save_path / "outputs.npz", output=test_output)

        print("  ✅ Saved test data (PyTorch reference):")
        print(f"     Input:  {save_path / 'inputs.npz'} shape={test_input.shape}")
        print(f"     Output: {save_path / 'outputs.npz'} shape={test_output.shape}")
