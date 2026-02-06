# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""MI-BMInet (Motor Imagery BMI Network) Model Exporter."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..core.base_exporter import BaseONNXExporter

# Import MI-BMInet PyTorch model from new location
from .pytorch_models.mibminet import MIBMINetDeploy


class MIBMInetExporter(BaseONNXExporter):
    """ONNX exporter for MI-BMInet (Motor Imagery BMI Network) model."""

    def __init__(self, save_path: str = None, config_file: str = "config.yaml"):
        """
        Initialize MI-BMInet exporter.

        Args:
            save_path: Optional custom path to save ONNX files
            config_file: Path to configuration YAML file
        """
        super().__init__(save_path, config_file)
        self.model_config = {}

    def load_config(self) -> Dict[str, Any]:
        """
        Load MI-BMInet configuration.

        Returns:
            Dictionary containing MI-BMInet configuration parameters
        """
        # Default MI-BMInet configuration
        config = {
            "batch_size": 1,
            "channels": 8,
            "time_steps": 2000,
            "num_classes": 2,
            "F1": 8,
            "D": 2,
            "Nf": 64,
            "Nf2": 16,
            "opset_version": 12,
            "activation": "relu",
        }

        self.model_config = config
        return config

    def create_model(self) -> torch.nn.Module:
        """
        Create MI-BMInet PyTorch model.

        Returns:
            MI-BMInet model ready for export
        """
        model = MIBMINetDeploy(
            F1=self.model_config["F1"],
            D=self.model_config["D"],
            C=self.model_config["channels"],
            T=self.model_config["time_steps"],
            N=self.model_config["num_classes"],
            Nf=self.model_config["Nf"],
            Nf2=self.model_config["Nf2"],
            activation=self.model_config.get("activation", "relu"),
        )

        return model

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the input tensor shape for MI-BMInet.

        Returns:
            Tuple representing input shape (batch_size, 1, channels, time_steps)
        """
        batch_size = self.config["batch_size"]
        channels = self.config["channels"]
        time_steps = self.config["time_steps"]
        return (batch_size, 1, channels, time_steps)

    def get_trainable_params(self, all_param_names: List[str]) -> List[str]:
        """
        Get list of trainable parameter names for MI-BMInet.

        Args:
            all_param_names: List of all parameter names in the model

        Returns:
            List of parameter names that should be trainable
        """
        # All parameters trainable by default
        return all_param_names

    def _get_config_string(self) -> str:
        """
        Get configuration string for folder naming.

        Returns:
            Configuration string like "_8_2000_2"
        """
        return (
            f"_{self.config['channels']}_{self.config['time_steps']}_{self.config['num_classes']}"
        )

    def save_test_data(self, model: torch.nn.Module, save_dir: str):
        """
        Save test input/output data for validation.

        Args:
            model: PyTorch model to run inference with
            save_dir: Directory to save test data
        """
        print("💾 Saving test input/output data...")

        # Create test input
        input_shape = self.get_input_shape()
        test_input = np.random.randn(*input_shape).astype(np.float32)

        # Get PyTorch output
        was_training = model.training
        model.eval()

        with torch.no_grad():
            input_tensor = torch.from_numpy(test_input)
            output_tensor = model(input_tensor)
            test_output = output_tensor.numpy()

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
