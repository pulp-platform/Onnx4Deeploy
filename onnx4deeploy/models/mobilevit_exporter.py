# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""MobileViT Model Exporter."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..core.base_exporter import BaseONNXExporter

# Import MobileViT PyTorch models
from .pytorch_models.mobilevit import mobile_vit_s, mobile_vit_xs, mobile_vit_xxs


class MobileViTExporter(BaseONNXExporter):
    """ONNX exporter for MobileViT models (XXS, XS, S variants)."""

    def __init__(self, save_path: str = None, config_file: str = "config.yaml"):
        """
        Initialize MobileViT exporter.

        Args:
            save_path: Optional custom path to save ONNX files
            config_file: Path to configuration YAML file
        """
        super().__init__(save_path, config_file)
        self.model_config = {}

    def load_config(self) -> Dict[str, Any]:
        """
        Load MobileViT configuration.

        Returns:
            Dictionary containing MobileViT configuration parameters
        """
        # Default MobileViT configuration
        config = {
            "batch_size": 1,
            "img_size": 256,  # MobileViT uses 256x256
            "input_channels": 3,  # RGB
            "num_classes": 1000,  # ImageNet classes
            "opset_version": 17,  # LayerNorm requires opset 17+
            "variant": "mobile_vit_xs",  # Options: "mobile_vit_xxs", "mobile_vit_xs", "mobile_vit_s"
            # Training configuration
            "training_strategy": "full",  # Options: "full", "head_only", "custom"
            "custom_trainable_params": [],
        }

        self.model_config = config
        return config

    def create_model(self) -> torch.nn.Module:
        """
        Create MobileViT PyTorch model.

        Returns:
            MobileViT model ready for export
        """
        variant = self.model_config.get("variant", "mobile_vit_xs")
        num_classes = self.model_config["num_classes"]
        batch_size = self.model_config["batch_size"]
        img_size = self.model_config["img_size"]

        # Select model variant with fixed dimensions
        if variant == "mobile_vit_xxs":
            model = mobile_vit_xxs(
                batch_size=batch_size,
                image_size=(img_size, img_size),
                num_classes=num_classes,
            )
        elif variant == "mobile_vit_xs":
            model = mobile_vit_xs(
                batch_size=batch_size,
                image_size=(img_size, img_size),
                num_classes=num_classes,
            )
        elif variant == "mobile_vit_s":
            model = mobile_vit_s(
                batch_size=batch_size,
                image_size=(img_size, img_size),
                num_classes=num_classes,
            )
        else:
            raise ValueError(
                f"Unknown MobileViT variant: {variant}. "
                f"Choose from: mobile_vit_xxs, mobile_vit_xs, mobile_vit_s"
            )

        return model

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the input tensor shape for MobileViT.

        Returns:
            Tuple representing input shape (batch_size, channels, height, width)
        """
        batch_size = self.config["batch_size"]
        img_size = self.config["img_size"]
        input_channels = self.config["input_channels"]
        return (batch_size, input_channels, img_size, img_size)

    def get_trainable_params(self, all_param_names: List[str]) -> List[str]:
        """
        Get list of trainable parameter names for MobileViT.

        Supports multiple training strategies:
        - "full": Train all parameters (default)
        - "head_only": Only train classification head
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
            "head_only": [name for name in all_param_names if "fc" in name],
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
            Configuration string like "_mobile_vit_xs_256_1000"
        """
        variant = self.config.get("variant", "mobile_vit_xs")
        return f"_{variant}_{self.config['img_size']}_{self.config['num_classes']}"

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
