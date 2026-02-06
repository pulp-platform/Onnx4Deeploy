"""TinyViT Model Exporter."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..core.base_exporter import BaseONNXExporter

# Import TinyViT PyTorch models
from .pytorch_models.tinyvit import tiny_vit_5m, tiny_vit_11m, tiny_vit_21m


class TinyViTExporter(BaseONNXExporter):
    """ONNX exporter for TinyViT models (5M, 11M, 21M variants)."""

    def __init__(self, save_path: str = None, config_file: str = "config.yaml"):
        """
        Initialize TinyViT exporter.

        Args:
            save_path: Optional custom path to save ONNX files
            config_file: Path to configuration YAML file
        """
        super().__init__(save_path, config_file)
        self.model_config = {}

    def load_config(self) -> Dict[str, Any]:
        """
        Load TinyViT configuration.

        Returns:
            Dictionary containing TinyViT configuration parameters
        """
        # Default TinyViT configuration
        config = {
            "batch_size": 1,
            "img_size": 224,  # Standard ImageNet size
            "input_channels": 3,  # RGB
            "num_classes": 1000,  # ImageNet classes
            "opset_version": 17,  # LayerNorm requires opset 17+
            "variant": "tiny_vit_5m",  # Options: "tiny_vit_5m", "tiny_vit_11m", "tiny_vit_21m"
            # Training configuration
            "training_strategy": "full",  # Options: "full", "head_only", "custom"
            "custom_trainable_params": [],
        }

        self.model_config = config
        return config

    def create_model(self) -> torch.nn.Module:
        """
        Create TinyViT PyTorch model.

        Returns:
            TinyViT model ready for export
        """
        variant = self.model_config.get("variant", "tiny_vit_5m")
        num_classes = self.model_config["num_classes"]
        img_size = self.model_config["img_size"]

        # Select model variant
        if variant == "tiny_vit_5m":
            model = tiny_vit_5m(num_classes=num_classes, img_size=img_size)
        elif variant == "tiny_vit_11m":
            model = tiny_vit_11m(num_classes=num_classes, img_size=img_size)
        elif variant == "tiny_vit_21m":
            model = tiny_vit_21m(num_classes=num_classes, img_size=img_size)
        else:
            raise ValueError(
                f"Unknown TinyViT variant: {variant}. "
                f"Choose from: tiny_vit_5m, tiny_vit_11m, tiny_vit_21m"
            )

        return model

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the input tensor shape for TinyViT.

        Returns:
            Tuple representing input shape (batch_size, channels, height, width)
        """
        batch_size = self.config["batch_size"]
        img_size = self.config["img_size"]
        input_channels = self.config["input_channels"]
        return (batch_size, input_channels, img_size, img_size)

    def get_inference_pipeline(self):
        """
        Get TinyViT-specific inference optimization pipeline.

        TinyViT uses transformer-specific optimizations like CCT.

        Returns:
            OptimizationPipeline configured for TinyViT inference
        """
        from ..core.optimization_passes import create_transformer_inference_pipeline

        # Get model config
        variant = self.config.get("variant", "tiny_vit_5m")

        # Map variant to embedding_dim
        variant_config = {
            "tiny_vit_5m": {"embed_dim": 192, "num_heads": 3},
            "tiny_vit_11m": {"embed_dim": 256, "num_heads": 4},
            "tiny_vit_21m": {"embed_dim": 384, "num_heads": 6},
        }

        config = variant_config.get(variant, variant_config["tiny_vit_5m"])

        # Create transformer-specific pipeline with model parameters
        pipeline = create_transformer_inference_pipeline(
            embedding_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            input_shape=self.get_input_shape(),
        )

        return pipeline

    def get_trainable_params(self, all_param_names: List[str]) -> List[str]:
        """
        Get list of trainable parameter names for TinyViT.

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
            "head_only": [name for name in all_param_names if "head" in name],
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
            Configuration string like "_tiny_vit_5m_224_1000"
        """
        variant = self.config.get("variant", "tiny_vit_5m")
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
