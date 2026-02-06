"""Mamba Model Exporter - ONNX with Custom Operators."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..core.base_exporter import BaseONNXExporter
from .pytorch_models.mamba import Mamba


class MambaExporter(BaseONNXExporter):
    """
    ONNX exporter for Mamba model (Selective State Space Model).

    Exports clean ONNX graphs using custom SelectiveSSM operator:
    - High-level operators only (LayerNorm, Linear, Conv1d, Silu, SelectiveSSM)
    - ~10-15 nodes per layer (vs 50-80 with standard export)
    - No fragmented graphs (no excessive Const/Shape/Gather nodes)
    """

    def __init__(self, save_path: str = None, config_file: str = "config.yaml"):
        """
        Initialize Mamba exporter.

        Args:
            save_path: Optional custom path to save ONNX files
            config_file: Path to configuration YAML file
        """
        super().__init__(save_path, config_file)

    def load_config(self) -> Dict[str, Any]:
        """
        Load Mamba configuration.

        Returns:
            Dictionary containing Mamba configuration parameters
        """
        config = {
            "batch_size": 1,
            "d_model": 256,  # Model dimension
            "n_layers": 4,  # Number of Mamba layers
            "d_state": 16,  # SSM state dimension
            "d_conv": 4,  # Convolution kernel size
            "expand_factor": 2,  # SSM expansion factor
            "max_seq_len": 512,  # Maximum sequence length
            "vocab_size": None,  # Vocabulary size (None for continuous input)
            "num_classes": 10,  # Number of output classes (for classification)
            "dropout": 0.0,  # No dropout for inference
            "use_embedding": False,  # Use embedding layer for token inputs
            "opset_version": 17,  # ONNX opset version
            # Training configuration
            "training_strategy": "full",  # Options: "full", "last_layer", "custom"
            "custom_trainable_params": [],
        }
        return config

    def create_model(self) -> torch.nn.Module:
        """
        Create Mamba PyTorch model with ONNX export.

        Returns:
            Mamba model ready for export with custom operators
        """
        model = Mamba(
            d_model=self.config["d_model"],
            n_layers=self.config["n_layers"],
            d_state=self.config["d_state"],
            d_conv=self.config["d_conv"],
            expand_factor=self.config["expand_factor"],
            vocab_size=self.config["vocab_size"],
            num_classes=self.config["num_classes"],
            max_seq_len=self.config["max_seq_len"],
            dropout=self.config["dropout"],
            use_embedding=self.config["use_embedding"],
        )

        # Print model info
        num_params = model.get_num_params()
        print("\n📊 Mamba Model Configuration:")
        print("   Model: Mamba (Custom operators)")
        print(f"   Model dimension: {self.config['d_model']}")
        print(f"   Number of layers: {self.config['n_layers']}")
        print(f"   SSM state dimension: {self.config['d_state']}")
        print(f"   Total parameters: {num_params:,}")

        print("\n✨ ONNX Export Strategy:")
        print("   Using custom SelectiveSSM operator")
        print("   Generating ONNX graph with high-level operators")

        print("\n🎯 Expected ONNX operators (per layer):")
        print("   • LayerNorm - Normalization")
        print("   • Linear (projections) - Input/output projections")
        print("   • Conv1d (temporal) - Temporal convolution")
        print("   • Silu - Activation function")
        print("   • ai.mamba::SelectiveSSM - Custom SSM operator")
        print("   • Add (residual) - Residual connection")
        print("   Total: ~10-15 nodes/layer (vs 50-80 with fragmented export)")

        return model

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the input tensor shape for Mamba.

        Returns:
            Tuple representing input shape:
            - If use_embedding: (batch_size, seq_len) - token indices
            - Otherwise: (batch_size, seq_len, d_model) - continuous input
        """
        batch_size = self.config["batch_size"]
        seq_len = self.config["max_seq_len"]

        if self.config["use_embedding"]:
            return (batch_size, seq_len)
        else:
            d_model = self.config["d_model"]
            return (batch_size, seq_len, d_model)

    def get_trainable_params(self, all_param_names: List[str]) -> List[str]:
        """
        Get list of trainable parameter names for Mamba.

        Supports multiple training strategies:
        - "full": Train all parameters (default)
        - "last_layer": Only train the final classification/output layer
        - "custom": Use custom_trainable_params from config

        Args:
            all_param_names: List of all parameter names in the model

        Returns:
            List of parameter names that should be trainable
        """
        strategy = self.config.get("training_strategy", "full")

        if strategy == "full":
            trainable_params = all_param_names
        elif strategy == "last_layer":
            trainable_params = [
                name
                for name in all_param_names
                if "classifier" in name or "lm_head" in name or "norm_f" in name
            ]
        elif strategy == "custom":
            trainable_params = self.config.get("custom_trainable_params", [])
        else:
            print(f"⚠️  Unknown training strategy '{strategy}', using 'full'")
            trainable_params = all_param_names

        requires_grad = [name for name in all_param_names if name in trainable_params]

        print(f"\n🎯 Training Strategy: '{strategy}'")
        print(f"   Total params: {len(all_param_names)}")
        print(f"   Trainable: {len(requires_grad)}")
        print(f"   Frozen: {len(all_param_names) - len(requires_grad)}")

        return requires_grad

    def _get_config_string(self) -> str:
        """Get configuration string for folder naming."""
        d_model = self.config["d_model"]
        n_layers = self.config["n_layers"]
        seq_len = self.config["max_seq_len"]
        num_classes = self.config["num_classes"]
        return f"_mamba_{d_model}_{n_layers}_{seq_len}_{num_classes}"

    def _export_to_onnx(
        self, model: torch.nn.Module, input_tensor: torch.Tensor, opset_version: int = 17
    ):
        """
        Export Mamba model to ONNX with custom SelectiveSSM operator.

        Args:
            model: Mamba PyTorch model
            input_tensor: Sample input tensor
            opset_version: ONNX opset version (default 17)

        Returns:
            ONNX model with custom operators
        """
        import io

        import onnx

        print("\n🔧 Preparing custom ONNX export...")
        print("   Custom operator: ai.mamba::SelectiveSSM")
        print("   Opset version: 17")
        print("   Domain: ai.mamba:1")

        # Export to ONNX with custom operator domain
        f = io.BytesIO()
        torch.onnx.export(
            model,
            input_tensor,
            f,
            input_names=["input"],
            output_names=["output"],
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            keep_initializers_as_inputs=False,
            custom_opsets={"ai.mamba": 1},  # Custom operator domain
        )

        onnx_model = onnx.load_model_from_string(f.getvalue())

        # Print export summary
        print("\n✨ ONNX Export Complete:")
        print("   Expected operators per layer:")
        print("   • LayerNorm")
        print("   • Linear (MatMul+Add)")
        print("   • Conv1d")
        print("   • Silu")
        print("   • ai.mamba::SelectiveSSM (custom operator)")
        print("   • Add (residual)")
        print("   Nodes per layer: ~10-15 (vs 50-80 with fragmented export)")

        return onnx_model

    def run_inference_optimization(self, onnx_file: str, output_file: str):
        """
        Skip aggressive optimizations to preserve custom operators.

        For Mamba with custom operators, we skip graph optimizations
        that might unfold or remove the SelectiveSSM custom operator.
        """
        print("   ⏭️  Skipping graph optimizations (preserving custom operators)")

        if onnx_file != output_file:
            import shutil

            shutil.copy(onnx_file, output_file)

    def save_test_data(self, model: torch.nn.Module, save_dir: str):
        """
        Save test input/output data for validation.

        Uses PyTorch model to generate reference output.

        Args:
            model: PyTorch model to run inference with
            save_dir: Directory to save test data
        """
        print("💾 Saving test input/output data...")

        input_shape = self.get_input_shape()

        if self.config["use_embedding"]:
            vocab_size = self.config["vocab_size"]
            test_input = np.random.randint(0, vocab_size, size=input_shape, dtype=np.int64)
        else:
            test_input = np.random.randn(*input_shape).astype(np.float32)

        # Get PyTorch output
        was_training = model.training
        model.eval()

        with torch.no_grad():
            if self.config["use_embedding"]:
                input_tensor = torch.from_numpy(test_input).long()
            else:
                input_tensor = torch.from_numpy(test_input).float()

            output_tensor = model(input_tensor)
            test_output = output_tensor.numpy()

        if was_training:
            model.train()

        # Save as .npz files
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        np.savez(save_path / "inputs.npz", input=test_input)
        np.savez(save_path / "outputs.npz", output=test_output)

        print("  ✅ Saved test data:")
        print(f"     Input:  {save_path / 'inputs.npz'} shape={test_input.shape}")
        print(f"     Output: {save_path / 'outputs.npz'} shape={test_output.shape}")
