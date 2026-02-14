# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""SleepConViT Model Exporter."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..core.base_exporter import BaseONNXExporter

# Import SleepConViT PyTorch model from new location
from .pytorch_models.sleep_convit import SleepConViT

# Note: We use F.gelu in the model (functional interface) which has
# built-in ONNX export support.
# CascadedConcat uses torch.autograd.Function.symbolic for ONNX export,
# so no manual registration is needed.


class SleepConViTExporter(BaseONNXExporter):
    """ONNX exporter for SleepConViT model."""

    def __init__(self, save_path: str = None, config_file: str = "config.yaml"):
        """
        Initialize SleepConViT exporter.

        Args:
            save_path: Optional custom path to save ONNX files
            config_file: Path to configuration YAML file
        """
        super().__init__(save_path, config_file)
        self.model_config = {}

    def get_inference_pipeline(self):
        """
        Get SleepConViT-specific inference optimization pipeline.

        Uses transformer-specific optimizations (same as CCT) including:
        - ONNX Runtime transformer optimizer (fuses GELU, LayerNorm, etc.)
        - Standard inference optimizations

        Returns:
            OptimizationPipeline configured for SleepConViT inference
        """
        from ..core.optimization_passes import create_transformer_inference_pipeline

        # Use transformer pipeline with SleepConViT parameters
        # Input shape is already 4D: (B, C, H, W)
        return create_transformer_inference_pipeline(
            embedding_dim=self.config["model_dim"],
            num_heads=self.config["num_heads"],
            input_shape=self.get_input_shape(),  # Already 4D: (B, 1, 1, 3000)
        )

    def load_config(self) -> Dict[str, Any]:
        """
        Load SleepConViT configuration.

        Returns:
            Dictionary containing SleepConViT configuration parameters
        """
        # Default SleepConViT configuration
        config = {
            "batch_size": 1,
            "input_channels": 1,
            "input_length": 3000,  # Time-series sequence length
            "model_dim": 48,
            "num_heads": 6,
            "num_patches": 94,  # Computed from ConvStem output
            "seq_len": 95,  # num_patches + 1 (CLS token)
            "attention_dropout": 0.0,  # No dropout for inference
            "mlp_head_hidden_dim": 48,
            "encoder_ff_dropout": 0.0,  # No dropout for inference
            "num_classes": 5,  # Sleep stages: Wake, N1, N2, N3, REM
            "opset_version": 17,  # Match CCT opset version for compatibility
            # Training configuration
            "training_strategy": "full",  # Options: "full", "last_layer", "custom"
            "custom_trainable_params": [],
            # ZO training configuration
            "zo": {
                "epsilon": 0.1,
                "seed": 42,
                "noise_type": "uniform",
            },
        }

        self.model_config = config
        return config

    def create_model(self) -> torch.nn.Module:
        """
        Create SleepConViT PyTorch model.

        Returns:
            SleepConViT model ready for export
        """
        model = SleepConViT(config=self.model_config)
        return model

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the input tensor shape for SleepConViT.

        Returns:
            Tuple representing input shape (batch_size, channels, height, width)
            Shape: (B, 1, 1, 3000) for compatibility with ViT/transformer optimizer
        """
        batch_size = self.config["batch_size"]
        channels = self.config["input_channels"]
        length = self.config["input_length"]
        return (batch_size, channels, 1, length)  # 4D: (B, C, H, W)

    def get_trainable_params(self, all_param_names: List[str]) -> List[str]:
        """
        Get list of trainable parameter names for SleepConViT.

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
                "classifier.lin1.weight",
                "classifier.lin1.bias",
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
            Configuration string like "_3000_48d_8h_5cls"
        """
        return (
            f"_{self.config['input_length']}"
            f"_{self.config['model_dim']}d"
            f"_{self.config['num_heads']}h"
            f"_{self.config['num_classes']}cls"
        )

    def save_test_data(self, model: torch.nn.Module, save_dir: str):
        """
        Save test input/output data for validation.

        Loads ONNX weights into PyTorch model to ensure outputs.npz matches ONNX exactly.
        This handles ONNX-specific optimizations like MatMul transpose and weight-sharing.

        Args:
            model: PyTorch model to load ONNX weights into
            save_dir: Directory to save test data
        """
        import onnx

        print("💾 Saving test input/output data...")

        # Create test input
        input_shape = self.get_input_shape()
        test_input = np.random.randn(*input_shape).astype(np.float32)

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Load ONNX weights
        onnx_path = save_path / "network.onnx"

        if not onnx_path.exists():
            print(f"  ⚠️  ONNX model not found at {onnx_path}")
            print("  ⚠️  Run export() first, then call save_test_data()")
            return

        # Load ONNX weights
        print(f"  📦 Loading ONNX weights from {onnx_path.name}...")
        onnx_model = onnx.load(str(onnx_path))
        onnx_weights = {}

        for init in onnx_model.graph.initializer:
            if init.data_type == 1:  # FLOAT
                data = np.frombuffer(init.raw_data, dtype=np.float32)
            elif init.data_type == 7:  # INT64
                data = np.frombuffer(init.raw_data, dtype=np.int64)
            elif init.data_type == 6:  # INT32
                data = np.frombuffer(init.raw_data, dtype=np.int32)
            else:
                continue

            try:
                onnx_weights[init.name] = data.reshape(init.dims) if init.dims else data
            except:
                onnx_weights[init.name] = data

        print(f"  ✅ Loaded {len(onnx_weights)} weights from ONNX")

        # Map ONNX weights to PyTorch (with MatMul transpose handling)
        state_dict = {}
        pytorch_params = {name: param for name, param in model.named_parameters()}

        onnx_to_pytorch_map = {
            "conv_stem_branch1_0_weight": "conv_stem.branch1.0.weight",
            "conv_stem_branch1_3_weight": "conv_stem.branch1.3.weight",
            "conv_stem_branch2_0_weight": "conv_stem.branch2.0.weight",
            "conv_stem_branch2_3_weight": "conv_stem.branch2.3.weight",
            "conv_stem_branch3_0_weight": "conv_stem.branch3.0.weight",
            "conv_stem_branch3_3_weight": "conv_stem.branch3.3.weight",
            "pos_embed": "pos_embed",
            "node_0_Expand__0": "cls_token",
            "cls_selector": "cls_selector",
            "encoder_ln_1_weight": "encoder.ln_1.weight",
            "encoder_ln_1_bias": "encoder.ln_1.bias",
            "onnx__MatMul_109": "encoder.mha.q_proj.weight",
            "onnx__MatMul_110": "encoder.mha.k_proj.weight",
            "onnx__MatMul_111": "encoder.mha.v_proj.weight",
            "onnx__MatMul_112": "encoder.mha.out_proj.weight",
            "onnx__MatMul_113": "encoder.ff.ff1.weight",
            "onnx__MatMul_114": "encoder.ff.ff2.weight",
            "classifier_lin1_weight": "classifier.lin1.weight",
            "classifier_lin1_bias": "classifier.lin1.bias",
        }

        for onnx_name, pytorch_name in onnx_to_pytorch_map.items():
            if onnx_name in onnx_weights and pytorch_name in pytorch_params:
                onnx_weight = onnx_weights[onnx_name]

                # CRITICAL: ONNX MatMul weights must be transposed for PyTorch Linear
                needs_transpose = "MatMul" in onnx_name and ".weight" in pytorch_name

                if needs_transpose and len(onnx_weight.shape) == 2:
                    state_dict[pytorch_name] = torch.from_numpy(onnx_weight.T.copy()).float()
                else:
                    state_dict[pytorch_name] = torch.from_numpy(onnx_weight.copy()).float()

        # Load mapped weights
        model.load_state_dict(state_dict, strict=False)
        print(f"  ✅ Loaded {len(state_dict)} weights into PyTorch model")

        # Apply ONNX weight-sharing optimization
        if "encoder.ln_1.weight" in state_dict and "encoder.ln_1.bias" in state_dict:
            ln_weight = state_dict["encoder.ln_1.weight"]
            ln_bias = state_dict["encoder.ln_1.bias"]

            # All LayerNorms reuse encoder.ln_1 weights in ONNX
            if hasattr(model.encoder, "ln_2"):
                model.encoder.ln_2.weight.data.copy_(ln_weight)
                model.encoder.ln_2.bias.data.copy_(ln_bias)

            if hasattr(model, "norm"):
                model.norm.weight.data.copy_(ln_weight)
                model.norm.bias.data.copy_(ln_bias)

            # All Linear biases use encoder.ln_1.bias in ONNX
            if (
                hasattr(model.encoder.mha.out_proj, "bias")
                and model.encoder.mha.out_proj.bias is not None
            ):
                model.encoder.mha.out_proj.bias.data.copy_(ln_bias)

            if hasattr(model.encoder.ff.ff1, "bias") and model.encoder.ff.ff1.bias is not None:
                model.encoder.ff.ff1.bias.data.copy_(ln_bias)

            if hasattr(model.encoder.ff.ff2, "bias") and model.encoder.ff.ff2.bias is not None:
                model.encoder.ff.ff2.bias.data.copy_(ln_bias)

            print(f"  ✅ Applied ONNX weight-sharing (ln_1 → ln_2, norm, Linear biases)")

        # Generate output with ONNX weights
        model.eval()
        with torch.no_grad():
            input_tensor = torch.from_numpy(test_input).float()
            output_tensor = model(input_tensor)
            test_output = output_tensor.numpy()

        # Save as .npz files
        np.savez(save_path / "inputs.npz", input=test_input)
        np.savez(save_path / "outputs.npz", output=test_output)

        print(f"  ✅ Saved: {save_path / 'inputs.npz'} shape={test_input.shape}")
        print(f"  ✅ Saved: {save_path / 'outputs.npz'} shape={test_output.shape}")
        print(f"  📊 Output values: {test_output.flatten()}")
