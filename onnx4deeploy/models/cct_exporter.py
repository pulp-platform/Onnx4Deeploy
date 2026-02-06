"""CCT (Compact Convolutional Transformer) Model Exporter."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnx
import torch

from ..core.base_exporter import BaseONNXExporter
from ..optimization import remove_identity_nodes
from ..transform.model_transform import randomize_layernorm_params

# Import CCT PyTorch models from new location
from .pytorch_models.cct import cct_test


class CCTExporter(BaseONNXExporter):
    """ONNX exporter for CCT (Compact Convolutional Transformer) model."""

    def __init__(self, save_path: str = None, config_file: str = "config.yaml"):
        """
        Initialize CCT exporter.

        Args:
            save_path: Optional custom path to save ONNX files
            config_file: Path to configuration YAML file
        """
        super().__init__(save_path, config_file)
        self.model_config = {}

    def load_config(self) -> Dict[str, Any]:
        """
        Load CCT configuration.

        Returns:
            Dictionary containing CCT configuration parameters
        """
        # Default CCT configuration for testing
        config = {
            "batch_size": 1,
            "img_size": 32,
            "embedding_dim": 128,
            "num_heads": 2,
            "num_layers": 2,
            "num_classes": 10,
            "opset_version": 17,  # LayerNormalization requires opset 17+
            "n_conv_layers": 1,
            "kernel_size": 3,
            "positional_embedding": "learnable",
            # Training configuration
            "training_strategy": "linear",  # Options: "linear", "last_attention", "last_2_attention", "lora_block1", "lora_block2", "full", "custom"
            "custom_trainable_params": [],  # Used when training_strategy = "custom"
        }

        self.model_config = config
        return config

    def create_model(self) -> torch.nn.Module:
        """
        Create CCT PyTorch model.

        Returns:
            CCT model ready for export
        """
        model = cct_test(
            pretrained=False,
            img_size=self.model_config["img_size"],
            num_classes=self.model_config["num_classes"],
            embedding_dim=self.model_config["embedding_dim"],
            num_heads=self.model_config["num_heads"],
            num_layers=self.model_config["num_layers"],
            n_conv_layers=self.model_config.get("n_conv_layers", 1),
            positional_embedding=self.model_config.get("positional_embedding", "learnable"),
        )

        # Randomize LayerNorm parameters (for testing)
        model = randomize_layernorm_params(model)

        return model

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the input tensor shape for CCT.

        Returns:
            Tuple representing input shape (batch_size, channels, height, width)
        """
        batch_size = self.config["batch_size"]
        img_size = self.config["img_size"]
        return (batch_size, 3, img_size, img_size)

    def get_trainable_params(self, all_param_names: List[str]) -> List[str]:
        """
        Get list of trainable parameter names for CCT based on training strategy.

        Supports multiple training strategies for different fine-tuning scenarios:
        - "linear": Only train final classification layer (default)
        - "last_attention": Train last attention block + classifier
        - "last_2_attention": Train last 2 attention blocks + classifier
        - "lora_block1": LoRA-style training for block 1
        - "lora_block2": LoRA-style training for both blocks
        - "full": Train all parameters
        - "custom": Use custom_trainable_params from config

        Args:
            all_param_names: List of all parameter names in the model

        Returns:
            List of parameter names that should be trainable
        """
        strategy = self.config.get("training_strategy", "linear")

        # Define training strategies
        strategy_params = {
            "linear": [
                "classifier_fc_weight",
                "classifier_fc_bias",
            ],
            "last_attention": [
                "classifier_fc_weight",
                "classifier_fc_bias",
                "node_0_classifier_attention_pool_Transpose__0",
                "classifier_attention_pool_bias",
                "node_0_classifier_blocks_1_linear2_Transpose__0",
                "classifier_blocks_0_self_attn_proj_bias_Identity_31",
                "node_0_classifier_blocks_1_linear1_Transpose__0",
                "classifier_blocks_0_self_attn_proj_bias_Identity_33",
                "node_0_classifier_blocks_1_self_attn_proj_Transpose__0",
                "node_0_classifier_blocks_1_self_attn_v_proj_Transpose__0",
                "node_0_classifier_blocks_1_self_attn_k_proj_Transpose__0",
                "node_0_classifier_blocks_1_self_attn_q_proj_Transpose__0",
            ],
            "last_2_attention": [
                "classifier_fc_weight",
                "classifier_fc_bias",
                "node_0_classifier_attention_pool_Transpose__0",
                "classifier_attention_pool_bias",
                "node_0_classifier_blocks_1_linear2_Transpose__0",
                "classifier_blocks_0_self_attn_proj_bias_Identity_31",
                "node_0_classifier_blocks_1_linear1_Transpose__0",
                "classifier_blocks_0_self_attn_proj_bias_Identity_33",
                "node_0_classifier_blocks_1_self_attn_proj_Transpose__0",
                "node_0_classifier_blocks_1_self_attn_v_proj_Transpose__0",
                "node_0_classifier_blocks_1_self_attn_k_proj_Transpose__0",
                "node_0_classifier_blocks_1_self_attn_q_proj_Transpose__0",
                "node_0_classifier_blocks_0_linear2_Transpose__0",
                "classifier_blocks_0_self_attn_proj_bias_Identity_34",
                "node_0_classifier_blocks_0_linear1_Transpose__0",
                "node_0_classifier_blocks_0_self_attn_proj_Transpose__0",
                "classifier_blocks_0_self_attn_proj_bias__classifier_blocks_0_self_attn_proj_Add",
                "node_0_classifier_blocks_0_self_attn_k_proj_Transpose__0",
                "node_0_classifier_blocks_0_self_attn_q_proj_Transpose__0",
                "node_0_classifier_blocks_0_self_attn_v_proj_Transpose__0",
            ],
            "lora_block1": [
                "classifier_fc_weight",
                "classifier_fc_bias",
                "node_0_classifier_attention_pool_Transpose__0",
                "classifier_attention_pool_bias",
                "node_0_classifier_blocks_1_linear2_Transpose_1__0",
                "node_0_classifier_blocks_1_linear2_Transpose__0",
                "node_0_classifier_blocks_1_linear1_Transpose_1__0",
                "node_0_classifier_blocks_1_linear1_Transpose__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_11__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_10__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_4__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_3__0",
                "node_0_classifier_blocks_1_self_attn_Transpose__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_1__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_5__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_6__0",
            ],
            "lora_block2": [
                "classifier_fc_weight",
                "classifier_fc_bias",
                "node_0_classifier_attention_pool_Transpose__0",
                "classifier_attention_pool_bias",
                # Block 1
                "node_0_classifier_blocks_1_linear2_Transpose_1__0",
                "node_0_classifier_blocks_1_linear2_Transpose__0",
                "node_0_classifier_blocks_1_linear1_Transpose_1__0",
                "node_0_classifier_blocks_1_linear1_Transpose__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_11__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_10__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_4__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_3__0",
                "node_0_classifier_blocks_1_self_attn_Transpose__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_1__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_5__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_6__0",
                # Block 0
                "node_0_classifier_blocks_0_linear2_Transpose_1__0",
                "node_0_classifier_blocks_0_linear2_Transpose__0",
                "node_0_classifier_blocks_0_linear1_Transpose_1__0",
                "node_0_classifier_blocks_0_linear1_Transpose__0",
                "node_0_classifier_blocks_0_self_attn_Transpose_11__0",
                "node_0_classifier_blocks_0_self_attn_Transpose_10__0",
                "node_0_classifier_blocks_0_self_attn_Transpose_4__0",
                "node_0_classifier_blocks_0_self_attn_Transpose_3__0",
                "node_0_classifier_blocks_0_self_attn_Transpose__0",
                "node_0_classifier_blocks_0_self_attn_Transpose_1__0",
                "node_0_classifier_blocks_0_self_attn_Transpose_5__0",
                "node_0_classifier_blocks_0_self_attn_Transpose_6__0",
            ],
            "full": all_param_names,  # Train everything
            "custom": self.config.get("custom_trainable_params", []),
        }

        # Get trainable params based on strategy
        if strategy not in strategy_params:
            print(f"⚠️  Unknown training strategy '{strategy}', using 'linear' as fallback")
            strategy = "linear"

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
            Configuration string like "_32_128_2_2"
        """
        return f"_{self.config['img_size']}_{self.config['embedding_dim']}_{self.config['num_heads']}_{self.config['num_layers']}"

    def get_inference_pipeline(self):
        """
        Get CCT-specific inference optimization pipeline.

        CCT uses transformer-specific optimizations including:
        - Randomize initializers (for testing)
        - ONNX Runtime transformer optimizer (includes LayerNorm fusion)
        - Standard inference optimizations (GEMM, Identity, etc.)

        Returns:
            OptimizationPipeline configured for CCT inference
        """
        from ..core.optimization_passes import create_transformer_inference_pipeline

        # Create transformer-specific pipeline with model parameters
        pipeline = create_transformer_inference_pipeline(
            embedding_dim=self.config["embedding_dim"],
            num_heads=self.config["num_heads"],
            input_shape=self.get_input_shape(),
        )

        return pipeline

    def run_training_optimization(self, onnx_file: str, output_file: str):
        """
        Run ONNX optimizations for CCT training mode.

        Training-specific optimizations:
        - Shape inference for training graph
        - Identity node removal
        - Basic graph cleanup

        Args:
            onnx_file: Path to input ONNX file
            output_file: Path to save optimized ONNX file
        """
        print("🔧 Running CCT-specific training optimizations...")

        # Load model
        model = onnx.load(onnx_file)

        # 1. Shape inference
        print("  ➤ Shape inference for training graph...")
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception as e:
            print(f"    Warning: Shape inference failed (expected for training): {e}")

        # Save intermediate result
        onnx.save(model, output_file)

        # 2. Remove identity nodes
        print("  ➤ Removing identity nodes...")
        remove_identity_nodes(output_file, output_file)

        print("  ✅ Training optimization complete")

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
        # Temporarily switch to eval mode if in training mode
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
