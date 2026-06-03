# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""QLite CNN Model Exporter."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.onnx.utils
from brevitas.quant_tensor import QuantTensor

from DeepQuant.ExportBrevitas import exportBrevitas


from ..core.base_exporter import BaseONNXExporter

# Import QLiteCNN PyTorch model from new location
from .pytorch_models.lightweight_cnn import QLiteCNN
import brevitas.onnx as bo
import json
import onnx
import onnx_graphsurgeon as gs
from onnx4deeploy.transform.quant_transform import replace_qdq_with_deeploy, insert_rqs_from_map


class QLiteCnnExporter(BaseONNXExporter):
    """ONNX exporter for QLite CNN model."""

    def __init__(self, save_path: str = None, config_file: str = "config.yaml"):
        """
        Initialize QLite CNN exporter.

        Args:
            save_path: Optional custom path to save ONNX files
            config_file: Path to configuration YAML file
        """
        super().__init__(save_path, config_file)
        self.model_config = {}

    def load_config(self) -> Dict[str, Any]:
        """
        Load QLite CNN configuration.

        Returns:
            Dictionary containing QLite CNN configuration parameters
        """
        # Default QLite CNN configuration
        config = {
            "batch_size": 1,
            "input_height": 28,
            "input_width": 28,
            "input_channels": 1,  # Grayscale images
            "num_classes": 10,
            "opset_version": 17,
            "dropout": 0.0,  # No dropout for inference
            # Training configuration
            "training_strategy": "full",  # Options: "full", "last_layer", "custom"
            "custom_trainable_params": [],
            "zo": {
                "epsilon": 0.1,
                "seed": 42
            },
            "weights_path":"onnx4deeploy/models/pytorch_models/lightweight_cnn/qlite_cnn.pth",
            "scales_path":"onnx4deeploy/models/pytorch_models/lightweight_cnn/qlite_cnn_scales.json"
        }

        self.model_config = config
        return config

    def create_model(self) -> torch.nn.Module:
        """
        Create QLite CNN PyTorch model.

        Returns:
            QLite CNN model ready for export
        """
        model = QLiteCNN(
            batch_size=self.model_config["batch_size"],
            input_channels=self.model_config["input_channels"],
            num_classes=self.model_config["num_classes"],
            dropout=self.model_config["dropout"],
        )

        return model

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the input tensor shape for QLite CNN.

        Returns:
            Tuple representing input shape (batch_size, channels, height, width)
        """
        batch_size = self.config["batch_size"]
        channels = self.config["input_channels"]
        height = self.config["input_height"]
        width = self.config["input_width"]
        return (batch_size, channels, height, width)

    def get_trainable_params(self, all_param_names: List[str]) -> List[str]:
        """
        Get list of trainable parameter names for QLite CNN.

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
                "fc.weight",
                "fc.bias",
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
            Configuration string like "_28x28_1ch_10"
        """
        return (
            f"_{self.config['input_height']}x{self.config['input_width']}"
            f"_{self.config['input_channels']}ch_{self.config['num_classes']}"
        )

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
            if isinstance(output_tensor, QuantTensor):
                output_tensor = output_tensor.value
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

    def _build_rqs_map(self, graph: gs.Graph, brevitas_scales: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate the flat Brevitas scales dump into the Deeploy edges map
        expected by insert_rqs_from_map.
        """
        rqs_map = {"edges": []}

        # Traverse the ONNX graph to find operators that require precision reduction
        for node in graph.nodes:
            if node.op in ["Conv", "Gemm", "MatMul"]:
                layer_name = None

                # Match ONNX node name (e.g., "/conv1/Conv") with Brevitas scale keys
                for key in brevitas_scales.keys():
                    if key.endswith(".weight_quant"):
                        base_name = key.split(".")[0]  # Extracts "conv1", "fc", etc.
                        if f"/{base_name}/" in node.name or node.name.startswith(base_name):
                            layer_name = base_name
                            break

                if layer_name:
                    in_scale = brevitas_scales.get(f"{layer_name}.input_quant")
                    w_scale = brevitas_scales.get(f"{layer_name}.weight_quant")
                    out_scale = brevitas_scales.get(f"{layer_name}.output_quant")


                    if in_scale is not None and w_scale is not None and out_scale is not None:
                        # Conv accumulation scale = input_scale * weight_scale

                        w_flat = np.array(w_scale).flatten()
                        src_scale = in_scale * w_flat

                        out_tensor_name = node.outputs[0].name

                        # We use out_tensor_name for both src and dst to intercept and rewire
                        # all nodes strictly consuming the output of this convolution.
                        rqs_map["edges"].append({
                            "src_tensor": out_tensor_name,
                            "dst_tensor": out_tensor_name,
                            "src_scale": src_scale.tolist(),
                            "dst_scale": float(out_scale)
                        })
        return rqs_map