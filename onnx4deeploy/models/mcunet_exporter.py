# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""MCUNet-In1 ONNX Exporter.

fp32 (infer): builds via ProxylessNASNets.build_from_config, loads the
  CIFAR-10 pretrained weights, replaces the Sequential(Dropout, Linear)
  classifier head with a clean Linear(160, 10), folds all BatchNorm layers
  into the preceding Conv2d layers, then exports.

int8 (q-infer): exports fp32 ONNX then applies OnnxRuntime static PTQ.
"""

import io
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
import torch
import torch.nn as nn

from ..core.base_exporter import BaseONNXExporter, ExportMode
from ..core.onnx_utils import print_model_info
from .pytorch_models.mcunet import build_mcunet_in1

# Default path to the pretrained CIFAR-10 weights shipped with on-device-learning.
_DEFAULT_WEIGHTS = "/app/on-device-learning/weights/MCUNet-In1_CIFAR10.pth"


def _replace_classifier_head(model: nn.Module, num_classes: int = 10, dropout_p: float = 0.1) -> nn.Module:
    """Replace classifier.linear with Sequential(Dropout, Linear(in_f, num_classes)).

    This matches the key structure of the pretrained CIFAR-10 checkpoint
    (classifier.linear.0 = Dropout, classifier.linear.1 = Linear).
    """
    if (hasattr(model, "classifier")
            and hasattr(model.classifier, "linear")
            and isinstance(model.classifier.linear, nn.Linear)):
        in_f = model.classifier.linear.in_features
        model.classifier.linear = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_f, num_classes),
        )
    return model


def _fold_bn_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
    """Fold BN parameters into conv weight and bias in-place."""
    scale = bn.weight.data / (bn.running_var.data + bn.eps).sqrt()  # gamma / std
    conv.weight.data.mul_(scale.view(-1, 1, 1, 1))
    bias = conv.bias.data if conv.bias is not None else torch.zeros_like(bn.running_mean)
    conv.bias = nn.Parameter(scale * (bias - bn.running_mean) + bn.bias.data)


def fold_batchnorms(model: nn.Module) -> nn.Module:
    """Fold every Conv2d immediately followed by BatchNorm2d in the same parent module.

    Replaces the BatchNorm2d with nn.Identity() so the graph topology is unchanged
    but the BN parameters are absorbed into the Conv bias.  The identity nodes are
    later stripped by the inference optimisation pipeline.
    """
    model.eval()
    for parent in model.modules():
        children = list(parent._modules.items())
        for i in range(len(children) - 1):
            _, layer_a = children[i]
            name_b, layer_b = children[i + 1]
            if isinstance(layer_a, nn.Conv2d) and isinstance(layer_b, nn.BatchNorm2d):
                _fold_bn_into_conv(layer_a, layer_b)
                parent._modules[name_b] = nn.Identity()
    return model


class MCUNetExporter(BaseONNXExporter):
    """ONNX exporter for MCUNet-In1 (CIFAR-10, 96×96 RGB input)."""

    def __init__(self, save_path: str = None, config_file: str = "config.yaml"):
        super().__init__(save_path, config_file)
        self.model_config = {}

    def load_config(self) -> Dict[str, Any]:
        config = {
            "batch_size": 1,
            "input_channels": 3,
            "input_height": 96,
            "input_width": 96,
            "num_classes": 10,
            "opset_version": 17,
            "training_strategy": "full",
            "custom_trainable_params": [],
            "zo": {"epsilon": 0.1, "seed": 42},
            "weights_path": _DEFAULT_WEIGHTS,
        }
        self.model_config = config
        return config

    def create_model(self) -> nn.Module:
        return build_mcunet_in1(num_classes=self.model_config["num_classes"])

    def get_input_shape(self) -> Tuple[int, ...]:
        return (
            self.config["batch_size"],
            self.config["input_channels"],
            self.config["input_height"],
            self.config["input_width"],
        )

    def get_trainable_params(self, all_param_names: List[str]) -> List[str]:
        strategy = self.config.get("training_strategy", "full")
        if strategy == "last_layer":
            return [n for n in all_param_names if "classifier" in n]
        if strategy == "custom":
            custom = self.config.get("custom_trainable_params", [])
            return [n for n in all_param_names if n in custom]
        return all_param_names

    def _get_config_string(self) -> str:
        return (
            f"_{self.config['input_height']}x{self.config['input_width']}"
            f"_{self.config['num_classes']}cls"
        )

    def save_test_data(self, model: nn.Module, save_dir: str):
        print("Saving test input/output data...")
        input_shape = self.get_input_shape()
        test_input = np.random.randn(*input_shape).astype(np.float32)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(test_input))
            test_output = out.numpy() if isinstance(out, torch.Tensor) else out.value.numpy()
        if was_training:
            model.train()

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        np.savez(save_path / "inputs.npz", input=test_input)
        np.savez(save_path / "outputs.npz", output=test_output)
        print(f"  Saved inputs.npz {test_input.shape}, outputs.npz {test_output.shape}")

    def export_inference(self, save_path: Optional[str] = None, quant: bool = False) -> str:
        if quant:
            return self._export_ptq(save_path)

        if save_path:
            self.save_path = save_path
        self.config = self.load_config()
        self.paths = self.setup_paths(ExportMode.INFERENCE)

        print(f"\n{'='*60}")
        print("Exporting MCUNet to ONNX (Inference Mode)")
        print(f"{'='*60}\n")

        # 1. Build model from the mcunet ProxylessNASNets config.
        model = self.create_model()

        # 2. Apply the Sequential(Dropout, Linear) head so the state-dict keys
        #    match the CIFAR-10 checkpoint (classifier.linear.0/1).
        model = _replace_classifier_head(model, num_classes=self.config["num_classes"], dropout_p=0.1)

        # 3. Load pretrained CIFAR-10 weights.
        weights_path = self.config.get("weights_path", "")
        if not weights_path or not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"MCUNet-In1 pretrained weights not found: {weights_path}\n"
                "Set 'weights_path' in the config to the MCUNet-In1_CIFAR10.pth file."
            )
        print(f"Loading pretrained weights: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)

        # 4. Strip the Dropout wrapper — keep only the Linear for clean ONNX export.
        seq = model.classifier.linear  # Sequential([Dropout, Linear])
        model.classifier.linear = seq[1]
        print(f"Classifier head: Linear({model.classifier.linear.in_features} -> "
              f"{model.classifier.linear.out_features})")

        # 5. Fold every Conv2d+BN2d pair into a single bias-aware Conv2d.
        print("Folding BatchNorm layers into Conv layers...")
        model.eval()
        fold_batchnorms(model)

        # 6. Export to ONNX.
        input_shape = self.get_input_shape()
        input_tensor = torch.randn(*input_shape, dtype=torch.float32)
        print(f"   Input shape: {input_shape}")
        print("\nExporting to ONNX...")
        opset_version = self.config.get("opset_version", 17)
        onnx_model = self._export_to_onnx(model, input_tensor, opset_version)
        onnx.save(onnx_model, self.paths["network"])
        print(f"ONNX model saved: {self.paths['network']}")

        # 7. Inference optimisations (removes Identity nodes, renames, etc.).
        print("\nRunning inference optimizations...")
        self.run_inference_optimization(self.paths["network"], self.paths["network"])

        # 8. Shape inference.
        print("\nRunning shape inference...")
        from ..optimization.shape_optimizer import infer_shapes_with_custom_ops
        infer_shapes_with_custom_ops(self.paths["network"], self.paths["network"])

        # 9. Save reference I/O for downstream testing.
        try:
            self.save_test_data(model, self.paths["output_dir"])
        except Exception as e:
            print(f"Failed to save test data: {e}")

        print_model_info(self.paths["network"])

        print(f"\n{'='*60}")
        print(f"Export Complete! Final model: {self.paths['network']}")
        print(f"{'='*60}\n")

        return self.paths["network"]

    def _export_ptq(self, save_path: Optional[str] = None) -> str:
        """Export MCUNet-In1 using OnnxRuntime static post-training quantization."""
        if save_path:
            self.save_path = save_path

        self.config = self.load_config()
        self.paths = self.setup_paths(ExportMode.INFERENCE)

        print(f"\n{'='*60}")
        print("Exporting MCUNet-In1 to ONNX (INT8 PTQ mode)")
        print(f"{'='*60}\n")

        model = self.create_model()
        model.eval()

        weights_path = self.config.get("weights_path", "")
        if weights_path and os.path.exists(weights_path):
            print(f"Loading pre-trained weights from {weights_path}")
            model = _replace_classifier_head(model, num_classes=self.config["num_classes"], dropout_p=0.1)
            model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
            seq = model.classifier.linear
            model.classifier.linear = seq[1]
        else:
            print("No weights_path found; using random weights for PTQ calibration.")
            for _, param in model.named_parameters():
                if param.requires_grad:
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)

        input_shape = self.get_input_shape()
        input_tensor = torch.randn(*input_shape)

        print("Exporting fp32 ONNX...")
        opset = self.config.get("opset_version", 17)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp_path = tmp.name

        f = io.BytesIO()
        torch.onnx.export(
            model, input_tensor, f,
            input_names=["input"], output_names=["output"],
            opset_version=opset, do_constant_folding=True,
            export_params=True, keep_initializers_as_inputs=False,
        )
        with open(tmp_path, "wb") as fout:
            fout.write(f.getvalue())
        print(f"  fp32 ONNX saved (temp): {tmp_path}")

        print("Applying OnnxRuntime static PTQ...")
        quant_path = self.paths["network"]
        self._apply_static_ptq(tmp_path, quant_path, input_shape)
        os.remove(tmp_path)
        print(f"  INT8 ONNX saved: {quant_path}")

        print("Running inference optimizations...")
        self.run_inference_optimization(quant_path, quant_path)

        print("Running shape inference...")
        from ..optimization.shape_optimizer import infer_shapes_with_custom_ops
        infer_shapes_with_custom_ops(quant_path, quant_path)

        try:
            self.save_test_data(model, self.paths["output_dir"])
        except Exception as e:
            print(f"Failed to save test data: {e}")

        print(f"\n{'='*60}")
        print(f"INT8 PTQ export complete: {quant_path}")
        print(f"{'='*60}\n")
        return quant_path

    @staticmethod
    def _apply_static_ptq(fp32_onnx_path: str, output_path: str, input_shape: Tuple):
        """Apply OnnxRuntime static post-training quantization with random calibration data."""
        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantFormat,
            QuantType,
            quantize_static,
        )

        class _RandomCalibrationReader(CalibrationDataReader):
            def __init__(self, shape, num_samples: int = 64):
                self._data = iter([
                    {"input": np.random.randn(*shape).astype(np.float32)}
                    for _ in range(num_samples)
                ])

            def get_next(self):
                return next(self._data, None)

        quantize_static(
            model_input=fp32_onnx_path,
            model_output=output_path,
            calibration_data_reader=_RandomCalibrationReader(input_shape),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
        )
