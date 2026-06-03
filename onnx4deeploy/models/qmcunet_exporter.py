# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""QMCUNet-In1 Brevitas ONNX Exporter.

Exports QMCUNetIn1 — an INT8 Brevitas-quantized version of MCUNet-In1 — using
Export4Deeploy.exportBrevitas, producing the same RequantShift-based ONNX
format as QSleepConViT and QTSDR.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
import torch
from brevitas.quant_tensor import QuantTensor

from DeepQuant.Export4Deeploy import exportBrevitas
from ..core.base_exporter import BaseONNXExporter, ExportMode
from ..core.onnx_utils import print_model_info
from ..optimization.shape_optimizer import infer_shapes_with_custom_ops
from ..transform.quant_transform import fix_duplicate_tensor_names
from .pytorch_models.mcunet import QMCUNetIn1

_DEFAULT_WEIGHTS = "/app/on-device-learning/weights/MCUNet-In1_CIFAR10.pth"


class QMCUNetExporter(BaseONNXExporter):
    """ONNX exporter for INT8 Brevitas-quantized MCUNet-In1 (CIFAR-10, 96×96 RGB)."""

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
            # Optional: path to QAT-trained weights.
            # "weights_path": _DEFAULT_WEIGHTS,
        }
        self.model_config = config
        return config

    def create_model(self) -> torch.nn.Module:
        return QMCUNetIn1(num_classes=self.model_config["num_classes"])

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
            return [n for n in all_param_names if "fc" in n]
        if strategy == "custom":
            custom = self.config.get("custom_trainable_params", [])
            return [n for n in all_param_names if n in custom]
        return all_param_names

    def _get_config_string(self) -> str:
        return (
            f"_{self.config['input_height']}x{self.config['input_width']}"
            f"_{self.config['num_classes']}cls"
        )

    def save_test_data(self, model: torch.nn.Module, save_dir: str):
        print("Saving test input/output data...")
        input_shape = self.get_input_shape()
        test_input = np.random.randn(*input_shape).astype(np.float32)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(test_input))
            if isinstance(out, QuantTensor):
                out = out.value
            test_output = out.numpy()
        if was_training:
            model.train()

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        np.savez(save_path / "inputs.npz", input=test_input)
        np.savez(save_path / "outputs.npz", output=test_output)
        print(f"  Saved inputs.npz {test_input.shape}, outputs.npz {test_output.shape}")

    def export_inference(self, save_path: Optional[str] = None, quant: bool = False) -> str:
        if not quant:
            # fp32 path — QMCUNetIn1 is Brevitas-only so use quant=True regardless.
            quant = True

        if save_path:
            self.save_path = save_path
        self.config = self.load_config()
        self.paths = self.setup_paths(ExportMode.INFERENCE)

        print(f"\n{'='*60}")
        print("Exporting QMCUNet-In1 to ONNX (Brevitas INT8)")
        print(f"{'='*60}\n")

        model = self.create_model()
        model.eval()

        input_shape = self.get_input_shape()
        input_tensor = torch.randn(*input_shape, dtype=torch.float32)
        print(f"   Input shape: {input_shape}")

        weights_path = self.config.get("weights_path", "")
        if weights_path and os.path.exists(weights_path):
            print(f"   Loading weights: {weights_path}")
            model.load_state_dict(
                torch.load(weights_path, map_location="cpu"), strict=False
            )
        else:
            print("   No weights_path; using random weights.")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if "bias" in name:
                        torch.nn.init.uniform_(param, a=0.01, b=0.02)
                    else:
                        torch.nn.init.normal_(param, mean=0.0, std=0.02)

        print("\nExporting via Export4Deeploy.exportBrevitas...")
        onnx_model = exportBrevitas(model, input_tensor, debug=False)

        onnx.save(onnx_model, self.paths["network"])
        print(f"ONNX model saved: {self.paths['network']}")

        print("\nRunning inference optimizations...")
        self.run_inference_optimization(self.paths["network"], self.paths["network"])

        print("\nRunning shape inference...")
        infer_shapes_with_custom_ops(self.paths["network"], self.paths["network"])

        _m = onnx.load(self.paths["network"])
        _m = fix_duplicate_tensor_names(_m)
        with open(self.paths["network"], "wb") as _f:
            _f.write(_m.SerializeToString())

        try:
            self.save_test_data(model, self.paths["output_dir"])
        except Exception as e:
            print(f"Failed to save test data: {e}")

        print_model_info(self.paths["network"])

        print(f"\n{'='*60}")
        print(f"Export Complete: {self.paths['network']}")
        print(f"{'='*60}\n")

        return self.paths["network"]
