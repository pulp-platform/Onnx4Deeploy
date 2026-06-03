# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Q-T-SDR (QSpokenNumberRecognizer) INT8 Brevitas ONNX Exporter."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
import torch
from brevitas.quant_tensor import QuantTensor

from DeepQuant.Export4Deeploy import exportBrevitas
from ..core.base_exporter import BaseONNXExporter, ExportMode
from ..optimization.shape_optimizer import infer_shapes_with_custom_ops
from ..transform.quant_transform import fix_duplicate_tensor_names
from .pytorch_models.tsdr import QSpokenNumberRecognizer


class QTSDRExporter(BaseONNXExporter):
    """ONNX exporter for the INT8 QSpokenNumberRecognizer (Q-T-SDR).

    Uses Brevitas QAT + DeepQuant's exportBrevitas, following the same
    approach as QSleepConViTExporter.

    Input: (batch_size, 80, time_steps) — 80-band mel spectrogram.
    """

    def __init__(self, save_path: str = None, config_file: str = "config.yaml"):
        super().__init__(save_path, config_file)
        self.model_config = {}

    def load_config(self) -> Dict[str, Any]:
        config = {
            "batch_size": 1,
            "mel_bands": 80,
            "time_steps": 101,
            "num_classes": 10,
            "d_model": 128,
            "nhead": 8,
            "num_layers": 3,             # Matches SpokenDigitTransformer
            "dim_feedforward": 192,      # d_model + d_model//2
            "max_len": 5000,
            "opset_version": 17,
            "training_strategy": "full",
            "custom_trainable_params": [],
            "zo": {
                "epsilon": 0.1,
                "seed": 42,
                "exceptions": ["node_matmul", "node_bmm_requant", "node_bmm_1_requant"],
            },
            # Set weights_path to load pre-trained Brevitas QAT weights.
            # "weights_path": "path/to/qtsdr_weights.pth",
            # "scales_path": "path/to/qtsdr_scales.json",
        }
        self.model_config = config
        return config

    def create_model(self) -> torch.nn.Module:
        return QSpokenNumberRecognizer(
            num_classes=self.model_config["num_classes"],
            d_model=self.model_config["d_model"],
            nhead=self.model_config["nhead"],
            num_layers=self.model_config["num_layers"],
            dim_feedforward=self.model_config["dim_feedforward"],
            max_len=self.model_config["max_len"],
            time_steps=self.model_config["time_steps"],
        )

    def get_input_shape(self) -> Tuple[int, ...]:
        return (
            self.config["batch_size"],
            self.config["mel_bands"],
            self.config["time_steps"],
            1,  # dummy spatial dim for Conv2d compatibility
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
            f"_{self.config['mel_bands']}mel"
            f"_{self.config['time_steps']}t"
            f"_{self.config['d_model']}d"
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
            return super().export_inference(save_path, quant=False)

        if save_path:
            self.save_path = save_path
        self.config = self.load_config()
        self.paths = self.setup_paths(ExportMode.INFERENCE)

        print(f"\n{'='*60}")
        print("🚀 Exporting QTSDR to ONNX (Inference Mode)")
        print(f"{'='*60}\n")

        model = self.create_model()
        model.eval()
        input_shape = self.get_input_shape()
        input_tensor = torch.randn(*input_shape, dtype=torch.float32)
        print(f"   Input shape: {input_shape}")

        weights_path = self.config.get("weights_path")
        if weights_path:
            abs_path = weights_path if os.path.isabs(weights_path) else os.path.join(os.getcwd(), weights_path)
            if os.path.exists(abs_path):
                state_dict = torch.load(abs_path, map_location="cpu")
                model.load_state_dict(state_dict, strict=False)
                print(f"   Loaded weights: {abs_path}")
            else:
                print(f"   weights_path not found ({abs_path}), using random weights.")
                self._init_random_weights(model)
        else:
            print("   No weights_path; using random weights.")
            self._init_random_weights(model)

        print("\n📤 Exporting to ONNX via exportBrevitas...")
        onnx_model = exportBrevitas(model, input_tensor, debug=False)

        onnx.save(onnx_model, self.paths["network"])
        print(f"✅ ONNX model saved: {self.paths['network']}")

        print("\n🔧 Running inference optimizations...")
        self.run_inference_optimization(self.paths["network"], self.paths["network"])

        print("\n🔍 Running shape inference...")
        infer_shapes_with_custom_ops(self.paths["network"], self.paths["network"])

        _m = onnx.load(self.paths["network"])
        _m = fix_duplicate_tensor_names(_m)
        with open(self.paths["network"], "wb") as _f:
            _f.write(_m.SerializeToString())

        try:
            self.save_test_data(model, self.paths["output_dir"])
        except Exception as e:
            print(f"⚠️  Failed to save test data: {e}")

        return self.paths["network"]

    @staticmethod
    def _init_random_weights(model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "bias" in name:
                    torch.nn.init.uniform_(param, a=0.01, b=0.02)
                else:
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
