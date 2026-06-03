# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""T-SDR (SpokenNumberRecognizer) fp32 ONNX Exporter."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..core.base_exporter import BaseONNXExporter
from .pytorch_models.tsdr import SpokenNumberRecognizer


class TSDRExporter(BaseONNXExporter):
    """ONNX exporter for the fp32 SpokenNumberRecognizer (T-SDR).

    Input: (batch_size, 80, time_steps) — 80-band mel spectrogram.
    """

    def __init__(self, save_path: str = None, config_file: str = "config.yaml"):
        super().__init__(save_path, config_file)
        self.model_config = {}

    def load_config(self) -> Dict[str, Any]:
        config = {
            "batch_size": 1,
            "mel_bands": 80,
            "time_steps": 101,       # Fixed for ONNX export
            "num_classes": 10,
            "d_model": 128,
            "nhead": 8,
            "num_layers": 4,
            "max_len": 5000,
            "opset_version": 17,
            "training_strategy": "full",
            "custom_trainable_params": [],
            "zo": {"epsilon": 0.1, "seed": 42},
        }
        self.model_config = config
        return config

    def create_model(self) -> torch.nn.Module:
        return SpokenNumberRecognizer(
            num_classes=self.model_config["num_classes"],
            d_model=self.model_config["d_model"],
            nhead=self.model_config["nhead"],
            num_layers=self.model_config["num_layers"],
            max_len=self.model_config["max_len"],
        )

    def get_input_shape(self) -> Tuple[int, ...]:
        return (
            self.config["batch_size"],
            self.config["mel_bands"],
            self.config["time_steps"],
        )

    def get_trainable_params(self, all_param_names: List[str]) -> List[str]:
        strategy = self.config.get("training_strategy", "full")
        if strategy == "last_layer":
            return [n for n in all_param_names if n in ("fc.weight", "fc.bias")]
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
            test_output = model(torch.from_numpy(test_input)).numpy()
        if was_training:
            model.train()

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        np.savez(save_path / "inputs.npz", input=test_input)
        np.savez(save_path / "outputs.npz", output=test_output)
        print(f"  Saved inputs.npz {test_input.shape}, outputs.npz {test_output.shape}")
