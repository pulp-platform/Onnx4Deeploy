# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Tiny Transformer Exporter — small patch-based Vision Transformer for MNIST.

Design goals:
- Fast compilation: ~10 K parameters → TrainingNetwork.c ≪ 1 MB
- MNIST-native: 28×28 images split into 16 patches of 7×7 (patch_dim=49)
- Clear loss descent: with data_size≥5 + epoch cycling, loss descends from
  ~2.3 (random) toward 0 within a few dozen mini-batches.
- Deeploy-compatible: only MatMul/Gemm, LayerNorm, ReLU, Softmax, Add —
  uses attention pooling instead of ReduceMean to avoid gradient shape issues.

Typical command:
    python Onnx4Deeploy.py --model TinyTransformer --mode train \\
        --dataset mnist --data-size 5 --n-epochs 20 --n-accum 8
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from ..core.base_exporter import BaseONNXExporter
from .pytorch_models.tiny_transformer import TinyTransformerMnist


class TinyTransformerExporter(BaseONNXExporter):
    """ONNX exporter for the Tiny Patch Transformer (MNIST)."""

    def __init__(self, save_path: Optional[str] = None, config_file: str = "config.yaml"):
        super().__init__(save_path, config_file)

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    def load_config(self) -> Dict[str, Any]:
        """Return default Tiny Transformer configuration."""
        config = {
            "batch_size": 1,
            # Patch grid: img_size must be divisible by patch_size.
            # Default: 28×28 MNIST → 4×4 grid of 7×7 patches (16 patches, patch_dim=49)
            "img_size": 28,
            "patch_size": 7,
            # Model size knobs
            "embed_dim": 32,
            "ffn_hidden": 64,
            "num_classes": 10,
            "opset_version": 17,
            # Training configuration
            "training_strategy": "full",  # "full" | "head" | "custom" — see get_trainable_params()
            "custom_trainable_params": [],
            # Training loop
            "learning_rate": 0.001,
            "n_batches": 4,
            "n_accum": 1,
            # Data source
            "dataset": "random",  # "random" | "mnist"
            "data_path": None,
            "data_split": "train",
            "data_size": None,
            "classes": None,  # list of int labels to restrict MNIST, e.g. [0, 8]
        }

        if hasattr(self, "_config_overrides") and self._config_overrides:
            config.update(self._config_overrides)

        return config

    def create_model(self) -> torch.nn.Module:
        """Create the TinyTransformerMnist PyTorch model."""
        cfg = self.config
        patch_size = cfg["patch_size"]
        img_size = cfg["img_size"]
        num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size

        return TinyTransformerMnist(
            num_patches=num_patches,
            patch_dim=patch_dim,
            embed_dim=cfg["embed_dim"],
            ffn_hidden=cfg["ffn_hidden"],
            num_classes=cfg["num_classes"],
        )

    def get_input_shape(self) -> Tuple[int, ...]:
        """Return (batch_size, num_patches, patch_dim).

        For default config: (1, 16, 49) — matches flattened MNIST pixels when
        num_patches × patch_dim = 16 × 49 = 784.
        """
        cfg = self.config
        patch_size = cfg["patch_size"]
        img_size = cfg["img_size"]
        num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size
        return (cfg["batch_size"], num_patches, patch_dim)

    def get_trainable_params(self, all_param_names: List[str]) -> List[str]:
        """Select trainable parameters using pattern-based strategy.

        Strategies:
            "full"   — train all parameters (default; safe for PULP L2 since no frozen weights).
            "head"   — train only the classifier head (freeze all other params).
            "custom" — train only params listed in config["custom_trainable_params"].
        """
        strategy = self.config.get("training_strategy", "full")

        _FREEZE = {
            # Train everything — all ~10K params, no frozen constants
            "full": lambda n: False,
            # Freeze everything except the classifier head
            "head": lambda n: "classifier" not in n,
            # Explicit allow-list from config
            "custom": lambda n: n not in self.config.get("custom_trainable_params", []),
        }

        if strategy not in _FREEZE:
            print(f"  ⚠️  Unknown strategy '{strategy}', using 'full'")
            strategy = "full"

        requires_grad = [n for n in all_param_names if not _FREEZE[strategy](n)]
        frozen = [n for n in all_param_names if _FREEZE[strategy](n)]

        print(f"\n  Training Strategy: '{strategy}'")
        print(
            f"   Total: {len(all_param_names)}  Trainable: {len(requires_grad)}  Frozen: {len(frozen)}"
        )
        if frozen:
            print(f"   Frozen (→ constant): {frozen}")

        return requires_grad

    # ------------------------------------------------------------------
    # Data source: MNIST or random
    # ------------------------------------------------------------------

    def get_data_source(self):
        """Return MNISTDataSource when dataset='mnist', else RandomDataSource."""
        dataset = (self.config or {}).get("dataset", "random")
        if dataset == "mnist":
            from ..data.mnist_datasource import MNISTDataSource

            cfg = self.config or {}
            return MNISTDataSource(
                data_path=cfg.get("data_path", None),
                split=cfg.get("data_split", "train"),
                data_size=cfg.get("data_size", None),
                classes=cfg.get("classes", None),
            )
        from ..data.random_datasource import RandomDataSource

        return RandomDataSource()

    # ------------------------------------------------------------------
    # Multi-batch training test data (gradient accumulation loop)
    # ------------------------------------------------------------------

    def create_training_test_data(
        self, n_batches: int = None, num_data_inputs: int = 2, n_accum: int = None
    ) -> None:
        """Multi-batch training test data with gradient accumulation support.

        Delegates to SimpleMlpExporter's implementation — the algorithm
        (accumulation loop, npz layout, meta keys) is identical; all model-
        specific config is already set on self.config / self.paths.
        """
        from .simple_mlp_exporter import SimpleMlpExporter

        SimpleMlpExporter.create_training_test_data(
            self, n_batches=n_batches, num_data_inputs=num_data_inputs, n_accum=n_accum
        )

    # ------------------------------------------------------------------
    # Optional helpers
    # ------------------------------------------------------------------

    def _get_config_string(self) -> str:
        cfg = self.config or {}
        return (
            f"_{cfg.get('img_size', 28)}"
            f"_p{cfg.get('patch_size', 7)}"
            f"_d{cfg.get('embed_dim', 32)}"
        )
