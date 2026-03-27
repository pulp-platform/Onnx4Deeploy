# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Tests for QLiteCNN quantized model export.

Tests the full PTQ calibration (Brevitas) → Onnx4Deeploy export → numerical
verification pipeline for the QLiteCNN model.

Numerical verification is performed by ``run_onnx_graph`` from
``onnx_node_implementations``, a pure Python / PyTorch graph executor that
supports standard ONNX ops, Deeploy custom nodes (Quant, Dequant,
RequantShift) and the MeZO perturbation operators (PerturbNormal, etc.).
"""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from brevitas.quant_tensor import QuantTensor

from onnx4deeploy.models.pytorch_models.lightweight_cnn import QLiteCNN

from .onnx_node_implementations import run_onnx_graph
from .test_utils import load_and_check_onnx_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WEIGHTS_PATH = "onnx4deeploy/models/pytorch_models/lightweight_cnn/qlite_cnn.pth"
_INPUT_SHAPE = (1, 1, 28, 28)
_TOLERANCE = 1.0 / 2**8  # 1/256 ≈ 0.0039


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_brevitas_model(weights_path: str, num_classes: int = 10) -> torch.nn.Module:
    """Load QLiteCNN Brevitas model with pre-calibrated PTQ weights."""
    model = QLiteCNN(
        batch_size=1,
        input_channels=1,
        num_classes=num_classes,
        dropout=0.0,
    )
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _run_brevitas_inference(model: torch.nn.Module, test_input: np.ndarray) -> np.ndarray:
    """Run Brevitas model inference and return a float32 numpy array."""
    with torch.no_grad():
        output = model(torch.from_numpy(test_input))
        if isinstance(output, QuantTensor):
            output = output.value
    return output.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.quantized
class TestQLiteCNNQuantized:
    """Test QLiteCNN quantized inference export and numerical correctness."""

    def test_qlitecnn_ptq_export_and_numerical_correctness(
        self, model_test_dir, qlitecnn_config
    ):
        """
        End-to-end test: PTQ calibration → Onnx4Deeploy export → numerical check.

        Steps:
          1. Load the pre-calibrated Brevitas QLiteCNN model.
          2. Run the Onnx4Deeploy ``q-infer`` command to export the model to ONNX.
          3. Loop over 10 random input samples (seeds 0–9) and verify that the
             ONNX graph output matches the Brevitas reference within a tolerance
             of 1/2^8 for every sample.
        """
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        # ------------------------------------------------------------------
        # Step 1 – Load Brevitas model
        # ------------------------------------------------------------------
        print("\n[PTQ] Loading QLiteCNN Brevitas model with pre-calibrated weights...")
        weights_path = os.path.join(project_root, _WEIGHTS_PATH)
        model = _load_brevitas_model(
            weights_path, num_classes=qlitecnn_config["num_classes"]
        )
        print("[PTQ] Model loaded.")

        # ------------------------------------------------------------------
        # Step 2 – Run Onnx4Deeploy q-infer export command (once)
        # ------------------------------------------------------------------
        cli_script = os.path.join(project_root, "Onnx4Deeploy.py")
        cmd = [
            sys.executable, cli_script,
            "-model", "QLiteCNN",
            "-mode", "q-infer",
            "-o", model_test_dir,
        ]
        print(f"\n[Onnx4Deeploy] Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[Onnx4Deeploy] stdout:\n{result.stdout}")
            print(f"[Onnx4Deeploy] stderr:\n{result.stderr}")
            pytest.fail(
                f"Onnx4Deeploy command failed with return code {result.returncode}"
            )

        onnx_file = os.path.join(model_test_dir, "network.onnx")
        assert os.path.exists(onnx_file), f"ONNX file not found: {onnx_file}"

        # Verify basic ONNX validity (relaxed: skip strict check for custom ops)
        load_and_check_onnx_model(onnx_file, skip_shape_check=True)
        print(f"[Onnx4Deeploy] Export complete. ONNX saved at: {onnx_file}")

        # ------------------------------------------------------------------
        # Step 3 – Loop over 10 random inputs and check numerical correctness
        # ------------------------------------------------------------------
        print(
            "\n[Check] Running numerical check over 10 random input samples "
            "(seeds 0–9) ..."
        )
        failures = []
        for seed in range(10):
            rng = np.random.default_rng(seed)
            test_input = rng.standard_normal(_INPUT_SHAPE).astype(np.float32)

            brevitas_output = _run_brevitas_inference(model, test_input)
            onnx_output = run_onnx_graph(onnx_file, {"input": test_input})

            max_diff = float(np.max(np.abs(onnx_output - brevitas_output)))
            if max_diff > _TOLERANCE:
                failures.append(
                    f"  seed={seed}: max |onnx − brevitas| = {max_diff:.6f} "
                    f"(limit {_TOLERANCE:.6f})\n"
                    f"    Brevitas: {brevitas_output}\n"
                    f"    ONNX:     {onnx_output}"
                )
            else:
                print(
                    f"[Check] seed={seed} PASSED: "
                    f"max |onnx − brevitas| = {max_diff:.6f} ≤ {_TOLERANCE:.6f}"
                )

        if failures:
            pytest.fail(
                f"Numerical check FAILED for {len(failures)}/10 samples:\n"
                + "\n".join(failures)
            )
