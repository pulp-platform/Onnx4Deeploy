# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Pytest fixtures for model tests."""

import os
from pathlib import Path

import pytest


@pytest.fixture
def model_test_dir(request):
    """Create directory for model test artifacts.

    By default, saves to: /app/Onnx4Deeploy/test_outputs/<test_name>/
    This allows easy inspection of generated ONNX files.

    Set PYTEST_USE_TEMP=1 environment variable to use temporary directories instead.
    """
    # Check if we should use temp directory
    use_temp = os.environ.get("PYTEST_USE_TEMP", "0") == "1"

    if use_temp:
        # Use pytest's tmp_path for temporary storage
        import tempfile

        test_dir = Path(tempfile.mkdtemp(prefix="model_test_"))
    else:
        # Use fixed location for persistent storage
        project_root = Path(__file__).parent.parent.parent
        test_name = request.node.name
        test_dir = project_root / "test_outputs" / test_name
        test_dir.mkdir(parents=True, exist_ok=True)

    return str(test_dir)


@pytest.fixture
def cct_config():
    """Default configuration for CCT model."""
    return {
        "batch_size": 1,
        "img_size": 32,
        "embedding_dim": 128,
        "num_heads": 2,
        "num_layers": 2,
        "num_classes": 10,
        "opset_version": 12,
    }


@pytest.fixture
def epidenet_config():
    """Default configuration for EpiDeNet model."""
    return {
        "batch_size": 1,
        "channels": 8,
        "time_steps": 2000,
        "num_classes": 11,
        "opset_version": 12,
    }


@pytest.fixture
def mibminet_config():
    """Default configuration for MI-BMInet model."""
    return {
        "batch_size": 1,
        "channels": 8,
        "time_steps": 2000,
        "num_classes": 2,
        "F1": 8,
        "D": 2,
        "Nf": 64,
        "Nf2": 16,
        "opset_version": 12,
    }
