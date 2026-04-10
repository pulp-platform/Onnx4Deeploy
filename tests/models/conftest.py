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


@pytest.fixture
def simplemlp_config():
    """Default configuration for Simple MLP model."""
    return {
        "batch_size": 1,
        "input_height": 8,
        "input_width": 8,
        "hidden_size": 8,
        "num_classes": 10,
        "opset_version": 17,
    }


@pytest.fixture
def resnet_config():
    """Minimal ResNet-8 configuration for fast tests (MLperf Tiny IC)."""
    return {
        "batch_size": 1,
        "variant": "resnet8",
        "img_size": 32,
        "input_channels": 3,
        "num_classes": 10,
        "base_channels": 16,
        "opset_version": 17,
    }


@pytest.fixture
def mobilenetv2_config():
    """Minimal MobileNetV2-0.35 configuration for fast tests (MLperf Tiny VWW)."""
    return {
        "batch_size": 1,
        "img_size": 32,  # Reduced from 96 for speed
        "input_channels": 3,
        "num_classes": 2,
        "width_mult": 0.35,
        "opset_version": 17,
    }


@pytest.fixture
def mobilenetv2_training_config():
    """MobileNetV2-0.35 training configuration (MLperf Tiny VWW).

    Uses img_size=64 because img_size=32 collapses the final feature map to
    1x1 which BatchNorm2d rejects in training mode.
    """
    return {
        "batch_size": 1,
        "img_size": 64,
        "input_channels": 3,
        "num_classes": 2,
        "width_mult": 0.35,
        "opset_version": 17,
    }


@pytest.fixture
def dscnn_config():
    """Default DS-CNN-XS configuration for fast tests (MLperf Tiny KWS)."""
    return {
        "batch_size": 1,
        "n_time": 25,
        "n_freq": 10,
        "num_classes": 12,
        "base_channels": 16,
        "n_ds_blocks": 2,  # Reduced from 4 for speed
        "opset_version": 17,
    }


@pytest.fixture
def lightweight_cnn_config():
    """Default Lightweight CNN configuration for fast tests."""
    return {
        "batch_size": 1,
        "input_height": 28,
        "input_width": 28,
        "input_channels": 1,
        "num_classes": 10,
        "opset_version": 17,
    }


@pytest.fixture
def autoencoder_config():
    """Minimal Autoencoder configuration for fast tests (MLperf Tiny AD)."""
    return {
        "batch_size": 1,
        "input_dim": 32,  # Reduced from 128 for speed
        "hidden_dims": [16, 8, 16],  # Reduced for speed
        "variant": "tiny",
        "opset_version": 17,
    }


@pytest.fixture
def simple_cnn_config():
    """Default Simple CNN configuration for fast tests."""
    return {
        "batch_size": 1,
        "input_channels": 1,
        "input_height": 16,
        "input_width": 16,
        "hidden_channels": 16,
        "num_classes": 10,
        "opset_version": 17,
    }


@pytest.fixture
def tiny_transformer_config():
    """Default Tiny Transformer configuration for fast tests."""
    return {
        "batch_size": 1,
        "img_size": 28,
        "patch_size": 7,
        "embed_dim": 32,
        "ffn_hidden": 64,
        "num_classes": 10,
        "opset_version": 17,
    }


@pytest.fixture
def sleep_convit_config():
    """Default SleepConViT configuration (tests use library defaults).

    SleepConViT's `num_patches`/`seq_len` are derived from ConvStem strides,
    so we cannot freely shrink `input_length` without also updating those.
    """
    return {
        "batch_size": 1,
        "input_channels": 1,
        "input_length": 3000,
        "model_dim": 48,
        "num_heads": 6,
        "num_classes": 5,
        "opset_version": 17,
    }


@pytest.fixture
def tinyvit_config():
    """Minimal TinyViT configuration for fast tests."""
    return {
        "batch_size": 1,
        "img_size": 32,
        "input_channels": 3,
        "num_classes": 5,
        "variant": "tiny_vit_5m",
        "opset_version": 17,
    }


@pytest.fixture
def mobilevit_config():
    """Minimal MobileViT configuration for fast tests (XXS variant)."""
    return {
        "batch_size": 1,
        "img_size": 64,
        "input_channels": 3,
        "num_classes": 5,
        "variant": "mobile_vit_xxs",
        "opset_version": 17,
    }


@pytest.fixture
def mamba_config():
    """Minimal Mamba configuration for fast tests (inference only)."""
    return {
        "batch_size": 1,
        "d_model": 32,
        "n_layers": 2,
        "d_state": 8,
        "d_conv": 4,
        "expand_factor": 2,
        "max_seq_len": 64,
        "num_classes": 5,
        "opset_version": 17,
    }
