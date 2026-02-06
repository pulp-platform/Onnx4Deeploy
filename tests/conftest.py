# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Pytest configuration and shared fixtures for Onnx4Deeploy tests
"""

from pathlib import Path

import onnx
import onnxruntime as ort
import pytest


@pytest.fixture
def baseline_dir():
    """Directory containing baseline ONNX models for comparison"""
    return Path(__file__).parent / "baseline"


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs"""
    output_dir = tmp_path / "onnx_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def cct_config():
    """Configuration for CCT model"""
    return {
        "batch_size": 1,
        "img_size": 32,
        "embedding_dim": 128,
        "num_heads": 1,
        "num_layers": 2,
        "num_classes": 10,
        "opset_version": 12,
    }


@pytest.fixture
def epidenet_config():
    """Configuration for EpiDeNet model"""
    return {
        "batch_size": 1,
        "channels": 8,
        "time_steps": 2000,
        "num_classes": 11,
        "opset_version": 12,
    }


@pytest.fixture
def mibminet_config():
    """Configuration for MIBMINet model"""
    return {
        "batch_size": 1,
        "channels": 8,
        "time_steps": 2000,
        "num_classes": 2,
        "opset_version": 12,
    }


def compare_onnx_outputs(
    model1_path: Path, model2_path: Path, tolerance: float = 1e-5, num_samples: int = 5
) -> bool:
    """
    Compare outputs of two ONNX models with random inputs

    Args:
        model1_path: Path to first ONNX model
        model2_path: Path to second ONNX model
        tolerance: Numerical tolerance for comparison
        num_samples: Number of random input samples to test

    Returns:
        True if outputs match within tolerance, False otherwise
    """
    import numpy as np

    # Load models
    onnx.load(str(model1_path))
    onnx.load(str(model2_path))

    # Create inference sessions
    sess1 = ort.InferenceSession(str(model1_path))
    sess2 = ort.InferenceSession(str(model2_path))

    # Get input info
    input1 = sess1.get_inputs()[0]
    input2 = sess2.get_inputs()[0]

    # Check input shapes match
    if input1.shape != input2.shape:
        print(f"Input shape mismatch: {input1.shape} vs {input2.shape}")
        return False

    # Test with multiple random inputs
    for i in range(num_samples):
        # Generate random input
        input_shape = [dim if isinstance(dim, int) else 1 for dim in input1.shape]
        input_data = np.random.randn(*input_shape).astype(np.float32)

        # Run inference
        outputs1 = sess1.run(None, {input1.name: input_data})
        outputs2 = sess2.run(None, {input2.name: input_data})

        # Compare outputs
        for out1, out2 in zip(outputs1, outputs2):
            if not np.allclose(out1, out2, rtol=tolerance, atol=tolerance):
                max_diff = np.max(np.abs(out1 - out2))
                print(f"Output mismatch (sample {i+1}): max_diff={max_diff}, tolerance={tolerance}")
                return False

    return True


def compare_onnx_models(model1_path: Path, model2_path: Path) -> bool:
    """
    Compare two ONNX models structurally

    Args:
        model1_path: Path to first ONNX model
        model2_path: Path to second ONNX model

    Returns:
        True if models are structurally equivalent
    """
    model1 = onnx.load(str(model1_path))
    model2 = onnx.load(str(model2_path))

    # Compare number of nodes
    if len(model1.graph.node) != len(model2.graph.node):
        print(f"Node count mismatch: {len(model1.graph.node)} vs {len(model2.graph.node)}")
        return False

    # Compare op types
    ops1 = [node.op_type for node in model1.graph.node]
    ops2 = [node.op_type for node in model2.graph.node]

    if ops1 != ops2:
        print("Op type mismatch")
        return False

    return True
