"""Shared utility functions for model tests.

This module provides common validation and testing utilities to reduce
code duplication across model test files.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort

# ============================================================================
# File Verification Utilities
# ============================================================================


def verifyonnx_file_exists(onnx_file: str, expected_suffix: str = "network.onnx") -> None:
    """Verify ONNX file exists with expected suffix.

    Args:
        onnx_file: Path to ONNX file
        expected_suffix: Expected file name suffix

    Raises:
        AssertionError: If file doesn't exist or has wrong suffix
    """
    assert os.path.exists(onnx_file), f"ONNX file not found: {onnx_file}"
    assert onnx_file.endswith(
        expected_suffix
    ), f"Expected suffix {expected_suffix}, got {onnx_file}"


def verify_training_artifacts(output_dir: str, required_files: Optional[List[str]] = None) -> None:
    """Verify training artifacts exist in output directory.

    Args:
        output_dir: Directory containing training artifacts
        required_files: List of required file names (default: checkpoint, optimizer_model.onnx, eval_model.onnx)

    Raises:
        AssertionError: If required files are missing
    """
    if required_files is None:
        required_files = ["checkpoint", "optimizer_model.onnx", "eval_model.onnx"]

    for filename in required_files:
        filepath = os.path.join(output_dir, filename)
        assert os.path.exists(filepath), f"Missing required file: {filepath}"


# ============================================================================
# ONNX Model Verification Utilities
# ============================================================================


def load_and_check_onnx_model(onnx_file: str, skip_shape_check: bool = False) -> onnx.ModelProto:
    """Load ONNX model and verify it's valid.

    Args:
        onnx_file: Path to ONNX file
        skip_shape_check: If True, skip strict shape validation (useful for training models)

    Returns:
        Loaded ONNX model

    Raises:
        AssertionError: If model is invalid
    """
    model = onnx.load(onnx_file)

    if skip_shape_check:
        # For training models, only check basic structure
        assert len(model.graph.node) > 0, "Model has no nodes"
        assert len(model.graph.input) >= 1, "Model has no inputs"
        assert len(model.graph.output) >= 1, "Model has no outputs"
    else:
        # Full validation for inference models
        onnx.checker.check_model(model)

    return model


def verify_model_structure(
    model: onnx.ModelProto,
    min_inputs: int = 1,
    min_outputs: int = 1,
    expected_input_name: Optional[str] = None,
    expected_output_name: Optional[str] = None,
) -> None:
    """Verify basic ONNX model structure.

    Args:
        model: ONNX model to verify
        min_inputs: Minimum number of inputs expected
        min_outputs: Minimum number of outputs expected
        expected_input_name: Expected name of first input (optional)
        expected_output_name: Expected name of first output (optional)

    Raises:
        AssertionError: If structure doesn't match expectations
    """
    assert (
        len(model.graph.input) >= min_inputs
    ), f"Expected at least {min_inputs} inputs, got {len(model.graph.input)}"
    assert (
        len(model.graph.output) >= min_outputs
    ), f"Expected at least {min_outputs} outputs, got {len(model.graph.output)}"

    if expected_input_name:
        actual_input_name = model.graph.input[0].name
        assert (
            actual_input_name == expected_input_name
        ), f"Expected input name '{expected_input_name}', got '{actual_input_name}'"

    if expected_output_name:
        actual_output_name = model.graph.output[0].name
        assert (
            actual_output_name == expected_output_name
        ), f"Expected output name '{expected_output_name}', got '{actual_output_name}'"


# ============================================================================
# Shape Verification Utilities
# ============================================================================


def get_tensor_shape(tensor_proto) -> List[int]:
    """Extract shape from ONNX tensor proto.

    Args:
        tensor_proto: ONNX tensor value info

    Returns:
        List of dimension values
    """
    return [dim.dim_value for dim in tensor_proto.type.tensor_type.shape.dim]


def verify_input_shape(
    model: onnx.ModelProto, expected_shape: List[int], input_index: int = 0
) -> None:
    """Verify input tensor shape.

    Args:
        model: ONNX model
        expected_shape: Expected shape as list of integers
        input_index: Index of input to check (default: 0)

    Raises:
        AssertionError: If shape doesn't match
    """
    actual_shape = get_tensor_shape(model.graph.input[input_index])
    assert (
        actual_shape == expected_shape
    ), f"Input shape mismatch: expected {expected_shape}, got {actual_shape}"


def verify_output_shape(
    model: onnx.ModelProto,
    expected_shape: Optional[List[int]] = None,
    expected_batch_size: Optional[int] = None,
    expected_last_dim: Optional[int] = None,
    output_index: int = 0,
) -> None:
    """Verify output tensor shape.

    Args:
        model: ONNX model
        expected_shape: Expected full shape (optional)
        expected_batch_size: Expected batch dimension (optional)
        expected_last_dim: Expected last dimension (optional)
        output_index: Index of output to check (default: 0)

    Raises:
        AssertionError: If shape doesn't match expectations
    """
    actual_shape = get_tensor_shape(model.graph.output[output_index])

    if expected_shape is not None:
        assert (
            actual_shape == expected_shape
        ), f"Output shape mismatch: expected {expected_shape}, got {actual_shape}"

    if expected_batch_size is not None:
        assert (
            actual_shape[0] == expected_batch_size
        ), f"Batch size mismatch: expected {expected_batch_size}, got {actual_shape[0]}"

    if expected_last_dim is not None:
        assert (
            actual_shape[-1] == expected_last_dim
        ), f"Last dimension mismatch: expected {expected_last_dim}, got {actual_shape[-1]}"


# ============================================================================
# ONNX Runtime Inference Utilities
# ============================================================================


def run_onnxruntime_inference(
    onnx_file: str, test_input: np.ndarray, input_name: str = "input"
) -> np.ndarray:
    """Run inference using ONNX Runtime.

    Args:
        onnx_file: Path to ONNX model
        test_input: Input data as numpy array
        input_name: Name of input tensor (default: "input")

    Returns:
        Output array from inference
    """
    session = ort.InferenceSession(onnx_file)
    outputs = session.run(None, {input_name: test_input})
    return outputs[0]


def verify_onnxruntime_compatibility(
    onnx_file: str,
    test_input: np.ndarray,
    expected_output_shape: Tuple[int, ...],
    input_name: str = "input",
) -> None:
    """Verify model can run with ONNX Runtime and produces correct output shape.

    Args:
        onnx_file: Path to ONNX model
        test_input: Input data as numpy array
        expected_output_shape: Expected output shape
        input_name: Name of input tensor (default: "input")

    Raises:
        AssertionError: If inference fails or output shape is wrong
    """
    output = run_onnxruntime_inference(onnx_file, test_input, input_name)
    assert (
        output.shape == expected_output_shape
    ), f"Output shape mismatch: expected {expected_output_shape}, got {output.shape}"


# ============================================================================
# Model Export Testing Utilities
# ============================================================================


def verify_inference_export(
    exporter,
    model_test_dir: str,
    expected_input_shape: List[int],
    expected_output_shape: Optional[List[int]] = None,
    expected_batch_size: Optional[int] = None,
    expected_output_classes: Optional[int] = None,
) -> str:
    """Complete verification of inference model export.

    This is a convenience function that performs all common checks for inference exports.

    Args:
        exporter: Model exporter instance
        model_test_dir: Directory for test outputs
        expected_input_shape: Expected input shape
        expected_output_shape: Expected full output shape (optional)
        expected_batch_size: Expected batch size (optional)
        expected_output_classes: Expected number of output classes (optional)

    Returns:
        Path to exported ONNX file

    Raises:
        AssertionError: If any verification fails
    """
    # Export model
    onnx_file = exporter.export(mode="infer")

    # Verify file exists
    verifyonnx_file_exists(onnx_file)

    # Load and check model
    model = load_and_check_onnx_model(onnx_file)

    # Verify structure
    verify_model_structure(model, expected_input_name="input", expected_output_name="output")

    # Verify shapes
    verify_input_shape(model, expected_input_shape)

    if expected_output_shape:
        verify_output_shape(model, expected_shape=expected_output_shape)
    else:
        verify_output_shape(
            model,
            expected_batch_size=expected_batch_size,
            expected_last_dim=expected_output_classes,
        )

    return onnx_file


def verify_training_export(
    exporter, model_test_dir: str, required_artifacts: Optional[List[str]] = None
) -> str:
    """Complete verification of training model export.

    This is a convenience function that performs all common checks for training exports.

    Args:
        exporter: Model exporter instance
        model_test_dir: Directory for test outputs
        required_artifacts: List of required artifact files (optional)

    Returns:
        Path to exported ONNX file

    Raises:
        AssertionError: If any verification fails
    """
    # Export model
    onnx_file = exporter.export(mode="train")

    # Verify file exists
    verifyonnx_file_exists(onnx_file)

    # Verify training artifacts
    output_dir = os.path.dirname(onnx_file)
    verify_training_artifacts(output_dir, required_artifacts)

    # Load and check model (skip shape validation for training models)
    model = load_and_check_onnx_model(onnx_file, skip_shape_check=True)

    return onnx_file


# ============================================================================
# Test Input Generation Utilities
# ============================================================================


def create_random_input(shape: Tuple[int, ...], seed: int = 42) -> np.ndarray:
    """Create random input tensor for testing.

    Args:
        shape: Shape of input tensor
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Random numpy array with given shape
    """
    np.random.seed(seed)
    return np.random.randn(*shape).astype(np.float32)


# ============================================================================
# Trainable Parameters Verification
# ============================================================================


def verify_trainable_params(exporter, model) -> None:
    """Verify trainable parameters are correctly identified.

    Args:
        exporter: Model exporter with get_trainable_params method
        model: PyTorch model

    Raises:
        AssertionError: If trainable params validation fails
    """
    all_params = [name for name, _ in model.named_parameters()]
    trainable = exporter.get_trainable_params(all_params)

    assert len(trainable) > 0, "No trainable parameters found"
    assert all(
        param in all_params for param in trainable
    ), "Some trainable params not in model parameters"
