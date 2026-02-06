"""Pytest configuration for operator tests."""

from pathlib import Path

import pytest


@pytest.fixture
def operator_test_dir(request):
    """
    Create a persistent directory for operator test artifacts.

    Files are saved to: onnx/operator/{test_name}/

    This allows generated ONNX models, inputs, and outputs to be persisted
    in a centralized location at the project root.
    """
    # Get the test function name
    test_name = request.node.name

    # Create persistent directory structure
    # /app/Onnx4Deeploy/onnx/operator/test_add_basic/
    project_root = Path(__file__).parent.parent.parent
    onnx_base_dir = project_root / "onnx" / "operator"
    test_dir = onnx_base_dir / test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    return str(test_dir)


@pytest.fixture
def legacy_operators_dir():
    """Path to legacy operator tests directory."""
    return Path(__file__).parent.parent.parent / "Tests" / "Operators"
