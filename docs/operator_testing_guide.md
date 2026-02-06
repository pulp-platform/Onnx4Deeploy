# Operator Testing Guide

## Overview

Onnx4Deeploy provides a unified testing framework for ONNX operators through the `BaseOperatorTest` class. This guide explains how to create and run operator tests.

## Quick Start

### Running Tests

```bash
# Run all operator tests
pytest tests/operators/ -v

# Run specific test class
pytest tests/operators/test_operators.py::TestAddOperator -v

# Run with coverage
pytest tests/operators/ -v --cov=onnx4deeploy.operators

# Run all tests (operators + core + integration)
pytest tests/ -v
```

## Creating a New Operator Test

### Step 1: Create Operator Test Class

Create a new file in `onnx4deeploy/operators/` (e.g., `conv.py`):

```python
"""Conv2D operator test implementation."""

import numpy as np
from typing import Dict, Any
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class Conv2DOperatorTest(BaseOperatorTest):
    """Test generator for ONNX Conv operator."""

    def get_operator_name(self) -> str:
        return "Conv"

    def load_config(self) -> Dict[str, Any]:
        """Load Conv-specific configuration."""
        config = super().load_config()

        conv_config = config.get("conv", {})
        self.input_shape = tuple(conv_config["input_shape"])
        self.kernel_shape = tuple(conv_config["kernel_shape"])
        # ... load other params

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random input and kernel data."""
        inputs = {
            "input": np.random.randn(*self.input_shape).astype(np.float32),
            "kernel": np.random.randn(*self.kernel_shape).astype(np.float32)
        }
        return inputs

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for Conv operator."""
        # Create input tensors
        input_tensor = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, self.input_shape
        )
        kernel_tensor = helper.make_tensor_value_info(
            "kernel", TensorProto.FLOAT, self.kernel_shape
        )

        # Create output tensor
        output_shape = self._compute_output_shape()
        output_tensor = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, output_shape
        )

        # Create Conv node
        conv_node = helper.make_node(
            "Conv",
            inputs=["input", "kernel"],
            outputs=["output"],
            kernel_shape=self.kernel_shape[-2:],
            # ... other attributes
        )

        # Create graph
        graph = helper.make_graph(
            [conv_node],
            "conv_graph",
            [input_tensor, kernel_tensor],
            [output_tensor]
        )

        return graph

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Optional: Compute expected output for validation.
        Return None to skip validation (ONNX Runtime validates instead).
        """
        return None  # Or implement NumPy-based validation
```

### Step 2: Add to Package Exports

Update `onnx4deeploy/operators/__init__.py`:

```python
from .conv import Conv2DOperatorTest

__all__ = [
    # ... existing exports
    "Conv2DOperatorTest",
]
```

### Step 3: Create Pytest Tests

Add tests in `tests/operators/test_operators.py`:

```python
class TestConv2DOperator:
    """Tests for Conv2D operator."""

    def test_conv2d_basic(self, operator_test_dir):
        """Test basic Conv2D functionality."""
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write("""conv:
  input_shape: [1, 3, 32, 32]
  kernel_shape: [16, 3, 3, 3]
  stride: [1, 1]
  padding: [1, 1]
""")

        test = Conv2DOperatorTest(config_path=config_path, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        assert os.path.exists(onnx_file)
        assert os.path.exists(input_file)
        assert os.path.exists(output_file)
```

## Using SimpleElementwiseOperator

For simple operators with one or two inputs of the same shape, use `SimpleElementwiseOperator`:

```python
from .base_operator import SimpleElementwiseOperator


class SubOperatorTest(SimpleElementwiseOperator):
    """Test for Sub (subtraction) operator."""

    def get_operator_name(self) -> str:
        return "Sub"

    def get_config_key(self) -> str:
        return "subtract"

    def load_config(self):
        config = super().load_config()

        # Override for two inputs
        op_config = config.get(self.get_config_key(), {})
        if "input_shape" in op_config:
            shape = tuple(op_config["input_shape"])
            self.input_shapes = [shape, shape]
            self.input_names = ["input_a", "input_b"]

        return config

    def compute_expected_output(self, inputs):
        return {"output": inputs["input_a"] - inputs["input_b"]}
```

## Configuration File Format

Each operator test requires a `config.yaml` file:

```yaml
# For Add operator
adder:
  input_shape: [1, 64, 128]

# For Relu operator
relu:
  input_shape: [1, 64, 32, 32]

# For GEMM operator
gemm:
  input_a_shape: [8, 8]
  input_b_shape: [8, 8]
  transA: 0
  transB: 0
  alpha: 1.0
  beta: 1.0
  use_bias: true
```

## Test Output Structure

Generated tests create three files:

```
operator_test_dir/
├── network.onnx     # ONNX model
├── inputs.npz       # Input data (all inputs as numpy arrays)
└── outputs.npz      # Output data (all outputs as numpy arrays)
```

## Best Practices

1. **Use ONNX Runtime for Validation**: Let ONNX Runtime validate correctness during `generate()`. Only implement `compute_expected_output()` if you need additional validation.

2. **Keep Tests Focused**: Each test should verify one specific aspect of the operator.

3. **Use Fixtures**: Leverage pytest fixtures like `operator_test_dir` for temporary directories.

4. **Test Edge Cases**: Include tests for different shapes, data types, and attribute values.

5. **Mark Slow Tests**: Use `@pytest.mark.slow` for tests that take >1 second.

## Migrating Legacy Tests

To migrate a legacy test from `Tests/Operators/`:

1. Read the existing `testgenerate.py` to understand the operator
2. Create a new operator test class following the template above
3. Port the configuration to the new format
4. Add pytest tests
5. Verify with `pytest tests/operators/test_operators.py::TestYourOperator -v`

## CI Integration

Tests automatically run on GitHub Actions for:
- Python 3.8, 3.9, 3.10, 3.11
- On push to main, devel, or refactor/** branches
- On pull requests

View test results in the Actions tab of your repository.

## Troubleshooting

### Test Fails with "Invalid Graph"

Check that:
- Input/output shapes are correct
- Node inputs/outputs match tensor names
- Attribute values are valid for the operator

### ONNX Runtime Error

- Verify the operator is supported in your ONNX opset version
- Check that all required attributes are provided
- Ensure input data types match the operator requirements

### Import Errors

- Ensure you've installed the package: `pip install -e .`
- Check that the operator class is exported in `__init__.py`
- Verify Python path includes the project root
