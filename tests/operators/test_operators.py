"""
Unified operator tests using pytest framework.

This module tests all ONNX operators using the new BaseOperatorTest framework.
"""

import os

import pytest

from onnx4deeploy.operators import (
    AddOperatorTest,
    AveragePoolGradOperatorTest,
    AveragePoolOperatorTest,
    Conv2DOperatorTest,
    GemmOperatorTest,
    GroupNormOperatorTest,
    LayerNormOperatorTest,
    MatMulOperatorTest,
    MaxPoolOperatorTest,
    ReduceSumOperatorTest,
    ReluGradOperatorTest,
    ReluOperatorTest,
    SGDOperatorTest,
    SplitOperatorTest,
    TransposeOperatorTest,
)

# isort: off
# Import numpy AFTER onnx4deeploy to prevent numpy reload issues with PyTorch
import numpy as np  # noqa: E402

# isort: on


class TestAddOperator:
    """Tests for Add operator."""

    def test_add_basic(self, operator_test_dir):
        """Test basic Add operator functionality."""
        # Create config
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write("adder:\n  input_shape: [1, 64, 128]\n")

        # Generate operator test
        test = AddOperatorTest(config_path=config_path, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        # Verify files were created
        assert os.path.exists(onnx_file)
        assert os.path.exists(input_file)
        assert os.path.exists(output_file)

        # Verify data
        inputs = np.load(input_file)
        outputs = np.load(output_file)

        assert "input_a" in inputs
        assert "input_b" in inputs
        assert "output" in outputs

        # Verify output correctness
        expected = inputs["input_a"] + inputs["input_b"]
        np.testing.assert_allclose(outputs["output"], expected, rtol=1e-5, atol=1e-5)

    def test_add_different_shapes(self, operator_test_dir):
        """Test Add operator with different input shapes."""
        shapes = [[1, 32, 64], [8, 16, 32], [1, 128, 256]]

        for shape in shapes:
            config_path = os.path.join(operator_test_dir, f"config_{shape[1]}.yaml")
            with open(config_path, "w") as f:
                f.write(f"adder:\n  input_shape: {shape}\n")

            test = AddOperatorTest(config_path=config_path, save_path=operator_test_dir)
            test.generate()


class TestReluOperator:
    """Tests for Relu operator."""

    def test_relu_basic(self, operator_test_dir):
        """Test basic Relu operator functionality."""
        # Create config
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write("relu:\n  input_shape: [1, 64, 32, 32]\n")

        # Generate operator test
        test = ReluOperatorTest(config_path=config_path, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        # Verify files were created
        assert os.path.exists(onnx_file)
        assert os.path.exists(input_file)
        assert os.path.exists(output_file)

        # Verify data
        inputs = np.load(input_file)
        outputs = np.load(output_file)

        assert "input" in inputs
        assert "output" in outputs

        # Verify output correctness
        expected = np.maximum(0, inputs["input"])
        np.testing.assert_allclose(outputs["output"], expected, rtol=1e-5, atol=1e-5)

    def test_relu_negative_values(self, operator_test_dir):
        """Test that Relu correctly zeros out negative values."""
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write("relu:\n  input_shape: [1, 10, 10]\n")

        test = ReluOperatorTest(config_path=config_path, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        outputs = np.load(output_file)

        # Verify no negative values in output
        assert np.all(outputs["output"] >= 0)


class TestGemmOperator:
    """Tests for Gemm operator."""

    def test_gemm_basic(self, operator_test_dir):
        """Test basic Gemm operator functionality."""
        # Create config
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(
                """gemm:
  input_a_shape: [8, 8]
  input_b_shape: [8, 8]
  transA: 0
  transB: 0
  alpha: 1.0
  beta: 1.0
  a_is_const: false
  b_is_const: false
  use_bias: true
"""
            )

        # Generate operator test
        test = GemmOperatorTest(config_path=config_path, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        # Verify files were created
        assert os.path.exists(onnx_file)
        assert os.path.exists(input_file)
        assert os.path.exists(output_file)

        # Verify data
        inputs = np.load(input_file)
        outputs = np.load(output_file)

        assert "input_a" in inputs
        assert "input_b" in inputs
        assert "input_c" in inputs
        assert "output" in outputs

        # Verify output correctness
        expected = np.matmul(inputs["input_a"], inputs["input_b"]) + inputs["input_c"]
        np.testing.assert_allclose(outputs["output"], expected, rtol=1e-4, atol=1e-4)

    def test_gemm_transpose_a(self, operator_test_dir):
        """Test Gemm with transA=1."""
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            # A: [8, 4], transA=1 -> [4, 8]
            # B: [4, 6] (rows must match A's columns after transpose)
            # Result: [8, 6]
            f.write(
                """gemm:
  input_a_shape: [4, 8]
  input_b_shape: [4, 6]
  transA: 1
  transB: 0
  alpha: 1.0
  beta: 1.0
  use_bias: false
"""
            )

        test = GemmOperatorTest(config_path=config_path, save_path=operator_test_dir)

        # Just verify generation works, ONNX Runtime validates correctness
        onnx_file, input_file, output_file = test.generate()

        assert os.path.exists(onnx_file)
        assert os.path.exists(input_file)
        assert os.path.exists(output_file)

    def test_gemm_transpose_b(self, operator_test_dir):
        """Test Gemm with transB=1."""
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            # A: [4, 8]
            # B: [6, 8], transB=1 -> [8, 6]
            # Result: [4, 6]
            f.write(
                """gemm:
  input_a_shape: [4, 8]
  input_b_shape: [6, 8]
  transA: 0
  transB: 1
  alpha: 1.0
  beta: 1.0
  use_bias: false
"""
            )

        test = GemmOperatorTest(config_path=config_path, save_path=operator_test_dir)

        # Just verify generation works, ONNX Runtime validates correctness
        onnx_file, input_file, output_file = test.generate()

        assert os.path.exists(onnx_file)
        assert os.path.exists(input_file)
        assert os.path.exists(output_file)

    def test_gemm_no_bias(self, operator_test_dir):
        """Test Gemm without bias."""
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(
                """gemm:
  input_a_shape: [8, 8]
  input_b_shape: [8, 8]
  transA: 0
  transB: 0
  alpha: 1.0
  beta: 0.0
  use_bias: false
"""
            )

        test = GemmOperatorTest(config_path=config_path, save_path=operator_test_dir)
        test.generate()


@pytest.mark.slow
@pytest.mark.integration
class TestLegacyOperators:
    """
    Integration tests for legacy operators.

    These tests verify that all legacy operator test scripts still work correctly.
    """

    def test_legacy_add(self, operator_test_dir, legacy_operators_dir):
        """Test legacy Add operator."""
        from onnx4deeploy.operators import AddOperatorTest

        config_file = legacy_operators_dir / "testFloatAdd" / "config.yaml"
        if not config_file.exists():
            pytest.skip(f"Legacy config not found: {config_file}")

        test = AddOperatorTest(config_path=str(config_file), save_path=operator_test_dir)
        test.generate()

    def test_legacy_relu(self, operator_test_dir, legacy_operators_dir):
        """Test legacy Relu operator."""
        from onnx4deeploy.operators import ReluOperatorTest

        config_file = legacy_operators_dir / "testFloatRelu" / "config.yaml"
        if not config_file.exists():
            pytest.skip(f"Legacy config not found: {config_file}")

        test = ReluOperatorTest(config_path=str(config_file), save_path=operator_test_dir)
        test.generate()

    def test_legacy_gemm(self, operator_test_dir, legacy_operators_dir):
        """Test legacy GEMM operator."""
        from onnx4deeploy.operators import GemmOperatorTest

        config_file = legacy_operators_dir / "testFloatGEMM" / "config.yaml"
        if not config_file.exists():
            pytest.skip(f"Legacy config not found: {config_file}")

        test = GemmOperatorTest(config_path=str(config_file), save_path=operator_test_dir)
        test.generate()


class TestMatMulOperator:
    """Tests for MatMul operator."""

    def test_matmul_basic(self, operator_test_dir):
        """Test basic MatMul functionality."""
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(
                """matmul:
  input_a_shape: [4, 8]
  input_b_shape: [8, 6]
"""
            )

        test = MatMulOperatorTest(config_path=config_path, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        assert os.path.exists(onnx_file)
        assert os.path.exists(input_file)
        assert os.path.exists(output_file)

        # Verify correctness
        inputs = np.load(input_file)
        outputs = np.load(output_file)
        expected = np.matmul(inputs["input_a"], inputs["input_b"])
        np.testing.assert_allclose(outputs["output"], expected, rtol=1e-4, atol=1e-4)

    def test_matmul_batch(self, operator_test_dir):
        """Test MatMul with batch dimension."""
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(
                """matmul:
  input_a_shape: [2, 4, 8]
  input_b_shape: [2, 8, 6]
"""
            )

        test = MatMulOperatorTest(config_path=config_path, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        # Verify correctness
        inputs = np.load(input_file)
        outputs = np.load(output_file)
        expected = np.matmul(inputs["input_a"], inputs["input_b"])
        np.testing.assert_allclose(outputs["output"], expected, rtol=1e-4, atol=1e-4)


class TestReduceSumOperator:
    """Tests for ReduceSum operator."""

    def test_reducesum_single_axis(self, operator_test_dir):
        """Test ReduceSum with single axis."""
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(
                """reducesum:
  input_shape: [3, 4, 5]
  axes: [1]
  keepdims: 1
"""
            )

        test = ReduceSumOperatorTest(config_path=config_path, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        assert os.path.exists(onnx_file)
        assert os.path.exists(input_file)
        assert os.path.exists(output_file)

        # Verify correctness
        inputs = np.load(input_file)
        outputs = np.load(output_file)
        expected = np.sum(inputs["input"], axis=1, keepdims=True)
        np.testing.assert_allclose(outputs["output"], expected, rtol=1e-4, atol=1e-4)

    def test_reducesum_multiple_axes(self, operator_test_dir):
        """Test ReduceSum with multiple axes."""
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(
                """reducesum:
  input_shape: [2, 3, 4, 5]
  axes: [1, 3]
  keepdims: 0
"""
            )

        test = ReduceSumOperatorTest(config_path=config_path, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        # Verify correctness
        inputs = np.load(input_file)
        outputs = np.load(output_file)
        expected = np.sum(inputs["input"], axis=(1, 3), keepdims=False)
        np.testing.assert_allclose(outputs["output"], expected, rtol=1e-4, atol=1e-4)


class TestSplitOperator:
    """Tests for Split operator."""

    def test_split_basic(self, operator_test_dir):
        """Test basic Split functionality."""
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(
                """split:
  input_shape: [1, 64, 128]
  axis: 1
  num_outputs: 2
"""
            )

        test = SplitOperatorTest(config_path=config_path, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        assert os.path.exists(onnx_file)
        assert os.path.exists(input_file)
        assert os.path.exists(output_file)

        # Verify correctness
        inputs = np.load(input_file)
        outputs = np.load(output_file)

        assert "output_0" in outputs
        assert "output_1" in outputs

        # Verify splits
        expected_splits = np.split(inputs["input"], 2, axis=1)
        np.testing.assert_allclose(outputs["output_0"], expected_splits[0], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(outputs["output_1"], expected_splits[1], rtol=1e-4, atol=1e-4)

    def test_split_four_outputs(self, operator_test_dir):
        """Test Split with four outputs."""
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(
                """split:
  input_shape: [2, 8, 16]
  axis: 2
  num_outputs: 4
"""
            )

        test = SplitOperatorTest(config_path=config_path, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        outputs = np.load(output_file)
        assert len(outputs) == 4


class TestTransposeOperator:
    """Tests for Transpose operator."""

    def test_transpose_basic(self, operator_test_dir):
        """Test basic Transpose functionality."""
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(
                """transpose:
  input_shape: [1, 16, 16, 32]
  perm: [0, 3, 1, 2]
"""
            )

        test = TransposeOperatorTest(config_path=config_path, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        assert os.path.exists(onnx_file)
        assert os.path.exists(input_file)
        assert os.path.exists(output_file)

        # Verify correctness
        inputs = np.load(input_file)
        outputs = np.load(output_file)
        expected = np.transpose(inputs["input"], [0, 3, 1, 2])
        np.testing.assert_allclose(outputs["output"], expected, rtol=1e-4, atol=1e-4)

    def test_transpose_matrix(self, operator_test_dir):
        """Test Transpose on 2D matrix."""
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(
                """transpose:
  input_shape: [4, 8]
  perm: [1, 0]
"""
            )

        test = TransposeOperatorTest(config_path=config_path, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        # Verify correctness
        inputs = np.load(input_file)
        outputs = np.load(output_file)
        expected = np.transpose(inputs["input"], [1, 0])
        np.testing.assert_allclose(outputs["output"], expected, rtol=1e-4, atol=1e-4)


class TestMaxPoolOperator:
    """Tests for MaxPool operator."""

    def test_maxpool_basic(self, operator_test_dir):
        """Test basic MaxPool functionality."""
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(
                """maxpool:
  input_shape: [1, 16, 32, 32]
  kernel_shape: [2, 2]
  strides: [2, 2]
  pads: [0, 0, 0, 0]
  ceil_mode: 0
"""
            )

        test = MaxPoolOperatorTest(config_path=config_path, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        assert os.path.exists(onnx_file)
        assert os.path.exists(input_file)
        assert os.path.exists(output_file)

        # Verify output shape
        outputs = np.load(output_file)
        assert outputs["output"].shape == (1, 16, 16, 16)

    def test_maxpool_with_padding(self, operator_test_dir):
        """Test MaxPool with padding."""
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(
                """maxpool:
  input_shape: [1, 8, 16, 16]
  kernel_shape: [3, 3]
  strides: [1, 1]
  pads: [1, 1, 1, 1]
  ceil_mode: 0
"""
            )

        test = MaxPoolOperatorTest(config_path=config_path, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        outputs = np.load(output_file)
        assert outputs["output"].shape == (1, 8, 16, 16)


class TestAveragePoolOperator:
    """Tests for AveragePool operator."""

    def test_averagepool_basic(self, operator_test_dir):
        """Test basic AveragePool functionality."""
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(
                """averagepool:
  input_shape: [1, 8, 16, 16]
  kernel_shape: [2, 2]
  strides: [2, 2]
  pads: [0, 0, 0, 0]
  count_include_pad: 0
  ceil_mode: 0
"""
            )

        test = AveragePoolOperatorTest(config_path=config_path, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        assert os.path.exists(onnx_file)
        assert os.path.exists(input_file)
        assert os.path.exists(output_file)

        # Verify output shape
        outputs = np.load(output_file)
        assert outputs["output"].shape == (1, 8, 8, 8)

    def test_averagepool_with_padding(self, operator_test_dir):
        """Test AveragePool with padding and count_include_pad."""
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(
                """averagepool:
  input_shape: [1, 16, 32, 32]
  kernel_shape: [2, 2]
  strides: [2, 2]
  pads: [1, 1, 1, 1]
  count_include_pad: 1
  ceil_mode: 0
"""
            )

        test = AveragePoolOperatorTest(config_path=config_path, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        outputs = np.load(output_file)
        # Output shape with padding (32 + 1 + 1 - 2) / 2 + 1 = 17
        assert outputs["output"].shape == (1, 16, 17, 17)


class TestAveragePoolGradOperator:
    """Tests for AveragePoolGrad operator."""

    def test_averagepoolgrad_basic(self, operator_test_dir):
        """Test basic AveragePoolGrad functionality."""
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(
                """averagepoolgrad:
  input_grad_shape: [1, 16, 16, 16]
  original_input_shape: [1, 16, 32, 32]
  kernel_shape: [2, 2]
  strides: [2, 2]
  pads: [0, 0, 0, 0]
  count_include_pad: 0
  ceil_mode: 0
"""
            )

        test = AveragePoolGradOperatorTest(config_path=config_path, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        assert os.path.exists(onnx_file)
        assert os.path.exists(input_file)
        assert os.path.exists(output_file)

        # Verify output shape matches original input shape
        outputs = np.load(output_file)
        assert outputs["output_grad"].shape == (1, 16, 32, 32)


class TestReluGradOperator:
    """Tests for ReluGrad operator."""

    def test_relugrad_basic(self, operator_test_dir):
        """Test basic ReluGrad functionality."""
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(
                """relugrad:
  input_shape: [1, 64, 32, 32]
"""
            )

        test = ReluGradOperatorTest(config_path=config_path, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        assert os.path.exists(onnx_file)
        assert os.path.exists(input_file)
        assert os.path.exists(output_file)

        # Verify correctness
        inputs = np.load(input_file)
        outputs = np.load(output_file)
        expected = np.where(inputs["original_input"] > 0, inputs["grad_output"], 0)
        np.testing.assert_allclose(outputs["grad_input"], expected, rtol=1e-5, atol=1e-5)


class TestSGDOperator:
    """Tests for SGD operator."""

    def test_sgd_basic(self, operator_test_dir):
        """Test basic SGD functionality."""
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(
                """sgd:
  weight_shape: [10, 8]
  learning_rate: 0.01
"""
            )

        test = SGDOperatorTest(config_path=config_path, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        assert os.path.exists(onnx_file)
        assert os.path.exists(input_file)
        assert os.path.exists(output_file)

        # Verify correctness
        inputs = np.load(input_file)
        outputs = np.load(output_file)
        expected = inputs["weights"] - 0.01 * inputs["gradient"]
        np.testing.assert_allclose(outputs["updated_weights"], expected, rtol=1e-5, atol=1e-5)


class TestLayerNormOperator:
    """Tests for LayerNorm operator."""

    def test_layernorm_basic(self, operator_test_dir):
        """Test basic LayerNorm functionality."""
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(
                """layernorm:
  input_shape: [2, 3, 4]
  axis: -1
  epsilon: 0.00001
"""
            )

        test = LayerNormOperatorTest(config_path=config_path, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        assert os.path.exists(onnx_file)
        assert os.path.exists(input_file)
        assert os.path.exists(output_file)


class TestGroupNormOperator:
    """Tests for GroupNorm operator."""

    def test_groupnorm_basic(self, operator_test_dir):
        """Test basic GroupNorm functionality."""
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(
                """groupnorm:
  input_shape: [1, 8, 1, 64]
  num_groups: 1
  epsilon: 0.001
"""
            )

        test = GroupNormOperatorTest(config_path=config_path, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        assert os.path.exists(onnx_file)
        assert os.path.exists(input_file)
        assert os.path.exists(output_file)

        # Verify output shape
        outputs = np.load(output_file)
        assert outputs["Y"].shape == (1, 8, 1, 64)
        assert outputs["stat"].shape == (1, 1, 2)


class TestConv2DOperator:
    """Tests for Conv2D operator."""

    def test_conv2d_basic(self, operator_test_dir):
        """Test basic Conv2D functionality."""
        config_path = os.path.join(operator_test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(
                """conv2d:
  input_shape: [1, 3, 8, 8]
  kernel_size: 3
  stride: 1
  padding: 1
  out_channels: 16
  use_bias: true
  group: 1
  dilation: 1
"""
            )

        test = Conv2DOperatorTest(config_path=config_path, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        assert os.path.exists(onnx_file)
        assert os.path.exists(input_file)
        assert os.path.exists(output_file)

        # Verify output shape
        outputs = np.load(output_file)
        assert outputs["output"].shape == (1, 16, 8, 8)
