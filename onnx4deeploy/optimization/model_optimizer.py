"""
ONNX model optimization utilities.

This module provides functions for optimizing ONNX models using ONNX Runtime tools
and custom optimization passes.
"""

import subprocess
from typing import Tuple


def run_onnx_optimization_infer(
    onnx_file: str, embedding_dim: int, num_heads: int, input_shape: Tuple[int, int, int, int]
) -> None:
    """
    Run ONNX Runtime optimization for inference models.

    Args:
        onnx_file: Path to the ONNX model file
        embedding_dim: Embedding dimension for the model
        num_heads: Number of attention heads
        input_shape: Input shape as (batch_size, channels, height, width)
    """
    batch_size, channels, height, width = input_shape  # Extract input dimensions
    try:
        print("🔹 Fixing dynamic shape...")
        subprocess.run(
            [
                "python",
                "-m",
                "onnxruntime.tools.make_dynamic_shape_fixed",
                "--input_name",
                "input",
                "--input_shape",
                f"{batch_size},{channels},{height},{width}",
                onnx_file,
                onnx_file,
            ],
            check=True,
        )

        print("🔹 Running symbolic shape inference...")
        subprocess.run(
            [
                "python",
                "-m",
                "onnxruntime.tools.symbolic_shape_infer",
                "--input",
                onnx_file,
                "--output",
                onnx_file,
                "--verbose",
                "3",
            ],
            check=True,
        )

        print("🔹 Optimizing ONNX model for ViT...")
        subprocess.run(
            [
                "python",
                "-m",
                "onnxruntime.transformers.optimizer",
                "--input",
                onnx_file,
                "--output",
                onnx_file,
                "--model_type",
                "vit",
                "--num_heads",
                str(num_heads),  # Controlled via config
                "--hidden_size",
                str(embedding_dim),  # Ensures hidden size = embedding_dim
                "--use_multi_head_attention",
                "--disable_bias_skip_layer_norm",
                "--disable_skip_layer_norm",
                "--disable_bias_gelu",
            ],
            check=True,
        )

        print("✅ ONNX model optimization complete!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during ONNX optimization: {e}")


def run_onnx_optimization(
    onnx_file: str, embedding_dim: int, num_heads: int, input_shape: Tuple[int, int, int, int]
) -> None:
    """
    Run ONNX Runtime tools to optimize the model.

    Args:
        onnx_file: Path to the ONNX model file
        embedding_dim: Embedding dimension for the model
        num_heads: Number of attention heads
        input_shape: Input shape as (batch_size, channels, height, width)
    """
    batch_size, channels, height, width = input_shape  # Extract input dimensions

    try:
        print("🔹 Fixing dynamic shape...")
        subprocess.run(
            [
                "python",
                "-m",
                "onnxruntime.tools.make_dynamic_shape_fixed",
                "--input_name",
                "input",
                "--input_shape",
                f"{batch_size},{channels},{height},{width}",
                onnx_file,
                onnx_file,
            ],
            check=True,
        )

        print("🔹 Running symbolic shape inference...")
        subprocess.run(
            [
                "python",
                "-m",
                "onnxruntime.tools.symbolic_shape_infer",
                "--input",
                onnx_file,
                "--output",
                onnx_file,
                "--verbose",
                "3",
            ],
            check=True,
        )

        print("🔹 Optimizing ONNX model for ViT...")
        subprocess.run(
            [
                "python",
                "-m",
                "onnxruntime.transformers.optimizer",
                "--input",
                onnx_file,
                "--output",
                onnx_file,
                "--model_type",
                "vit",
                "--num_heads",
                str(num_heads),  # Controlled via config
                "--hidden_size",
                str(embedding_dim),  # Ensures hidden size = embedding_dim
                "--use_multi_head_attention",
                "--disable_bias_skip_layer_norm",
                "--disable_skip_layer_norm",
                "--disable_bias_gelu",
                "--disable_layer_norm",  # compatible with opset 15
            ],
            check=True,
        )

        print("✅ ONNX model optimization complete!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during ONNX optimization: {e}")

    # fuse_matmul_add_to_gemm(onnx_file, onnx_file)
    # print(
    #     f"✅ Successfully fused MatMul and Add to Gemm nodes. Saved as {onnx_file}"
    # )
