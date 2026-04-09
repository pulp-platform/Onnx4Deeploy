# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Standard Optimization Passes for ONNX Models.

This module provides a registry of common optimization passes that can be
used in optimization pipelines.
"""

from __future__ import annotations

from .optimization_pipeline import OptimizationPass, PassConfig


class RenameNodesPass(OptimizationPass):
    """Rename nodes for C compatibility."""

    def __init__(self):
        super().__init__(name="rename_nodes", description="Rename nodes for C compatibility")

    def apply(self, onnx_file: str, output_file: str, config: PassConfig) -> bool:
        try:
            from ..utils.node_naming import rename_and_save_onnx

            rename_and_save_onnx(onnx_file, output_file)
            return True
        except Exception as e:
            print(f"    Error: {e}")
            return False


class RemoveIdentityPass(OptimizationPass):
    """Remove Identity nodes from the graph."""

    def __init__(self):
        super().__init__(name="remove_identity", description="Remove Identity nodes")

    def apply(self, onnx_file: str, output_file: str, config: PassConfig) -> bool:
        try:
            from ..optimization.graph_cleaner import remove_identity_nodes

            remove_identity_nodes(onnx_file, output_file)
            return True
        except Exception as e:
            print(f"    Error: {e}")
            return False


class ReshapeFusionPass(OptimizationPass):
    """Optimize and fuse Reshape operations."""

    def __init__(self):
        super().__init__(name="reshape_fusion", description="Optimize Reshape operations")

    def apply(self, onnx_file: str, output_file: str, config: PassConfig) -> bool:
        try:
            from ..optimization.shape_optimizer import optimize_reshape_fusion

            optimize_reshape_fusion(onnx_file, output_file)
            return True
        except Exception as e:
            print(f"    Error: {e}")
            return False


class UnifyGemmPass(OptimizationPass):
    """Unify GEMM input dimensions."""

    def __init__(self):
        super().__init__(name="unify_gemm", description="Unify GEMM input dimensions")

    def apply(self, onnx_file: str, output_file: str, config: PassConfig) -> bool:
        try:
            from ..optimization.gemm_converter import unify_gemm_input_dims

            unify_gemm_input_dims(onnx_file, output_file)
            return True
        except Exception as e:
            print(f"    Error: {e}")
            return False


class BiasGeluOptPass(OptimizationPass):
    """Optimize BiasGelu operations."""

    def __init__(self):
        super().__init__(name="biasgelu_opt", description="Optimize BiasGelu operations")

    def apply(self, onnx_file: str, output_file: str, config: PassConfig) -> bool:
        try:
            from ..optimization.graph_cleaner import run_optmization_remove_biasgelu

            run_optmization_remove_biasgelu(onnx_file, output_file)
            return True
        except Exception as e:
            print(f"    Error: {e}")
            return False


class SqueezeUnsqueezePass(OptimizationPass):
    """Convert Squeeze/Unsqueeze input to attributes."""

    def __init__(self):
        super().__init__(
            name="squeeze_unsqueeze", description="Convert Squeeze/Unsqueeze to attributes"
        )

    def apply(self, onnx_file: str, output_file: str, config: PassConfig) -> bool:
        try:
            from ..optimization.shape_optimizer import convert_squeeze_unsqueeze_input_to_attr

            convert_squeeze_unsqueeze_input_to_attr(onnx_file, output_file)
            return True
        except Exception as e:
            print(f"    Error: {e}")
            return False


class SumToAddPass(OptimizationPass):
    """Convert Sum operations to Add."""

    def __init__(self):
        super().__init__(name="sum_to_add", description="Convert Sum to Add operations")

    def apply(self, onnx_file: str, output_file: str, config: PassConfig) -> bool:
        try:
            from ..optimization.shape_optimizer import convert_sum_to_add

            convert_sum_to_add(onnx_file, output_file)
            return True
        except Exception as e:
            print(f"    Error: {e}")
            return False


class SoftmaxAxisOptPass(OptimizationPass):
    """Optimize Softmax axis attribute."""

    def __init__(self):
        super().__init__(name="softmax_axis", description="Optimize Softmax axis")

    def apply(self, onnx_file: str, output_file: str, config: PassConfig) -> bool:
        try:
            from ..optimization.graph_cleaner import optimize_softmax_axis

            optimize_softmax_axis(onnx_file, output_file)
            return True
        except Exception:
            # Softmax optimization often fails, don't print error
            return False


class ShapeInferencePass(OptimizationPass):
    """Run shape inference with custom op support."""

    def __init__(self):
        super().__init__(
            name="shape_inference", description="Run shape inference with custom op support"
        )

    def apply(self, onnx_file: str, output_file: str, config: PassConfig) -> bool:
        try:
            from ..optimization.shape_optimizer import infer_shapes_with_custom_ops

            infer_shapes_with_custom_ops(onnx_file, output_file)
            return True
        except Exception as e:
            print(f"    Error: {e}")
            return False


class ONNXRuntimeTransformerPass(OptimizationPass):
    """Run ONNX Runtime transformer optimization (for ViT/transformer models)."""

    def __init__(self):
        super().__init__(
            name="onnxruntime_transformer",
            description="ONNX Runtime transformer optimization (includes LayerNorm fusion)",
        )

    def apply(self, onnx_file: str, output_file: str, config: PassConfig) -> bool:
        try:
            from ..optimization.model_optimizer import run_onnx_optimization_infer

            # Require embedding_dim, num_heads, input_shape parameters
            if "embedding_dim" not in config.params:
                raise ValueError("embedding_dim parameter required")
            if "num_heads" not in config.params:
                raise ValueError("num_heads parameter required")
            if "input_shape" not in config.params:
                raise ValueError("input_shape parameter required")

            run_onnx_optimization_infer(
                onnx_file,
                config.params["embedding_dim"],
                config.params["num_heads"],
                config.params["input_shape"],
            )
            return True
        except Exception as e:
            print(f"    Warning: {e}")
            return False


class RandomizeInitializersPass(OptimizationPass):
    """Randomize zero initializers (for testing)."""

    def __init__(self):
        super().__init__(
            name="randomize_initializers", description="Randomize zero initializers (for testing)"
        )

    def apply(self, onnx_file: str, output_file: str, config: PassConfig) -> bool:
        try:
            import onnx

            from ..transform.model_transform import randomize_onnx_initializers

            model = onnx.load(onnx_file)
            seed = config.params.get("seed", 42)
            model = randomize_onnx_initializers(model, seed=seed)
            onnx.save(model, output_file)
            return True
        except Exception as e:
            print(f"    Error: {e}")
            return False


class TrainingOptimizationPass(OptimizationPass):
    """Run comprehensive training optimization pipeline."""

    def __init__(self):
        super().__init__(
            name="training_optimization", description="Comprehensive training optimization pipeline"
        )

    def apply(self, onnx_file: str, output_file: str, config: PassConfig) -> bool:
        try:
            from ..optimization.train_optimizer import run_train_onnx_optimization

            split_layernormgrad = config.params.get("split_layernormgrad", False)

            run_train_onnx_optimization(
                onnx_train_file=onnx_file,
                onnx_output_file=output_file,
                split_layernormgrad=split_layernormgrad,
            )
            return True
        except Exception as e:
            print(f"    Error: {e}")
            return False


# Registry of standard passes
STANDARD_PASSES = {
    "rename_nodes": RenameNodesPass,
    "remove_identity": RemoveIdentityPass,
    "reshape_fusion": ReshapeFusionPass,
    "unify_gemm": UnifyGemmPass,
    "biasgelu_opt": BiasGeluOptPass,
    "squeeze_unsqueeze": SqueezeUnsqueezePass,
    "sum_to_add": SumToAddPass,
    "softmax_axis": SoftmaxAxisOptPass,
    "shape_inference": ShapeInferencePass,
    "onnxruntime_transformer": ONNXRuntimeTransformerPass,
    "randomize_initializers": RandomizeInitializersPass,
    "training_optimization": TrainingOptimizationPass,
}


def create_inference_pipeline() -> "OptimizationPipeline":
    """
    Create default inference optimization pipeline.

    Returns:
        OptimizationPipeline with standard inference optimizations
    """
    from .optimization_pipeline import OptimizationPipeline

    pipeline = OptimizationPipeline(name="inference")
    pipeline.add_pass(RenameNodesPass())
    pipeline.add_pass(RemoveIdentityPass())
    pipeline.add_pass(ReshapeFusionPass())
    pipeline.add_pass(UnifyGemmPass())
    pipeline.add_pass(BiasGeluOptPass())
    pipeline.add_pass(SqueezeUnsqueezePass())
    pipeline.add_pass(SumToAddPass())
    pipeline.add_pass(SoftmaxAxisOptPass())

    return pipeline


def create_training_pipeline() -> "OptimizationPipeline":
    """
    Create default training optimization pipeline.

    Returns:
        OptimizationPipeline with standard training optimizations
    """
    from .optimization_pipeline import OptimizationPipeline

    pipeline = OptimizationPipeline(name="training")
    pipeline.add_pass(TrainingOptimizationPass())

    return pipeline


def create_transformer_inference_pipeline(
    embedding_dim: int, num_heads: int, input_shape: tuple, skip_ort_transformer: bool = False
) -> "OptimizationPipeline":
    """
    Create inference pipeline for transformer models (CCT, ViT, etc.).

    Includes ONNX Runtime transformer optimizations with LayerNorm fusion.

    Args:
        embedding_dim: Model embedding dimension
        num_heads: Number of attention heads
        input_shape: Input tensor shape
        skip_ort_transformer: If True, skip the ONNXRuntimeTransformerPass. Must be
            set to True for training-mode exports: the ORT inference optimizer fuses
            ops into com.microsoft custom ops that have no standard ONNX shape inference
            support, causing generate_artifacts to fail (infer_shapes_on_base cannot
            infer the SoftmaxCrossEntropyLoss output shape).

    Returns:
        OptimizationPipeline with transformer-specific optimizations
    """
    from .optimization_pipeline import OptimizationPipeline, PassConfig

    pipeline = OptimizationPipeline(name="transformer_inference")

    # Add randomize pass (for testing)
    pipeline.add_pass(RandomizeInitializersPass())

    # Add rename pass before ONNX Runtime optimization
    pipeline.add_pass(RenameNodesPass())

    # Add ONNX Runtime transformer optimization (includes LayerNorm fusion).
    # Skipped for training-mode exports (see skip_ort_transformer docstring above).
    if not skip_ort_transformer:
        pipeline.add_pass(
            ONNXRuntimeTransformerPass(),
            config=PassConfig(
                enabled=True,
                params={
                    "embedding_dim": embedding_dim,
                    "num_heads": num_heads,
                    "input_shape": input_shape,
                },
            ),
        )
        # Rename again after ONNX Runtime optimization
        pipeline.add_pass(RenameNodesPass())

    # Add standard inference optimizations.
    # NOTE: SqueezeUnsqueezePass is intentionally omitted here.
    # In opset 13, Squeeze/Unsqueeze 'axes' must be an input tensor, not an
    # attribute. Converting to attribute format here would fail onnx.checker
    # validation (which runs before generate_artifacts). The conversion is
    # applied later by run_train_onnx_optimization on the final training graph.
    pipeline.add_pass(RemoveIdentityPass())
    pipeline.add_pass(ReshapeFusionPass())
    pipeline.add_pass(UnifyGemmPass())
    pipeline.add_pass(BiasGeluOptPass())
    pipeline.add_pass(SumToAddPass())
    # SoftmaxAxisOptPass is skipped for training-mode exports: it inserts Reshape
    # nodes with dynamic batch dim (-1) that cause ORT generate_artifacts() shape
    # inference to fail ("inferred shape (-1) differs from existing shape (1)").
    if not skip_ort_transformer:
        pipeline.add_pass(SoftmaxAxisOptPass())

    return pipeline
