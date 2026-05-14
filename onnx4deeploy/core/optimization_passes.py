# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Standard Optimization Passes for ONNX Models.

This module provides a registry of common optimization passes that can be
used in optimization pipelines.
"""

from __future__ import annotations

from typing import Optional

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


# ---------------------------------------------------------------------- #
# Quantization-export optimization passes                                  #
# ---------------------------------------------------------------------- #
#
# These wrap the in-place onnx.GraphProto helpers from
# `onnx4deeploy.optimization.qcdq_to_deeploy` so they can sit in the same
# `OptimizationPipeline` machinery as the inference/training passes.
#
# Pipeline factory: `create_quant_pipeline()` further below.
#
# Each pass:
#   * loads the ONNX from disk
#   * runs one transformation in-place
#   * saves to the output path
#   * reports the number of nodes affected via the pass description prefix


def _wrap_qcdq_pass(name: str, description: str, fn):
    """Build an `OptimizationPass` from a function taking an `onnx.GraphProto`
    (or a `(model, inputs_npz_path)` tuple for the input-quant pass).
    """
    import onnx as _onnx

    class _QcdqPass(OptimizationPass):
        def __init__(self):
            super().__init__(name=name, description=description)

        def apply(self, onnx_file: str, output_file: str, config: PassConfig) -> bool:
            try:
                m = _onnx.load(onnx_file)
                fn(m.graph)
                _onnx.save(m, output_file)
                return True
            except Exception as e:
                print(f"    Error: {e}")
                return False

    _QcdqPass.__name__ = f"Quant{name.title().replace('_','')}Pass"
    return _QcdqPass


def _qcdq_pass(name: str, fn_name: str, description: str):
    """Factory that defers the import so this module stays light when
    `onnx4deeploy.optimization.qcdq_to_deeploy` (depends on `onnx`) isn't
    importable in some minimal CI environments.
    """
    import onnx as _onnx

    class _Pass(OptimizationPass):
        def __init__(self):
            super().__init__(name=name, description=description)

        def apply(self, onnx_file: str, output_file: str, config: PassConfig) -> bool:
            try:
                from ..optimization import qcdq_to_deeploy

                fn = getattr(qcdq_to_deeploy, fn_name)
                m = _onnx.load(onnx_file)
                count = fn(m.graph)
                _onnx.save(m, output_file)
                print(f"    {name}: {count}")
                return True
            except Exception as e:
                print(f"    Error in {name}: {e}")
                return False

    _Pass.__name__ = f"Quant_{name}"
    return _Pass


# Each Quant_* class wraps one in-place ONNX rewrite from qcdq_to_deeploy.
QuantRemoveInitializersFromInputsPass = _qcdq_pass(
    "remove_initializers_from_inputs",
    "remove_initializers_from_inputs",
    "Drop weight initializers from graph.input (PyTorch keep_initializers_as_inputs cleanup)",
)
QuantUpgradeReduceMeanAxesPass = _qcdq_pass(
    "upgrade_reducemean_axes",
    "upgrade_reducemean_axes",
    "Rename opset-13 'axes' attribute to 'axis' so Deeploy's _remove_only_singleton_reduce_mean reads it",
)
QuantFoldQcdqToQuantDequantPass = _qcdq_pass(
    "fold_qcdq_to_quant_dequant",
    "fold_qcdq_to_quant_dequant",
    "Collapse Div/Add/Round/Clip → Quant and Sub/Mul → Dequant; chase Cast(Constant) for bounds",
)
QuantConstfoldQuantOfInitializerPass = _qcdq_pass(
    "constfold_quant_of_initializer",
    "constfold_quant_of_initializer",
    "Statically apply Quant() to constant weights/biases; emit int initializers directly",
)
QuantFoldDequantQuantToRequantShiftPass = _qcdq_pass(
    "fold_dequant_quant_to_requantshift",
    "fold_dequant_quant_to_requantshift",
    "Match Dequant → Quant and replace with RequantShift (the QCDQ → RequantShift bridge)",
)
QuantSkipDequantBeforeIntegerOpPass = _qcdq_pass(
    "skip_dequant_before_integer_op",
    "skip_dequant_before_integer_op",
    "Drop standalone Dequant whose output only flows into integer-friendly ops (Conv/Gemm/Add/ReduceMean)",
)
QuantFoldStandaloneQuantToRequantShiftPass = _qcdq_pass(
    "fold_standalone_quant_to_requantshift",
    "fold_standalone_quant_to_requantshift",
    "Replace `int_op → Quant` chains with `int_op → RequantShift`",
)
QuantSkipLeadingQuantDequantPass = _qcdq_pass(
    "skip_leading_quant_dequant",
    "skip_leading_quant_dequant",
    "Drop the trailing Dequant of a leading `input → Quant → Dequant → ...` pair",
)
QuantAbsorbConvBiasIntoFollowingRequantShiftPass = _qcdq_pass(
    "absorb_conv_bias_into_following_requantshift",
    "absorb_conv_bias_into_following_requantshift",
    "Fold Conv.bias × RequantShift.mul into RequantShift.add (4-input RequantizedConv)",
)
QuantStripTrailingDequantPass = _qcdq_pass(
    "strip_trailing_dequant",
    "strip_trailing_dequant",
    "Drop the trailing Dequant feeding the graph output; output stays int8",
)
QuantCleanupOrphanNodesPass = _qcdq_pass(
    "cleanup_orphan_nodes",
    "cleanup_orphan_nodes",
    "Remove unused Constants/Casts/Identity nodes and orphan initializers",
)


class QuantInputOfflinePass(OptimizationPass):
    """Pre-quantize inputs.npz and strip the leading Quant from the graph.

    Stands apart from the other `_qcdq_pass`-wrapped ones because it also
    needs to rewrite inputs.npz (not just the ONNX). The inputs.npz path
    is taken from config.params["inputs_npz_path"].
    """

    def __init__(self):
        super().__init__(
            name="quantize_input_offline",
            description="Pre-quantize inputs.npz to int8 and strip leading Quant from the graph",
        )

    def apply(self, onnx_file: str, output_file: str, config: PassConfig) -> bool:
        try:
            import onnx as _onnx

            from ..optimization.qcdq_to_deeploy import quantize_input_offline

            inputs_npz_path = config.params.get("inputs_npz_path")
            if inputs_npz_path is None:
                print(f"    Skipped: {self.name} requires inputs_npz_path in config.params")
                return True
            m = _onnx.load(onnx_file)
            count = quantize_input_offline(m, inputs_npz_path)
            _onnx.save(m, output_file)
            print(f"    {self.name}: {count}")
            return True
        except Exception as e:
            print(f"    Error in {self.name}: {e}")
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
    # Quantization-export passes
    "quant_remove_initializers_from_inputs": QuantRemoveInitializersFromInputsPass,
    "quant_upgrade_reducemean_axes": QuantUpgradeReduceMeanAxesPass,
    "quant_fold_qcdq_to_quant_dequant": QuantFoldQcdqToQuantDequantPass,
    "quant_constfold_quant_of_initializer": QuantConstfoldQuantOfInitializerPass,
    "quant_fold_dequant_quant_to_requantshift": QuantFoldDequantQuantToRequantShiftPass,
    "quant_skip_dequant_before_integer_op": QuantSkipDequantBeforeIntegerOpPass,
    "quant_fold_standalone_quant_to_requantshift": QuantFoldStandaloneQuantToRequantShiftPass,
    "quant_skip_leading_quant_dequant": QuantSkipLeadingQuantDequantPass,
    "quant_absorb_conv_bias_into_following_requantshift": QuantAbsorbConvBiasIntoFollowingRequantShiftPass,
    "quant_input_offline": QuantInputOfflinePass,
    "quant_strip_trailing_dequant": QuantStripTrailingDequantPass,
    "quant_cleanup_orphan_nodes": QuantCleanupOrphanNodesPass,
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


def create_quant_pipeline(inputs_npz_path: Optional[str] = None) -> "OptimizationPipeline":
    """Default quantization-export optimization pipeline.

    Run on the QCDQ ONNX produced by ``DeepQuant.exportBrevitas`` after
    Brevitas has done its job. Each pass closes one specific impedance
    mismatch between DeepQuant's QDQ representation and the integer-pipeline
    Deeploy expects. After this pipeline finishes the ONNX is directly
    compilable by vanilla ``pulp-platform/Deeploy:devel`` — no Deeploy patches
    required.

    Pass order matters; see the per-pass docstring for *why*. Categories:

    * Cleanup of PyTorch / ONNX-runtime export artefacts
        - quant_remove_initializers_from_inputs

    * Opset-13 → Deeploy-readable form
        - quant_upgrade_reducemean_axes

    * QCDQ recognition (Brevitas decomposed form → single Quant/Dequant ops)
        - quant_fold_qcdq_to_quant_dequant

    * Static weight quantization (compile-time Quant of constants)
        - quant_constfold_quant_of_initializer

    * QDQ → RequantShift bridge (the core translation)
        - quant_fold_dequant_quant_to_requantshift
        - quant_skip_dequant_before_integer_op
        - quant_fold_standalone_quant_to_requantshift

    * Graph-boundary normalization (Deeploy has no tile-constraint for
      Quant/Dequant, so the network must start and end on integer ops)
        - quant_skip_leading_quant_dequant
        - quant_input_offline    (requires inputs_npz_path)
        - quant_strip_trailing_dequant

    * Deeploy folding-rule gap workarounds
        - quant_absorb_conv_bias_into_following_requantshift

    * Hygiene
        - quant_cleanup_orphan_nodes

    Args:
        inputs_npz_path: path to the calibration `inputs.npz`. If provided,
            the `quantize_input_offline` pass rewrites the npz to int8 and
            strips the leading Quant from the graph. Otherwise that pass is
            skipped (network input stays fp32 with a Quant first; Deeploy
            won't compile this — useful only for inspection).

    Returns:
        OptimizationPipeline preconfigured with all 12 passes in the right
        order. Use ``pipeline.disable_pass(name)`` to skip any of them.
    """
    from .optimization_pipeline import OptimizationPipeline

    pipeline = OptimizationPipeline(name="quant")

    # 1. PyTorch / ORT export hygiene
    pipeline.add_pass(QuantRemoveInitializersFromInputsPass())
    # 2. Opset compatibility shims
    pipeline.add_pass(QuantUpgradeReduceMeanAxesPass())
    # 3. Recognise Brevitas decomposed QCDQ structure
    pipeline.add_pass(QuantFoldQcdqToQuantDequantPass())
    # 4. Compile-time static quantization of weight/bias constants
    pipeline.add_pass(QuantConstfoldQuantOfInitializerPass())
    # 5. QDQ → RequantShift core translation (mid-network)
    pipeline.add_pass(QuantFoldDequantQuantToRequantShiftPass())
    pipeline.add_pass(QuantSkipDequantBeforeIntegerOpPass())
    pipeline.add_pass(QuantFoldStandaloneQuantToRequantShiftPass())
    # 6. Graph-boundary normalisation
    pipeline.add_pass(QuantSkipLeadingQuantDequantPass())
    # 7. Deeploy folding-rule patch (Conv-bias→RQS-add)
    pipeline.add_pass(QuantAbsorbConvBiasIntoFollowingRequantShiftPass())
    # 8. Input/output boundary clean-up (run after all other folds)
    input_pass = QuantInputOfflinePass()
    if inputs_npz_path is not None:
        pipeline.add_pass(
            input_pass, config=PassConfig(params={"inputs_npz_path": inputs_npz_path})
        )
    else:
        pipeline.add_pass(input_pass)
        pipeline.disable_pass("quantize_input_offline")
    pipeline.add_pass(QuantStripTrailingDequantPass())
    # 9. Hygiene
    pipeline.add_pass(QuantCleanupOrphanNodesPass())

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
