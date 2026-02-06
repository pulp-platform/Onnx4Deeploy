"""ONNX model optimization utilities.

This package provides various optimization functions for ONNX models:
- GEMM conversion and optimization (gemm_converter)
- Gradient node optimization (gradient_optimizer)
- Graph cleaning and fixing (graph_cleaner)
- Shape operation optimization (shape_optimizer)
- Node utility functions (node_utils)
- Model optimization (model_optimizer)
- Training optimization (train_optimizer)
"""

# GEMM conversions
from .gemm_converter import (
    add_c_to_gemm,
    convert_fusedmatmul_to_no_bias_gemm,
    convert_gemm_to_transpose_matmul,
    fuse_matmul_add_to_gemm,
    unify_gemm_input_dims,
)

# Gradient optimizations
from .gradient_optimizer import (
    process_layernormgrad_nodes,
    remove_softmax_grad_loss_inputs,
    rename_softmaxgrad_op,
    run_optmization_remove_biasgelugrad,
)

# Graph cleaning
from .graph_cleaner import (
    fix_layernorm_output,
    modify_conflict_outputs,
    optimize_softmax_axis,
    remove_identity_nodes,
    remove_softmax_loss_outputs,
    replace_biasgelu_with_gelu_add,
    run_optmization_remove_biasgelu,
)

# Model optimization
from .model_optimizer import run_onnx_optimization, run_onnx_optimization_infer

# Node utilities
from .node_utils import (
    add_backward_markers_to_nodes,
    annotate_node_names_with_type,
    convert_layernorm_to_groupnorm,
    process_onnx_model_name_with_type,
    split_gn_to_single_stat_array,
)

# Shape optimizations
from .shape_optimizer import (
    convert_reducesum_axes_to_attr,
    convert_reducesum_to_reshape,
    convert_squeeze_unsqueeze_input_to_attr,
    convert_sum_to_add,
    optimize_reshape_fusion,
)

# Training optimization
from .train_optimizer import run_train_onnx_optimization

__all__ = [
    # GEMM
    "add_c_to_gemm",
    "convert_gemm_to_transpose_matmul",
    "unify_gemm_input_dims",
    "fuse_matmul_add_to_gemm",
    "convert_fusedmatmul_to_no_bias_gemm",
    # Gradient
    "run_optmization_remove_biasgelugrad",
    "rename_softmaxgrad_op",
    "remove_softmax_grad_loss_inputs",
    "process_layernormgrad_nodes",
    # Graph cleaning
    "remove_identity_nodes",
    "remove_softmax_loss_outputs",
    "modify_conflict_outputs",
    "fix_layernorm_output",
    "replace_biasgelu_with_gelu_add",
    "run_optmization_remove_biasgelu",
    "optimize_softmax_axis",
    # Shape
    "optimize_reshape_fusion",
    "convert_reducesum_to_reshape",
    "convert_squeeze_unsqueeze_input_to_attr",
    "convert_reducesum_axes_to_attr",
    "convert_sum_to_add",
    # Node utils
    "annotate_node_names_with_type",
    "process_onnx_model_name_with_type",
    "add_backward_markers_to_nodes",
    "convert_layernorm_to_groupnorm",
    "split_gn_to_single_stat_array",
    # Model optimization
    "run_onnx_optimization_infer",
    "run_onnx_optimization",
    # Training optimization
    "run_train_onnx_optimization",
]
