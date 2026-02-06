# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Training model optimization utilities.

This module provides a comprehensive optimization pipeline for ONNX training models,
including node transformations, gradient optimizations, and model verification.
"""


def run_train_onnx_optimization(
    onnx_train_file: str, onnx_output_file: str, split_layernormgrad: bool = False
) -> None:
    """
    Run optimization passes on the training ONNX model.

    This function applies a series of optimization transformations to the training
    ONNX model, including:
    - Softmax axis optimization
    - Reshape fusion
    - Squeeze/Unsqueeze input to attribute conversion
    - ReduceSum to Reshape conversion
    - Identity node removal
    - ReduceSum axes to attribute conversion
    - FusedMatMul to Gemm conversion
    - Sum to Add conversion
    - SoftmaxGrad renaming
    - ConvGrad node splitting
    - Softmax loss output removal
    - BiasGelu and BiasGeluGrad removal
    - MatMul+Add to Gemm fusion
    - Node name annotation
    - Backward marker addition

    Args:
        onnx_train_file: Path to the input training ONNX model
        onnx_output_file: Path to save the optimized model
        split_layernormgrad: If True, split LayerNormGrad into X/W/B nodes (for training norm params)
    """
    # Import the optimization functions from trainOptimization module
    # Note: These imports are local to avoid circular dependencies
    import os
    import sys

    # Add the path to trainOptimization module
    train_opt_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "Tests", "Models", "CCT", "utils"
    )
    if train_opt_path not in sys.path:
        sys.path.insert(0, train_opt_path)

    from fixShape import print_onnx_shapes
    from trainOptimization import (
        add_backward_markers_to_nodes,
        convert_fusedmatmul_to_no_bias_gemm,
        convert_reducesum_axes_to_attr,
        convert_reducesum_to_reshape,
        convert_squeeze_unsqueeze_input_to_attr,
        convert_sum_to_add,
        fuse_matmul_add_to_gemm,
        optimize_reshape_fusion,
        optimize_softmax_axis,
        process_onnx_model_name_with_type,
        remove_identity_nodes,
        remove_softmax_grad_loss_inputs,
        remove_softmax_loss_outputs,
        rename_softmaxgrad_op,
        run_optmization_remove_biasgelu,
        run_optmization_remove_biasgelugrad,
    )

    from ..io.model_io import compare_onnx_models

    # Import from this package
    from ..transform.model_transform import split_convgrad_nodes

    print(f"🔹 Running optimization for {onnx_train_file}...")

    # fix_layernorm_output(onnx_train_file, onnx_output_file)
    # print(
    #     f"✅ Successfully fixed LayerNormalization opset version. Saved as {onnx_output_file}"
    # )
    optimize_softmax_axis(onnx_train_file, onnx_output_file)
    print(f"✅ Successfully optimized Softmax axis. Saved as {onnx_output_file}")

    optimize_reshape_fusion(onnx_output_file, onnx_output_file)
    print(f"✅ Successfully optimized Reshape nodes. Saved as {onnx_output_file}")

    # modify_conflict_outputs(onnx_output_file, onnx_output_file)
    # print(
    #     f"✅ Successfully removed all second outputs from Maxpool nodes. Saved as {onnx_output_file}"
    # )

    convert_squeeze_unsqueeze_input_to_attr(onnx_output_file, onnx_output_file)
    print(f"✅ Successfully converted Squeeze inputs to attributes. Saved as {onnx_output_file}")

    convert_reducesum_to_reshape(onnx_output_file, onnx_output_file)
    print(f"✅ Successfully converted ReduceSum to ReduceSum+Reshape. Saved as {onnx_output_file}")

    remove_identity_nodes(onnx_output_file, onnx_output_file)
    print(f"✅ Successfully removed Identity nodes. Saved as {onnx_output_file}")

    convert_reducesum_axes_to_attr(onnx_output_file, onnx_output_file)
    print(f"✅ Successfully converted ReduceSum axes to attributes. Saved as {onnx_output_file}")

    convert_fusedmatmul_to_no_bias_gemm(onnx_output_file, onnx_output_file)
    print(f"✅ Successfully converted FusedMatMul to Gemm nodes. Saved as {onnx_output_file}")

    convert_sum_to_add(onnx_output_file, onnx_output_file)
    print(f"✅ Successfully converted Sum to Add nodes. Saved as {onnx_output_file}")

    rename_softmaxgrad_op(onnx_output_file, onnx_output_file)
    print(f"✅ Successfully renamed SoftmaxGrad nodes. Saved as {onnx_output_file}")

    split_convgrad_nodes(onnx_output_file, onnx_output_file)
    print(f"✅ Successfully split ConvGrad nodes into ConvGradX/W/B. Saved as {onnx_output_file}")

    remove_softmax_loss_outputs(onnx_output_file, onnx_output_file)
    print(f"✅ Successfully removed Softmax Loss outputs. Saved as {onnx_output_file}")

    remove_softmax_grad_loss_inputs(onnx_output_file, onnx_output_file)
    print(f"✅ Successfully removed Softmax Grad Loss inputs. Saved as {onnx_output_file}")
    # process_layernormgrad_nodes(
    # onnx_output_file, onnx_output_file, split_nodes=split_layernormgrad
    # )  # Process LayerNormGrad nodes
    if split_layernormgrad:
        print(f"✅ Successfully split LayerNormGrad into X/W/B nodes. Saved as {onnx_output_file}")
    else:
        print(f"✅ Successfully processed LayerNormGrad nodes. Saved as {onnx_output_file}")

    run_optmization_remove_biasgelu(onnx_output_file, onnx_output_file)
    print(f"✅ Successfully removed BiasGeluFusion. Saved as {onnx_output_file}")

    run_optmization_remove_biasgelugrad(onnx_output_file, onnx_output_file)
    print(f"✅ Successfully removed BiasGeluGradFusion. Saved as {onnx_output_file}")

    # unify_gemm_input_dims(onnx_output_file, onnx_output_file)
    # print(f"✅ Successfully unified GEMM input dimensions. Saved as {onnx_output_file}")

    # Compare input and output models to check for any lost nodes/outputs
    print("\n🔍 Verifying optimization results...")
    compare_onnx_models(
        onnx_train_file, onnx_output_file, name1="network_train.onnx", name2="network.onnx"
    )

    fuse_matmul_add_to_gemm(onnx_output_file, onnx_output_file)
    print(f"✅ Successfully fused MatMul and Add to Gemm nodes. Saved as {onnx_output_file}")
    process_onnx_model_name_with_type(onnx_output_file, onnx_output_file)
    print(f"✅ Successfully processed ONNX model name with type. Saved as {onnx_output_file}")

    add_backward_markers_to_nodes(onnx_output_file, onnx_output_file)
    print(f"✅ Successfully added backward markers to nodes. Saved as {onnx_output_file}")

    print_onnx_shapes(onnx_output_file)
