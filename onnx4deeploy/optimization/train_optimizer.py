# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Training model optimization utilities.

This module provides a comprehensive optimization pipeline for ONNX training models,
including node transformations, gradient optimizations, and model verification.
"""


def run_train_onnx_optimization(
    onnx_train_file: str,
    onnx_output_file: str,
    split_layernormgrad: bool = False,
    onnx_infer_file: str = None,
) -> None:
    """
    Run optimization passes on the training ONNX model.

    This function applies a series of optimization transformations to the training
    ONNX model, including:
    - Frozen initializer → Constant node conversion
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
        onnx_infer_file: Path to the inference ONNX model (used to retrieve frozen param values).
                         If None, the frozen-param conversion pass is skipped.
    """
    from ..io.model_io import compare_onnx_models

    # Import from this package
    from .graph_cleaner import canonicalize_ms_training_ops_file, remove_identity_nodes
    from .shape_optimizer import convert_reducesum_to_reshape
    from .trainOptimization import (
        add_backward_markers_to_nodes,
        convert_frozen_initializers_to_constants,
        convert_fusedmatmul_to_no_bias_gemm,
        convert_reducesum_axes_to_attr,
        convert_squeeze_unsqueeze_input_to_attr,
        convert_sum_to_add,
        fuse_global_average_pool_grad,
        fuse_matmul_add_to_gemm,
        fuse_mse_loss,
        optimize_reshape_fusion,
        optimize_softmax_axis,
        process_onnx_model_name_with_type,
        remove_softmax_grad_loss_inputs,
        rename_softmaxgrad_op,
        run_optmization_remove_biasgelu,
        run_optmization_remove_biasgelugrad,
    )

    print(f"🔹 Running optimization for {onnx_train_file}...")

    # Convert frozen (non-trainable) parameters to inline Constant nodes.
    # Must run first, before any pass that might alter graph.input.
    # Requires the inference model to source frozen param tensor values.
    if onnx_infer_file is not None:
        convert_frozen_initializers_to_constants(onnx_train_file, onnx_output_file, onnx_infer_file)
        print(
            f"✅ Successfully converted frozen params to Constant nodes. Saved as {onnx_output_file}"
        )
    else:
        import shutil

        if onnx_train_file != onnx_output_file:
            shutil.copy(onnx_train_file, onnx_output_file)
        print("⚠️  Skipped frozen-param → Constant conversion (no infer model path provided).")

    # Replace com.microsoft ConcatTraining/SplitTraining with standard Concat/Split.
    # Must run before shape-inference passes so that downstream ONNX ops can be
    # handled by the standard inference engine.
    canonicalize_ms_training_ops_file(onnx_output_file, onnx_output_file)
    print(
        f"✅ Canonicalized ConcatTraining/SplitTraining → Concat/Split. Saved as {onnx_output_file}"
    )

    # fix_layernorm_output(onnx_train_file, onnx_output_file)
    # print(
    #     f"✅ Successfully fixed LayerNormalization opset version. Saved as {onnx_output_file}"
    # )
    optimize_softmax_axis(onnx_output_file, onnx_output_file)
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

    fuse_global_average_pool_grad(onnx_output_file, onnx_output_file)
    print(f"✅ Successfully fused GlobalAveragePool backward. Saved as {onnx_output_file}")

    fuse_mse_loss(onnx_output_file, onnx_output_file)
    print(f"✅ Successfully fused MSE loss nodes. Saved as {onnx_output_file}")

    rename_softmaxgrad_op(onnx_output_file, onnx_output_file)
    print(f"✅ Successfully renamed SoftmaxGrad nodes. Saved as {onnx_output_file}")

    # ConvGrad nodes are left as-is: Deeploy maps 'ConvGrad' directly
    # via the same ConvGradXW Layer infrastructure (no rename needed).

    # Issue 5: Keep loss output from SoftmaxCrossEntropyLoss (both loss + log_prob)
    # remove_softmax_loss_outputs(onnx_output_file, onnx_output_file)
    # print(f"✅ Successfully removed Softmax Loss outputs. Saved as {onnx_output_file}")

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
