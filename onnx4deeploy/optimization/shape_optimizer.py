# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Shape optimization utilities for ONNX models.

This module contains functions for optimizing shape-related operations in ONNX models,
including reshape fusion, squeeze/unsqueeze conversions, reduce operations, and shape inference.
"""

from typing import List, Optional

import onnx
from onnx import helper, numpy_helper, shape_inference


def optimize_reshape_fusion(input_model_path: str, output_model_path: str) -> None:
    """
    Optimize ONNX model by fusing consecutive Reshape operations.

    Args:
        input_model_path: Path to the input ONNX model
        output_model_path: Path where the optimized ONNX model will be saved
    """
    print(f"Loading model: {input_model_path}")
    model = onnx.load(input_model_path)

    # Create mapping from node name to node
    node_map = {}
    for node in model.graph.node:
        node_map[node.name] = node

    # Create mapping from input name to producing node
    input_to_node = {}
    for node in model.graph.node:
        for output in node.output:
            input_to_node[output] = node

    # Create mapping from output name to consuming nodes
    output_to_nodes = {}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name not in output_to_nodes:
                output_to_nodes[input_name] = []
            output_to_nodes[input_name].append(node)

    # Find all Reshape nodes
    reshape_nodes = [node for node in model.graph.node if node.op_type == "Reshape"]

    # Track nodes to be removed by index rather than node objects
    # This avoids the "unhashable type: 'NodeProto'" error
    nodes_to_remove_indices = []

    # Track value info to keep
    value_info_to_keep = set(vi.name for vi in model.graph.value_info)

    # For each Reshape node, check if its input is also from a Reshape node
    for reshape_node in reshape_nodes:
        # Get the input of the current Reshape node
        input_name = reshape_node.input[0]

        # Check if the input comes from another Reshape operation
        if input_name in input_to_node and input_to_node[input_name].op_type == "Reshape":
            previous_reshape = input_to_node[input_name]

            # Check if the previous Reshape is only used by the current Reshape
            if input_name in output_to_nodes and len(output_to_nodes[input_name]) == 1:
                print(f"Found fusible Reshape pair: {previous_reshape.name} -> {reshape_node.name}")

                # Get the shape tensors for both Reshape nodes
                previous_reshape.input[1]
                reshape_node.input[1]

                # Modify the current Reshape node to connect directly to the input of the previous Reshape
                reshape_node.input[0] = previous_reshape.input[0]

                # Mark the previous Reshape node for removal by its index
                for i, node in enumerate(model.graph.node):
                    if (
                        node.name == previous_reshape.name
                        and node.op_type == previous_reshape.op_type
                        and node.input == previous_reshape.input
                        and node.output == previous_reshape.output
                    ):
                        nodes_to_remove_indices.append(i)
                        break

                # Intermediate value info doesn't need to be kept
                if input_name in value_info_to_keep:
                    value_info_to_keep.remove(input_name)

    # Handle custom nodes from Microsoft
    # Since Microsoft nodes might have a different structure or behavior
    # We need to be careful when dealing with them
    custom_nodes = [node for node in model.graph.node if node.domain.startswith("com.microsoft")]
    print(f"Found {len(custom_nodes)} Microsoft custom nodes. These will be preserved.")

    # Create a new graph excluding the nodes to be removed
    new_nodes = []
    for i, node in enumerate(model.graph.node):
        if i not in nodes_to_remove_indices:
            new_nodes.append(node)

    # Create a new value info list, keeping only the needed value info
    new_value_info = [vi for vi in model.graph.value_info if vi.name in value_info_to_keep]

    # Create a new graph
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=model.graph.initializer,
        value_info=new_value_info,
    )

    # Create a new model
    new_model = helper.make_model(
        new_graph,
        producer_name="ONNX Reshape Fusion Optimizer",
        ir_version=model.ir_version,
        opset_imports=model.opset_import,
    )

    # Preserve custom opsets from the original model
    new_model.opset_import.extend(
        [opset for opset in model.opset_import if opset.domain.startswith("com.microsoft")]
    )

    # Save the optimized model
    onnx.save(new_model, output_model_path)

    # Print statistics
    print(f"Original model node count: {len(model.graph.node)}")
    print(f"Optimized model node count: {len(new_model.graph.node)}")
    print(f"Removed Reshape nodes: {len(nodes_to_remove_indices)}")
    print(f"Optimized model saved to: {output_model_path}")


def convert_reducesum_to_reshape(input_model_path: str, output_model_path: str):
    """
    Convert ReduceSum nodes (with keepdims=0) to ReduceSum (keepdims=1) + Reshape.
    This makes the shape transformation explicit and easier for tiling optimization.
    """
    import numpy as np
    import onnx
    from onnx import helper

    model = onnx.load(input_model_path)
    graph = model.graph

    nodes_to_add = []
    nodes_to_remove = []

    print("\n🔍 Converting ReduceSum nodes to ReduceSum + Reshape:")

    for node in graph.node:
        if node.op_type == "ReduceSum":
            # Check if keepdims is 0
            keepdims = 1  # default value in ONNX
            for attr in node.attribute:
                if attr.name == "keepdims":
                    keepdims = attr.i
                    break

            if keepdims == 0:
                print(f"\n  Processing: {node.name}")
                print(f"    Original output: {node.output[0]}")

                # Get the axes being reduced
                axes = None
                for attr in node.attribute:
                    if attr.name == "axes":
                        axes = list(attr.ints)
                        break

                if axes is None:
                    print("    ⚠️  No axes specified, skipping")
                    continue

                print(f"    Reducing over axes: {axes}")

                # Create new ReduceSum with keepdims=1
                intermediate_output = node.output[0] + "_keepdims"

                # Create new ReduceSum node with keepdims=1
                new_reducesum = helper.make_node(
                    "ReduceSum",
                    inputs=node.input,
                    outputs=[intermediate_output],
                    name=node.name,
                    axes=axes,
                    keepdims=1,
                )

                # Create Reshape node to remove the reduced dimensions
                # Use -1 to indicate dynamic shape inference
                target_shape = np.array([-1], dtype=np.int64)
                shape_constant = helper.make_tensor(
                    name=node.name + "_shape",
                    data_type=onnx.TensorProto.INT64,
                    dims=[1],
                    vals=target_shape,
                )

                # Add shape constant to initializers if not already present
                if not any(init.name == shape_constant.name for init in graph.initializer):
                    graph.initializer.append(shape_constant)

                # Create Reshape node
                reshape_node = helper.make_node(
                    "Reshape",
                    inputs=[intermediate_output, shape_constant.name],
                    outputs=node.output,  # Keep original output name
                    name=node.name + "_Reshape",
                )

                nodes_to_add.append((new_reducesum, reshape_node))
                nodes_to_remove.append(node)
                print(f"    ✅ Split into: {new_reducesum.name} (keepdims=1) → {reshape_node.name}")

    # Remove old nodes and add new ones
    for node in nodes_to_remove:
        graph.node.remove(node)

    for new_reducesum, reshape_node in nodes_to_add:
        graph.node.extend([new_reducesum, reshape_node])

    onnx.save(model, output_model_path)
    print(f"\n✅ Converted {len(nodes_to_add)} ReduceSum node(s) to ReduceSum + Reshape")

    return model


def convert_squeeze_unsqueeze_input_to_attr(input_model_path: str, output_model_path: str):
    """
    Convert Squeeze and Unsqueeze nodes with axes as input to axes as attribute.
    This is useful for compatibility with older ONNX versions where axes was only supported as an attribute.

    Args:
        input_model_path: Path to the input ONNX model
        output_model_path: Path to save the converted ONNX model
    """
    model = onnx.load(input_model_path)

    modified_nodes = []
    modified_count = 0

    initializers = {init.name: init for init in model.graph.initializer}

    for node in model.graph.node:
        # Check if the node is Squeeze or Unsqueeze with more than one input
        if (node.op_type in ["Squeeze", "Unsqueeze"]) and len(node.input) > 1:
            modified_count += 1

            data_input = node.input[0]

            axes_input_name = node.input[1]

            if axes_input_name in initializers:
                # Get the axes values from the initializer
                axes_initializer = initializers[axes_input_name]
                axes_np = numpy_helper.to_array(axes_initializer)
                axes_list = axes_np.tolist()

                # Make the axes a scalar if it's a single value
                if isinstance(axes_list, list) and len(axes_list) == 1:
                    axes_list = axes_list[0]

                # Create a new node with axes as attribute instead of input
                new_node = helper.make_node(
                    op_type=node.op_type,
                    inputs=[data_input],
                    outputs=list(node.output),
                    name=node.name,
                    axes=axes_list,
                )

                # Copy other attributes if they exist
                for attr in node.attribute:
                    if attr.name != "axes":
                        new_node.attribute.append(attr)

                modified_nodes.append(new_node)
            else:
                # If we can't find the axes initializer, keep the original node
                print(
                    f"Warning: Cannot find '{node.name}' axes initializer. Keep the original node."
                )
                modified_nodes.append(node)
        else:
            # Keep all other nodes as they are
            modified_nodes.append(node)

    print(f"Modified {modified_count} Squeeze/Unsqueeze nodes")

    # Identify initializers that are no longer referenced
    # This happens when we convert the axes from input to attribute
    used_inputs = set()
    for node in modified_nodes:
        for input_name in node.input:
            used_inputs.add(input_name)

    unused_initializers = set()
    for init in model.graph.initializer:
        if init.name not in used_inputs:
            unused_initializers.add(init.name)

    # Create a new graph with the modified nodes and without unused initializers
    new_graph = helper.make_graph(
        nodes=modified_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=[
            init for init in model.graph.initializer if init.name not in unused_initializers
        ],
    )

    # Copy over value_info from the original model
    for vi in model.graph.value_info:
        new_graph.value_info.append(vi)

    # Create a new model with the updated graph
    new_model = helper.make_model(
        new_graph,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        domain=model.domain,
        model_version=model.model_version,
        doc_string=model.doc_string,
    )

    # Copy over IR version and opset imports
    new_model.ir_version = model.ir_version
    new_model.opset_import.extend(model.opset_import)

    # Save the model
    onnx.save(new_model, output_model_path)
    print(f"Saved to {output_model_path}")

    return new_model


def convert_reducesum_axes_to_attr(input_file: str, output_file: str):
    model = onnx.load(input_file)
    graph = model.graph

    new_nodes = []

    initializers = {init.name: init for init in graph.initializer}

    for node in graph.node:
        if node.op_type == "ReduceSum":
            if len(node.input) >= 2:
                data_input = node.input[0]
                axes_input = node.input[1]

                if axes_input in initializers:
                    axes_tensor = initializers[axes_input]
                    axes_np = numpy_helper.to_array(axes_tensor)
                    axes_list = axes_np.tolist()

                    new_node = helper.make_node(
                        op_type="ReduceSum",
                        inputs=[data_input],
                        outputs=node.output,
                        name=node.name,
                        axes=axes_list,
                    )

                    for attr in node.attribute:
                        if attr.name != "axes":
                            new_node.attribute.append(attr)

                    new_nodes.append(new_node)
                else:
                    new_nodes.append(node)
            else:
                new_nodes.append(node)
        else:
            new_nodes.append(node)

    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=graph.name,
        inputs=graph.input,
        outputs=graph.output,
        initializer=graph.initializer,
        value_info=graph.value_info,
    )

    new_model = helper.make_model(
        new_graph,
        producer_name="ReduceSumAxesConverter",
        ir_version=model.ir_version,
        opset_imports=model.opset_import,
    )

    new_model.metadata_props.extend(model.metadata_props)

    for domain in model.domain:
        new_model.domain.append(domain)

    onnx.save(new_model, output_file)
    print(f"Model converted and saved to: {output_file}")


def convert_sum_to_add(input_model_path: str, output_model_path: str):
    """
    Convert Sum operators to Add operators in an ONNX model.
    Sum operator can take multiple inputs, while Add takes exactly two inputs.
    This function breaks down Sum operators with >2 inputs into a series of Add operators.

    Args:
        input_model_path: Path to the input ONNX model
        output_model_path: Path to save the converted ONNX model
    """
    # Load the model
    model = onnx.load(input_model_path)

    # Build a name → (elem_type, shape) lookup so we can stamp value_info onto
    # the new intermediate tensors emitted when an N-ary Sum is split into a
    # chain of binary Adds. Without this Deeploy's front-end shape assertion
    # fails with "Found tensors with missing shape annotation:
    # <Sum_node>_intermediate_<j>" — a regression that surfaces whenever the
    # gradient at a residual feed-point has more than 3 contributors (e.g. LoRA
    # adapters add 3 extra branches per Q/K/V on the attention input).
    shape_lookup: dict = {}
    for source in (model.graph.input, model.graph.output, model.graph.value_info):
        for vi in source:
            if vi.type.tensor_type.HasField("shape"):
                dims = [
                    d.dim_value if d.HasField("dim_value") else 0
                    for d in vi.type.tensor_type.shape.dim
                ]
                shape_lookup[vi.name] = (vi.type.tensor_type.elem_type, dims)
    for init in model.graph.initializer:
        shape_lookup[init.name] = (init.data_type, list(init.dims))

    # Track necessary changes
    new_nodes = []
    new_value_info = []
    processed_nodes = set()

    # Process each node in the graph
    for i, node in enumerate(model.graph.node):
        # Skip already processed nodes
        if i in processed_nodes:
            continue

        # Check if the node is a Sum operator
        if node.op_type == "Sum":
            input_count = len(node.input)

            if input_count == 1:
                # Sum with one input is just an Identity
                identity_node = helper.make_node(
                    "Identity",
                    inputs=[node.input[0]],
                    outputs=node.output,
                    name=f"{node.name}_identity",
                )
                new_nodes.append(identity_node)

            elif input_count == 2:
                # Sum with two inputs can be directly converted to Add
                add_node = helper.make_node(
                    "Add",
                    inputs=[node.input[0], node.input[1]],
                    outputs=node.output,
                    name=f"{node.name}_add",
                )
                new_nodes.append(add_node)

            else:
                # Sum with more than two inputs needs to be broken down into a series of Add operations
                # We'll create intermediate outputs for all but the last Add
                intermediate_outputs = []

                # Source the elem-type/shape from any input we know about; Adds
                # are element-wise so every intermediate has the same shape.
                inferred_dtype = None
                inferred_shape = None
                for inp in node.input:
                    if inp in shape_lookup:
                        inferred_dtype, inferred_shape = shape_lookup[inp]
                        break

                for j in range(input_count - 1):
                    if j == 0:
                        # First Add takes the first two inputs of Sum
                        input1 = node.input[0]
                        input2 = node.input[1]
                    else:
                        # Subsequent Adds take the output of the previous Add and the next input
                        input1 = intermediate_outputs[-1]
                        input2 = node.input[j + 1]

                    # For the last Add, use the original output, otherwise create an intermediate output
                    if j == input_count - 2:
                        output = node.output[0]
                    else:
                        output = f"{node.name}_intermediate_{j}"
                        intermediate_outputs.append(output)
                        # Stamp value_info so downstream shape assertions pass.
                        if inferred_dtype is not None and inferred_shape is not None:
                            new_value_info.append(
                                helper.make_tensor_value_info(
                                    output, inferred_dtype, inferred_shape
                                )
                            )
                            shape_lookup[output] = (inferred_dtype, inferred_shape)

                    # Create the Add node
                    add_node = helper.make_node(
                        "Add",
                        inputs=[input1, input2],
                        outputs=[output],
                        name=f"{node.name}_add_{j}",
                    )
                    new_nodes.append(add_node)

            # Mark this node as processed
            processed_nodes.add(i)
        else:
            # Keep other nodes as they are
            new_nodes.append(node)

    # Create a new graph with updated nodes
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=model.graph.initializer,
        value_info=list(model.graph.value_info) + new_value_info,
    )

    # Create a new model with the updated graph
    # Preserve opset imports and other model metadata
    new_model = helper.make_model(
        new_graph,
        producer_name="SumToAddConverter",
        opset_imports=model.opset_import,
        ir_version=model.ir_version,
    )

    # Copy domain information for custom ops
    for domain in model.domain:
        new_model.domain.append(domain)

    # Copy model metadata
    new_model.metadata_props.extend(model.metadata_props)

    # Save the new model
    onnx.save(new_model, output_model_path)
    print(f"Converted model saved to {output_model_path}")

    return new_model


def register_custom_shape_inference():
    """
    Register custom shape inference functions for Microsoft training operators.

    This is required for training models that use custom Microsoft ops like
    SoftmaxCrossEntropyGrad, which don't have built-in shape inference.

    Note: This uses internal ONNX APIs that may not be available in all versions.
    If registration fails, fallback to manual shape inference will be used.
    """
    try:
        from onnx.shape_inference import _get_shape_calculator_dict

        def softmax_cross_entropy_grad_shape_inference(ctx):
            """Shape inference for SoftmaxCrossEntropyGrad operator."""
            ctx.node

            # Get log_prob input type (3rd input)
            log_prob_type_proto = ctx.get_input_type(2)
            if log_prob_type_proto is None:
                return

            # Output shape matches log_prob input shape
            ctx.set_output_type(0, log_prob_type_proto)
            print(
                f"  SoftmaxCrossEntropyGrad shape inference: output shape set from log_prob input"
            )

        def inplace_accumulator_v2_shape_inference(ctx):
            """Shape inference for InPlaceAccumulatorV2 operator.

            Output shape = buffer (input[0]) shape, same as gradient (input[1]).
            """
            buffer_type_proto = ctx.get_input_type(0)
            if buffer_type_proto is None:
                return
            ctx.set_output_type(0, buffer_type_proto)

        # Register the custom shape inference functions
        shape_calculator_dict = _get_shape_calculator_dict()
        shape_calculator_dict["com.microsoft.SoftmaxCrossEntropyGrad"] = (
            softmax_cross_entropy_grad_shape_inference
        )
        shape_calculator_dict["com.microsoft.InPlaceAccumulatorV2"] = (
            inplace_accumulator_v2_shape_inference
        )
        # Also register without domain prefix for models using node.op_type directly
        shape_calculator_dict["InPlaceAccumulatorV2"] = inplace_accumulator_v2_shape_inference
        return True
    except (ImportError, AttributeError):
        # Internal API not available, will use fallback
        return False


def infer_shapes_with_custom_ops(
    model_path: str, output_model_path: Optional[str] = None
) -> onnx.ModelProto:
    """
    Infer shapes for ONNX model with support for Microsoft custom training operators.

    This function handles shape inference for models that contain Microsoft custom ops
    (e.g., training models with gradient operators). It attempts standard shape inference
    first, then falls back to node-by-node inference with custom handling for Microsoft ops.

    Args:
        model_path: Path to the input ONNX model
        output_model_path: Optional path to save the model with inferred shapes

    Returns:
        ONNX model with inferred shapes
    """
    print("\n🔍 Running shape inference with custom op support...")
    model = onnx.load(model_path)

    # Strip 'description' custom attributes — ORT shape inference rejects them on
    # operators that don't declare this attribute in their schema (e.g. Size, Reshape).
    for node in model.graph.node:
        attrs_to_keep = [a for a in node.attribute if a.name != "description"]
        if len(attrs_to_keep) != len(node.attribute):
            del node.attribute[:]
            node.attribute.extend(attrs_to_keep)

    # Check for Microsoft custom ops
    op_types = set(node.op_type for node in model.graph.node)
    microsoft_ops = [op for op in op_types if "com.microsoft" in op]
    if microsoft_ops:
        print(f"  Found Microsoft custom ops: {microsoft_ops}")

    try:
        # Register custom shape inference for Microsoft ops (if available)
        registration_success = register_custom_shape_inference()
        if not registration_success:
            print("  ℹ️  Custom op registration not available, using fallback for Microsoft ops")

        # Try standard shape inference
        inferred_model = shape_inference.infer_shapes(model)
        print("  ✅ Shape inference succeeded")
    except Exception as e:
        print(f"  ⚠️  Standard shape inference failed: {str(e)}")
        print("  Attempting node-by-node inference...")

        inferred_model = model
        for i, node in enumerate(model.graph.node):
            try:
                # Try to infer shape for this node
                subgraph_model = extract_subgraph(model, [node])
                inferred_subgraph = shape_inference.infer_shapes(subgraph_model)
                update_model_with_inferred_shapes(inferred_model, inferred_subgraph, i)
                print(f"    Node {i}: {node.op_type} ✓")
            except Exception as node_err:
                print(f"    Node {i}: {node.op_type} failed: {str(node_err)}")
                # Try custom inference for Microsoft ops
                if "com.microsoft" in node.op_type:
                    print(f"    Applying custom inference for: {node.op_type}")
                    try:
                        apply_custom_inference(inferred_model.graph, node)
                        print(f"    Node {i}: {node.op_type} ✓ (custom)")
                    except Exception as custom_err:
                        print(f"    Custom inference failed: {str(custom_err)}")

    # Standard ONNX shape inference skips unknown custom ops (com.microsoft domain).
    # Always apply our custom handler for these nodes so that their output shapes
    # are populated even when the standard pass succeeds for all other nodes.
    custom_nodes = [
        node
        for node in inferred_model.graph.node
        if node.domain == "com.microsoft" or node.op_type == "InPlaceAccumulatorV2"
    ]
    if custom_nodes:
        print(f"  🔧 Applying custom shape inference for {len(custom_nodes)} Microsoft op(s)...")
        for node in custom_nodes:
            apply_custom_inference(inferred_model.graph, node)

    # Apply custom inference for ALL nodes (not just com.microsoft ones) to handle
    # training-specific ops like ConcatTraining, SplitTraining, LayerNormalizationGrad,
    # BatchNormInternal, BatchNormalizationGrad that appear in backward-pass graphs.
    all_custom_ops = {
        "ConcatTraining",
        "SplitTraining",
        "LayerNormalizationGrad",
        "BatchNormInternal",
        "BatchNormalizationGrad",
        "GlobalAveragePoolGrad",
    }
    extra_nodes = [node for node in inferred_model.graph.node if node.op_type in all_custom_ops]
    if extra_nodes:
        print(f"  🔧 Applying custom shape inference for {len(extra_nodes)} training op(s)...")
        for node in extra_nodes:
            apply_custom_inference(inferred_model.graph, node)

    # Standard ONNX infer_shapes cannot propagate past com.microsoft ops like
    # SoftmaxCrossEntropyLossGrad — their outputs remain unknown even if the
    # downstream nodes are standard Gemm ops with otherwise-known inputs.
    # Run a post-pass that manually infers any Gemm output still missing its shape.
    apply_gemm_shape_inference(inferred_model.graph)

    # Propagate shapes for downstream standard ops (ReduceSum, Reshape) that
    # depend on tensors whose shapes were just set by the custom handlers above.
    propagate_residual_shapes(inferred_model.graph)

    # Save if output path provided
    if output_model_path:
        onnx.save(inferred_model, output_model_path)
        print(f"  💾 Model with inferred shapes saved: {output_model_path}")

    return inferred_model


def apply_custom_inference(graph: onnx.GraphProto, node: onnx.NodeProto) -> None:
    """
    Apply custom shape inference for Microsoft training operators.

    Args:
        graph: ONNX graph containing the node
        node: Node to apply custom shape inference to
    """
    op = node.op_type

    if op in (
        "SoftmaxCrossEntropyLossGrad",
        "SoftmaxCrossEntropyGrad",
        "com.microsoft.SoftmaxCrossEntropyLossGrad",
        "com.microsoft.SoftmaxCrossEntropyGrad",
    ):
        # output_grad has same shape as log_prob.
        # In ORT-generated training models input[0] = log_prob (softmax log-probs,
        # same shape as model output), input[1] = labels (integer targets).
        if len(node.input) >= 1 and len(node.output) >= 1:
            # Skip if standard inference already set a multi-dimensional shape.
            existing = get_tensor_shape(graph, node.output[0])
            if existing and len(existing) > 1:
                return
            log_prob_shape = get_tensor_shape(graph, node.input[0])
            if log_prob_shape:
                set_tensor_shape(graph, node.output[0], log_prob_shape)
                print(f"    {op} output_grad shape: {log_prob_shape}")

    elif op == "InPlaceAccumulatorV2":
        # Output shape equals buffer (input[0]) / gradient (input[1]) shape.
        # InPlaceAccumulatorV2 semantics:
        #   if lazy_reset_grad: out = gradient          (reset)
        #   else:               out = buffer + gradient  (accumulate)
        # Either way output has the same shape as the first two inputs.
        # Also set the gradient (input[1]) shape from the buffer shape, because
        # ONNX shape inference cannot traverse com.microsoft ops (e.g. ReluGrad)
        # that produce the gradient tensor, leaving its shape unknown.
        if len(node.input) >= 2 and len(node.output) >= 1:
            buffer_shape = get_tensor_shape(graph, node.input[0])
            if buffer_shape:
                # Set output shape
                set_tensor_shape(graph, node.output[0], buffer_shape)
                print(f"    InPlaceAccumulatorV2 output shape: {buffer_shape}")
                # Propagate shape to gradient input (same shape as buffer)
                gradient_name = node.input[1]
                existing_grad_shape = get_tensor_shape(graph, gradient_name)
                if not existing_grad_shape or all(d == 0 for d in existing_grad_shape):
                    set_tensor_shape(graph, gradient_name, buffer_shape)
                    print(f"    InPlaceAccumulatorV2 gradient shape: {buffer_shape}")

    elif op in ("ConcatTraining", "com.microsoft.ConcatTraining"):
        # ConcatTraining: like Concat but also outputs per_input_length.
        # output[0] = concatenated tensor (standard ONNX can infer this from value_info)
        # output[1] = per_input_length: 1-D int64 tensor of length = num_inputs
        if len(node.output) >= 2 and node.output[1]:
            num_inputs = len(node.input)  # all inputs are tensors to concat
            sizes_name = node.output[1]
            existing = get_tensor_shape(graph, sizes_name)
            if not existing:
                set_tensor_shape(graph, sizes_name, [num_inputs])
                print(f"    ConcatTraining per_input_length shape: [{num_inputs}]")

    elif op in ("LayerNormalizationGrad", "com.microsoft.LayerNormalizationGrad"):
        # LayerNormalizationGrad outputs: [dX, d_Scale, d_Bias]
        # d_Scale and d_Bias have the same shape as the scale input (input[2]).
        if len(node.input) >= 3:
            scale_shape = get_tensor_shape(graph, node.input[2])
            if scale_shape:
                for out_idx in range(1, len(node.output)):
                    out_name = node.output[out_idx] if out_idx < len(node.output) else ""
                    if out_name:
                        existing = get_tensor_shape(graph, out_name)
                        if not existing:
                            set_tensor_shape(graph, out_name, scale_shape)
                            label = "d_Scale" if out_idx == 1 else "d_Bias"
                            print(f"    LayerNormalizationGrad {label} shape: {scale_shape}")

    elif op in ("SplitTraining", "com.microsoft.SplitTraining"):
        # SplitTraining: backward of ConcatTraining.
        # Splits input[0] along axis into N chunks, where N is determined by
        # input[1] = per_input_length tensor (produced by ConcatTraining).
        # We recover the output shapes by finding the corresponding ConcatTraining
        # node whose second output produced the per_input_length tensor.
        if len(node.input) >= 2 and len(node.output) >= 1:
            sizes_name = node.input[1]
            # Find the ConcatTraining node that produced this sizes tensor.
            for prod_node in graph.node:
                if len(prod_node.output) >= 2 and prod_node.output[1] == sizes_name:
                    # Output shapes of SplitTraining correspond to the input shapes
                    # of the ConcatTraining (same position).
                    for out_idx, out_name in enumerate(node.output):
                        if not out_name:
                            continue
                        existing = get_tensor_shape(graph, out_name)
                        if existing:
                            continue
                        if out_idx < len(prod_node.input):
                            src_shape = get_tensor_shape(graph, prod_node.input[out_idx])
                            if src_shape:
                                set_tensor_shape(graph, out_name, src_shape)
                                print(f"    SplitTraining output[{out_idx}] shape: {src_shape}")
                    break

    elif op in ("BatchNormInternal", "com.microsoft.BatchNormInternal"):
        # BatchNormInternal (ORT training-mode BN forward):
        #   inputs:  X[N,C,H,W], gamma[C], beta[C], running_mean[C], running_var[C]
        #   outputs: Y[N,C,H,W], updated_running_mean[C], updated_running_var[C],
        #            saved_mean[C], saved_inv_std[C]
        if len(node.input) >= 1 and len(node.output) >= 1:
            x_shape = get_tensor_shape(graph, node.input[0])
            if x_shape and len(x_shape) >= 2:
                C = x_shape[1]
                # outputs[0] = Y: same shape as X
                out_y = node.output[0] if len(node.output) > 0 else ""
                if out_y and not get_tensor_shape(graph, out_y):
                    set_tensor_shape(graph, out_y, list(x_shape))
                    print(f"    BatchNormInternal Y shape: {list(x_shape)}")
                # outputs[1,2] = updated_running_mean/var: [C]
                for out_idx in [1, 2]:
                    out_name = node.output[out_idx] if out_idx < len(node.output) else ""
                    if out_name and not get_tensor_shape(graph, out_name):
                        set_tensor_shape(graph, out_name, [C])
                        label = "updated_running_mean" if out_idx == 1 else "updated_running_var"
                        print(f"    BatchNormInternal {label} shape: [{C}]")
                # outputs[3,4] = saved_mean, saved_inv_std: [C]
                for out_idx in [3, 4]:
                    out_name = node.output[out_idx] if out_idx < len(node.output) else ""
                    if out_name and not get_tensor_shape(graph, out_name):
                        set_tensor_shape(graph, out_name, [C])
                        label = "saved_mean" if out_idx == 3 else "saved_inv_std"
                        print(f"    BatchNormInternal {label} shape: [{C}]")

    elif op in ("BatchNormalizationGrad", "com.microsoft.BatchNormalizationGrad"):
        # BatchNormalizationGrad (ORT BN backward):
        #   inputs:  dY[N,C,H,W], X[N,C,H,W], gamma[C], saved_mean[C], saved_inv_std[C]
        #   outputs: dX[N,C,H,W], dgamma[C], dbeta[C]
        if len(node.input) >= 1 and len(node.output) >= 1:
            dy_shape = get_tensor_shape(graph, node.input[0])
            if dy_shape and len(dy_shape) >= 2:
                C = dy_shape[1]
                # output[0] = dX: same shape as dY
                out_dx = node.output[0] if len(node.output) > 0 else ""
                if out_dx and not get_tensor_shape(graph, out_dx):
                    set_tensor_shape(graph, out_dx, list(dy_shape))
                    print(f"    BatchNormalizationGrad dX shape: {list(dy_shape)}")
                # outputs[1,2] = dgamma, dbeta: [C]
                for out_idx in [1, 2]:
                    out_name = node.output[out_idx] if out_idx < len(node.output) else ""
                    if out_name and not get_tensor_shape(graph, out_name):
                        set_tensor_shape(graph, out_name, [C])
                        label = "dgamma" if out_idx == 1 else "dbeta"
                        print(f"    BatchNormalizationGrad {label} shape: [{C}]")

    elif op in ("Scale", "com.microsoft.Scale"):
        # com.microsoft.Scale: element-wise scale (output = input * scalar).
        # Output shape equals input[0] shape.
        # NOTE: input[0] may be a scalar constant with dims=[], so use `is not None`
        # rather than a truthiness check ([] is falsy but valid).
        if len(node.input) >= 1 and len(node.output) >= 1:
            out_name = node.output[0]
            if out_name and get_tensor_shape(graph, out_name) is None:
                input_shape = get_tensor_shape(graph, node.input[0])
                if input_shape is not None:
                    set_tensor_shape(graph, out_name, input_shape)
                    print(f"    Scale output shape: {input_shape}")

    elif op == "GlobalAveragePoolGrad":
        # Fused GAP backward: dY[N,C,1,1] → dX[N,C,H,W]
        # kernel_shape=[H,W] attribute carries the spatial size.
        if len(node.input) >= 1 and len(node.output) >= 1:
            out_name = node.output[0]
            if out_name and not get_tensor_shape(graph, out_name):
                dy_shape = get_tensor_shape(graph, node.input[0])
                H, W = None, None
                for attr in node.attribute:
                    if attr.name == "kernel_shape" and len(attr.ints) == 2:
                        H, W = int(attr.ints[0]), int(attr.ints[1])
                if dy_shape and len(dy_shape) >= 2 and H is not None and W is not None:
                    N, C = int(dy_shape[0]), int(dy_shape[1])
                    dx_shape = [N, C, H, W]
                    set_tensor_shape(graph, out_name, dx_shape)
                    print(f"    GlobalAveragePoolGrad dX shape: {dx_shape}")

    elif node.domain == "com.microsoft" and "Grad" in op:
        # Generic handling for gradient ops (domain-safe check)
        if len(node.input) >= 2 and len(node.output) >= 1:
            input_shape = get_tensor_shape(graph, node.input[1])
            if input_shape:
                set_tensor_shape(graph, node.output[0], input_shape)
                print(f"    {op} output shape: {input_shape}")


def apply_gemm_shape_inference(graph: onnx.GraphProto) -> None:
    """
    Manually infer output shapes for Gemm nodes whose outputs are still unknown.

    Standard ONNX shape_inference cannot propagate past com.microsoft ops such as
    SoftmaxCrossEntropyLossGrad.  Any standard Gemm node that is downstream of such
    a node and whose output shape has not been filled in by ONNX is handled here:
    the output shape is computed directly from transA / transB and the input shapes.

    Args:
        graph: ONNX graph to update in-place
    """
    for node in graph.node:
        if node.op_type != "Gemm" or node.domain != "":
            continue
        if not node.output or not node.output[0]:
            continue
        output_name = node.output[0]
        # Skip if shape is already known
        existing = get_tensor_shape(graph, output_name)
        if existing and any(d > 0 for d in existing):
            continue
        if len(node.input) < 2:
            continue
        A_shape = get_tensor_shape(graph, node.input[0])
        B_shape = get_tensor_shape(graph, node.input[1])
        if not A_shape or not B_shape or len(A_shape) < 2 or len(B_shape) < 2:
            continue
        transA = 0
        transB = 0
        for attr in node.attribute:
            if attr.name == "transA":
                transA = attr.i
            elif attr.name == "transB":
                transB = attr.i
        if len(A_shape) > 2:
            # Batched Gemm: batch_dims + [M, N]
            batch_dims = list(A_shape[:-2])
            M = A_shape[-1] if transA else A_shape[-2]
            N = B_shape[-2] if transB else B_shape[-1]
            output_shape = batch_dims + [M, N]
        else:
            M = A_shape[1] if transA else A_shape[0]
            N = B_shape[0] if transB else B_shape[1]
            output_shape = [M, N]
        set_tensor_shape(graph, output_name, output_shape)
        print(f"    Gemm {node.name}: inferred output shape {output_shape}")


def propagate_residual_shapes(graph: onnx.GraphProto) -> None:
    """
    Iteratively propagate shapes for standard ops (ReduceSum, Reshape) whose
    inputs were given shapes by the custom inference handlers but whose outputs
    were still unknown because ONNX standard shape inference ran before the
    custom handlers.

    This pass repeats until no more shapes can be resolved.

    Args:
        graph: ONNX graph to update in-place
    """
    import numpy as np

    # Build initializer lookup: name → int64 data (for Reshape shape inputs)
    init_int64: dict = {}
    for init in graph.initializer:
        if init.data_type == onnx.TensorProto.INT64:
            try:
                data = np.frombuffer(init.raw_data, dtype=np.int64).tolist()
                init_int64[init.name] = data
            except Exception:
                pass

    # Build producer map: tensor_name → node that produces it
    producers: dict = {}
    for node in graph.node:
        for out_name in node.output:
            if out_name:
                producers[out_name] = node

    changed = True
    while changed:
        changed = False
        for node in graph.node:
            for out_idx, out_name in enumerate(node.output):
                if not out_name:
                    continue
                existing = get_tensor_shape(graph, out_name)
                if existing:
                    continue  # already has shape (non-empty list)

                if node.op_type == "ReduceSum" and len(node.input) >= 1:
                    input_shape = get_tensor_shape(graph, node.input[0])
                    if not input_shape:
                        continue
                    axes = []
                    keepdims = 1
                    for attr in node.attribute:
                        if attr.name == "axes":
                            axes = list(attr.ints)
                        elif attr.name == "keepdims":
                            keepdims = attr.i
                    ndim = len(input_shape)
                    norm_axes = {a % ndim for a in axes}
                    out_shape = []
                    for i, s in enumerate(input_shape):
                        if i in norm_axes:
                            if keepdims:
                                out_shape.append(1)
                        else:
                            out_shape.append(s)
                    if out_shape or not axes:
                        if not out_shape and not axes:
                            # No axes → reduce over everything
                            out_shape = [1] * ndim if keepdims else []
                        set_tensor_shape(graph, out_name, out_shape if out_shape else [1])
                        print(f"    propagate ReduceSum {node.name}: {out_name} → {out_shape}")
                        changed = True

                elif node.op_type == "Reshape" and len(node.input) >= 2:
                    shape_input = node.input[1]
                    target_shape = None
                    if shape_input in init_int64:
                        # Shape stored as initializer constant
                        target_shape = init_int64[shape_input]
                    else:
                        # Shape input produced by a Shape node:
                        # Shape(X) → values = shape of X = target_shape for Reshape
                        shape_producer = producers.get(shape_input)
                        if shape_producer and shape_producer.op_type == "Shape":
                            src = shape_producer.input[0] if shape_producer.input else None
                            if src:
                                src_shape = get_tensor_shape(graph, src)
                                if src_shape:
                                    target_shape = src_shape  # values = shape of src
                    if target_shape is not None:
                        set_tensor_shape(graph, out_name, target_shape)
                        print(f"    propagate Reshape {node.name}: {out_name} → {target_shape}")
                        changed = True

                elif node.op_type == "Gemm" and node.domain == "" and out_idx == 0:
                    # Fallback for Gemm nodes not handled by apply_gemm_shape_inference.
                    # Handles 2D and batched (3D+) cases.
                    if len(node.input) < 2:
                        continue
                    A_shape = get_tensor_shape(graph, node.input[0])
                    B_shape = get_tensor_shape(graph, node.input[1])
                    if not A_shape or not B_shape:
                        continue
                    if len(A_shape) < 2 or len(B_shape) < 2:
                        continue
                    transA = 0
                    transB = 0
                    for attr in node.attribute:
                        if attr.name == "transA":
                            transA = attr.i
                        elif attr.name == "transB":
                            transB = attr.i
                    if len(A_shape) > 2:
                        batch_dims = list(A_shape[:-2])
                        M = A_shape[-1] if transA else A_shape[-2]
                        N = B_shape[-2] if transB else B_shape[-1]
                        output_shape = batch_dims + [M, N]
                    else:
                        M = A_shape[1] if transA else A_shape[0]
                        N = B_shape[0] if transB else B_shape[1]
                        output_shape = [M, N]
                    set_tensor_shape(graph, out_name, output_shape)
                    print(f"    propagate Gemm {node.name}: {out_name} → {output_shape}")
                    changed = True


def extract_subgraph(model: onnx.ModelProto, nodes: List[onnx.NodeProto]) -> onnx.ModelProto:
    """
    Extract a subgraph containing specified nodes.

    Args:
        model: Original ONNX model
        nodes: List of nodes to include in subgraph

    Returns:
        Model containing only the specified nodes
    """
    subgraph = onnx.ModelProto()
    subgraph.CopyFrom(model)

    # Replace nodes with specified subset
    del subgraph.graph.node[:]
    subgraph.graph.node.extend(nodes)

    return subgraph


def update_model_with_inferred_shapes(
    model: onnx.ModelProto, inferred_subgraph: onnx.ModelProto, node_index: int
) -> None:
    """
    Update model with shape information from inferred subgraph.

    Args:
        model: Model to update
        inferred_subgraph: Subgraph with inferred shapes
        node_index: Index of node being updated
    """
    if node_index < len(model.graph.node):
        node = model.graph.node[node_index]
        inferred_node = inferred_subgraph.graph.node[0]

        # Update output shapes
        for i, output_name in enumerate(node.output):
            if i < len(inferred_node.output):
                value_info = get_value_info_by_name(
                    inferred_subgraph.graph, inferred_node.output[i]
                )
                update_value_info_shape(model.graph, output_name, value_info)


def get_value_info_by_name(graph: onnx.GraphProto, name: str) -> Optional[onnx.ValueInfoProto]:
    """
    Get ValueInfoProto by tensor name.

    Args:
        graph: ONNX graph
        name: Tensor name to search for

    Returns:
        ValueInfoProto if found, None otherwise
    """
    # Check outputs
    for info in graph.output:
        if info.name == name:
            return info

    # Check value_info
    for info in graph.value_info:
        if info.name == name:
            return info

    # Check inputs
    for info in graph.input:
        if info.name == name:
            return info

    return None


def update_value_info_shape(
    graph: onnx.GraphProto, name: str, value_info: Optional[onnx.ValueInfoProto]
) -> None:
    """
    Update shape information for a tensor in the graph.

    Args:
        graph: ONNX graph
        name: Tensor name to update
        value_info: ValueInfoProto with shape information
    """
    if not value_info or not value_info.type.tensor_type.shape:
        return

    # Find existing info
    existing_info = get_value_info_by_name(graph, name)
    if existing_info:
        existing_info.type.tensor_type.shape.CopyFrom(value_info.type.tensor_type.shape)
    else:
        # Create new value_info
        new_info = onnx.ValueInfoProto()
        new_info.name = name
        new_info.type.tensor_type.shape.CopyFrom(value_info.type.tensor_type.shape)
        graph.value_info.append(new_info)


def set_tensor_shape(graph: onnx.GraphProto, tensor_name: str, shape: List[int]) -> None:
    """
    Set shape for a tensor in the graph.

    Args:
        graph: ONNX graph
        tensor_name: Name of tensor to set shape for
        shape: Shape dimensions
    """
    # Build a fresh, fully-typed ValueInfoProto using the ONNX helper so that
    # the tensor_type oneof field is properly activated.  Then either replace
    # the existing entry or append a new one.
    new_vi = onnx.helper.make_tensor_value_info(
        tensor_name,
        onnx.TensorProto.FLOAT,  # default to float; shape is what matters here
        shape,
    )

    # Try to replace an existing entry in value_info
    for i, vi in enumerate(graph.value_info):
        if vi.name == tensor_name:
            graph.value_info[i].CopyFrom(new_vi)
            return

    # Not found — append a new entry
    graph.value_info.append(new_vi)


def get_tensor_shape(graph: onnx.GraphProto, tensor_name: str) -> Optional[List[int]]:
    """
    Get shape of a tensor from the graph.

    Args:
        graph: ONNX graph
        tensor_name: Name of tensor to get shape for

    Returns:
        List of dimension sizes, or None if not found
    """
    # Check initializers
    for initializer in graph.initializer:
        if initializer.name == tensor_name:
            return list(initializer.dims)

    # Check inputs
    for input_tensor in graph.input:
        if input_tensor.name == tensor_name:
            return [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]

    # Check outputs
    for output_tensor in graph.output:
        if output_tensor.name == tensor_name:
            return [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]

    # Check value_info
    for value_info in graph.value_info:
        if value_info.name == tensor_name:
            return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]

    return None
