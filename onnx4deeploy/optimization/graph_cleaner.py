# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Graph cleaning utilities for ONNX models.

This module contains functions for cleaning and optimizing ONNX computation graphs,
including removing identity nodes, handling softmax outputs, and BiasGelu optimizations.
"""

import copy

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


def remove_identity_nodes(input_onnx: str, output_onnx: str):
    """
    Remove Identity nodes from the ONNX graph.
    Identity nodes just pass through their input without modification.
    """
    model = onnx.load(input_onnx)
    graph = model.graph

    # Track nodes to remove
    nodes_to_remove = []

    # Build a mapping of tensor names to their replacement
    tensor_replacements = {}

    for node in graph.node:
        if node.op_type == "Identity":
            # Identity node has 1 input and 1 output
            if len(node.input) == 1 and len(node.output) == 1:
                input_name = node.input[0]
                output_name = node.output[0]

                # Map output to input (bypass the Identity node)
                tensor_replacements[output_name] = input_name
                nodes_to_remove.append(node)
                print(f"Removing Identity node '{node.name}': {output_name} -> {input_name}")

    # Remove Identity nodes
    for node in nodes_to_remove:
        graph.node.remove(node)

    # Resolve chains: if A→B and B→C, update A→C so chained identities are fully removed.
    for key in list(tensor_replacements.keys()):
        val = tensor_replacements[key]
        while val in tensor_replacements:
            val = tensor_replacements[val]
        tensor_replacements[key] = val

    # Update all references to removed Identity outputs
    for node in graph.node:
        for i, input_name in enumerate(node.input):
            if input_name in tensor_replacements:
                node.input[i] = tensor_replacements[input_name]
                print(
                    f"  Updated node '{node.name}' input: {input_name} -> {tensor_replacements[input_name]}"
                )

    # Update graph outputs if they reference Identity outputs
    for output in graph.output:
        if output.name in tensor_replacements:
            old_name = output.name
            output.name = tensor_replacements[old_name]
            print(f"  Updated graph output: {old_name} -> {output.name}")

    # Save the modified model
    onnx.save(model, output_onnx)

    print(f"✅ Removed {len(nodes_to_remove)} Identity node(s)")

    return model


def remove_softmax_loss_outputs(input_model_path: str, output_model_path: str):
    """
    Remove loss outputs from SoftmaxCrossEntropyLoss nodes, keeping only the log probability output.

    Args:
        input_model_path (str): Path to the input ONNX model
        output_model_path (str): Path to save the modified ONNX model
    """
    import onnx

    # Load the model
    model = onnx.load(input_model_path)
    graph = model.graph

    # Find SoftmaxCrossEntropyLoss nodes
    target_nodes = []
    for node in graph.node:
        if node.op_type == "SoftmaxCrossEntropyLoss":
            target_nodes.append(node)

    print(f"Found {len(target_nodes)} SoftmaxCrossEntropyLoss nodes")

    # Outputs to remove (first output - loss)
    outputs_to_remove = []

    # Create new nodes with modified outputs
    new_nodes = []
    for node in graph.node:
        if node.op_type == "SoftmaxCrossEntropyLoss" and len(node.output) > 1:
            # Keep only the second output (log probabilities) and remove the first (loss)
            outputs_to_remove.append(node.output[0])

            # Create a new node with only the second output
            new_node = onnx.NodeProto()
            new_node.CopyFrom(node)
            log_prob_output = node.output[1]

            # Clear outputs and set only the log probability output
            del new_node.output[:]
            new_node.output.append(log_prob_output)

            new_nodes.append(new_node)
        else:
            # Keep other nodes unchanged
            new_nodes.append(node)

    # Replace all nodes with the new set
    del graph.node[:]
    graph.node.extend(new_nodes)

    # Filter graph outputs to remove loss outputs
    new_outputs = []
    for output in graph.output:
        if output.name not in outputs_to_remove:
            new_outputs.append(output)

    # Replace graph outputs with filtered list
    del graph.output[:]
    graph.output.extend(new_outputs)

    # Save the modified model
    onnx.save(model, output_model_path)
    print(f"Saved model with loss outputs removed to: {output_model_path}")


def modify_conflict_outputs(input_model_path: str, output_model_path: str):
    model = onnx.load(input_model_path)
    graph = model.graph

    select_nodes = []
    for node in graph.node:
        if node.op_type == "LayerNormalization" or node.op_type == "MaxPool":
            # if node.op_type == 'MaxPool':
            select_nodes.append(node)

    print(f"Find {len(select_nodes)} Maxpool")

    outputs_to_remove = []

    new_nodes = []

    for node in graph.node:
        if (node.op_type == "LayerNormalization" or node.op_type == "MaxPool") and len(
            node.output
        ) > 1:
            # if (node.op_type == 'MaxPool') and len(node.output) > 1:
            outputs_to_remove.extend(node.output[1:])

            new_node = onnx.NodeProto()
            new_node.CopyFrom(node)
            first_output = node.output[0]

            del new_node.output[:]
            new_node.output.append(first_output)

            new_nodes.append(new_node)
        else:
            new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)

    new_outputs = []
    for output in graph.output:
        if output.name not in outputs_to_remove:
            new_outputs.append(output)

    del graph.output[:]
    graph.output.extend(new_outputs)

    onnx.save(model, output_model_path)
    print(f"Saved to: {output_model_path}")


def fix_layernorm_output(input_model_path: str, output_model_path: str) -> bool:
    """
    Fix output types and shapes for all LayerNorm operators in an ONNX model.

    Args:
        input_model_path (str): Path to the input model file
        output_model_path (str): Path to save the output model file

    Returns:
        bool: True if the operation succeeded, False otherwise
    """
    try:
        # Load the model
        model = onnx.load(input_model_path)
        graph = model.graph

        # Find all LayerNorm nodes
        layernorm_count = 0
        updated_count = 0
        tensor_info = {}

        # Collect tensor information
        # Process input tensors
        for input_tensor in graph.input:
            name = input_tensor.name
            shape = [
                dim.dim_value if dim.dim_value > 0 else None
                for dim in input_tensor.type.tensor_type.shape.dim
            ]
            elem_type = input_tensor.type.tensor_type.elem_type
            tensor_info[name] = {"shape": shape, "elem_type": elem_type}

        # Process intermediate and output tensors
        for value_info in list(graph.value_info) + list(graph.output):
            name = value_info.name
            shape = [
                dim.dim_value if dim.dim_value > 0 else None
                for dim in value_info.type.tensor_type.shape.dim
            ]
            elem_type = value_info.type.tensor_type.elem_type
            tensor_info[name] = {"shape": shape, "elem_type": elem_type}

        # Fix each LayerNorm node
        for node in graph.node:
            if node.op_type == "LayerNormalization":
                layernorm_count += 1

                if not node.input:
                    continue

                # Get input information
                input_name = node.input[0]
                if input_name not in tensor_info:
                    continue

                input_info = tensor_info[input_name]
                input_shape = input_info["shape"]
                input_elem_type = input_info["elem_type"]

                # Get axis attribute
                axis = -1
                for attr in node.attribute:
                    if attr.name == "axis":
                        axis = attr.i
                        break

                # Process all outputs
                for i, output_name in enumerate(node.output):
                    # Determine correct output shape and type
                    output_shape = None
                    output_elem_type = input_elem_type

                    if i == 0:  # Main output - same shape as input
                        output_shape = input_shape
                    else:  # mean and std outputs - shape depends on normalization axis
                        # Handle negative axis index
                        if axis < 0 and input_shape and None not in input_shape:
                            axis = len(input_shape) + axis

                        # Create shape for mean and std (remove normalization axis)
                        if input_shape and None not in input_shape and 0 <= axis < len(input_shape):
                            output_shape = list(input_shape)
                            output_shape.pop(axis)  # Remove normalization axis

                    # Find and remove existing value info
                    for value_info in list(graph.value_info):
                        if value_info.name == output_name:
                            graph.value_info.remove(value_info)
                            break

                    # Create new value info
                    if output_shape and None not in output_shape:
                        new_value_info = onnx.helper.make_tensor_value_info(
                            output_name, output_elem_type, output_shape
                        )
                        graph.value_info.append(new_value_info)

                    # Update graph output if needed
                    for j, output in enumerate(list(graph.output)):
                        if output.name == output_name:
                            if output_shape and None not in output_shape:
                                new_output = onnx.helper.make_tensor_value_info(
                                    output_name, output_elem_type, output_shape
                                )
                                graph.output.remove(output)
                                graph.output.insert(j, new_output)
                            break

                    # Update tensor info dictionary
                    tensor_info[output_name] = {
                        "shape": output_shape,
                        "elem_type": output_elem_type,
                    }

                    print(f"  Output {i}: {output_name}, shape={output_shape}")  # Debug info

                updated_count += 1

        # Save the model
        onnx.save(model, output_model_path)
        print(
            f"Updated {updated_count}/{layernorm_count} LayerNorm nodes, model saved to {output_model_path}"
        )
        return True

    except Exception as e:
        print(f"Error fixing LayerNorm outputs: {str(e)}")
        return False


def replace_biasgelu_with_gelu_add(input_model_path: str, output_model_path: str):

    model = onnx.load(input_model_path)

    # Collect all value_info entries by name for easy lookup
    value_info_map = {}
    for vi in model.graph.value_info:
        value_info_map[vi.name] = vi

    # Add input and output value_info to the map
    for inp in model.graph.input:
        value_info_map[inp.name] = inp

    for out in model.graph.output:
        value_info_map[out.name] = out

    # Create new node list and value_info list
    new_nodes = []
    new_value_info = []
    biasgelu_count = 0

    # Counter for generating unique names
    unique_id = 0

    def get_unique_name(prefix):
        nonlocal unique_id
        name = f"{prefix}_{unique_id}"
        unique_id += 1
        return name

    # Process all nodes
    for node in model.graph.node:
        if node.op_type == "BiasGelu":
            biasgelu_count += 1

            # Get BiasGelu inputs and outputs
            input_name = node.input[0]  # X
            bias_name = node.input[1]  # Bias
            output_name = node.output[0]  # Y

            # Generate unique name prefix
            prefix = node.name if node.name else f"gelu_add"

            # Step 1: First apply Add operation to add bias
            add_output = get_unique_name(f"{prefix}_add_out")
            add_node = helper.make_node(
                "Add", inputs=[input_name, bias_name], outputs=[add_output], name=f"{prefix}_add"
            )
            new_nodes.append(add_node)

            # Create value_info for add_output with proper type and shape
            # Use the same type and shape as the input tensor if available
            if input_name in value_info_map:
                input_value_info = value_info_map[input_name]
                add_output_value_info = helper.make_tensor_value_info(
                    add_output,
                    input_value_info.type.tensor_type.elem_type,
                    [
                        d.dim_value if d.dim_value else d.dim_param
                        for d in input_value_info.type.tensor_type.shape.dim
                    ],
                )
                new_value_info.append(add_output_value_info)
                value_info_map[add_output] = add_output_value_info

            # Step 2: Then apply Gelu activation function
            gelu_node = helper.make_node(
                "Gelu", inputs=[add_output], outputs=[output_name], name=f"{prefix}_gelu"
            )
            new_nodes.append(gelu_node)

            # If we have output value_info, make sure it's preserved
            # Otherwise, create it with the same shape and type as the input to Gelu
            if output_name not in value_info_map and add_output in value_info_map:
                output_value_info = helper.make_tensor_value_info(
                    output_name,
                    value_info_map[add_output].type.tensor_type.elem_type,
                    [
                        d.dim_value if d.dim_value else d.dim_param
                        for d in value_info_map[add_output].type.tensor_type.shape.dim
                    ],
                )
                new_value_info.append(output_value_info)
                value_info_map[output_name] = output_value_info
        else:
            # Keep other nodes unchanged
            new_nodes.append(node)

    print(f"Replaced {biasgelu_count} BiasGelu nodes with Gelu+Add combinations")

    # Create new graph with all collected value_info
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=model.graph.initializer,
        value_info=list(model.graph.value_info) + new_value_info,
    )

    # Build new model, preserving original model metadata
    new_model = helper.make_model(
        new_graph,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        domain=model.domain,
        model_version=model.model_version,
        doc_string=model.doc_string,
    )

    # Copy opset imports
    del new_model.opset_import[:]
    new_model.opset_import.extend(model.opset_import)

    # Add Microsoft domain if not present (for Gelu)
    has_ms_domain = any(opset.domain == "com.microsoft" for opset in new_model.opset_import)
    if not has_ms_domain:
        ms_opset = helper.make_opsetid("com.microsoft", 1)
        new_model.opset_import.append(ms_opset)

    # Copy IR version
    new_model.ir_version = model.ir_version

    # Run shape inference to ensure all shapes are properly defined
    try:
        new_model = shape_inference.infer_shapes(new_model)
        print("Shape inference successful")
    except Exception as e:
        print(f"Warning: Shape inference failed: {e}")

    # Skip validation and directly save if needed
    try:
        onnx.checker.check_model(new_model)
        print("Model validation successful!")
    except Exception as e:
        print(f"Warning: Model validation failed, but still saving: {e}")

    # Save the modified model
    onnx.save(new_model, output_model_path)
    print(f"Saved modified model to {output_model_path}")

    return new_model


def run_optmization_remove_biasgelu(onnx_train_file: str, onnx_out_file: str):
    """
    Replace BiasGelu operations with Add+Gelu while maintaining shape consistency.

    Args:
        onnx_train_file: Path to input ONNX model file
        onnx_out_file: Path to output ONNX model file
    """
    # Load the model
    model = onnx.load(onnx_train_file)
    graph = model.graph

    # Create new nodes list to replace the old ones
    new_nodes = []
    replaced_count = 0

    # Keep track of used initializers
    used_initializers = set()
    new_initializers = []

    # Process all nodes
    for node in graph.node:
        if node.op_type == "BiasGelu":
            print(f"🔄 Replacing BiasGeluFusion: {node.name}")
            replaced_count += 1

            # Get input and output tensors
            X, Bias = node.input
            output = node.output[0]

            # Create intermediate tensor name
            intermediate_output = f"{X}_add_bias"

            # Create a new unique bias name to avoid sharing
            new_bias_name = f"{Bias}_{node.name}"

            # Find the original bias initializer
            bias_initializer = None
            for initializer in graph.initializer:
                if initializer.name == Bias:
                    bias_initializer = initializer
                    break

            # Create a copy of the bias initializer with a new name if found
            if bias_initializer is not None:
                new_bias = onnx.helper.make_tensor(
                    name=new_bias_name,
                    data_type=bias_initializer.data_type,
                    dims=bias_initializer.dims,
                    vals=(
                        bias_initializer.raw_data
                        if bias_initializer.raw_data
                        else bias_initializer.float_data
                        or bias_initializer.int32_data
                        or bias_initializer.int64_data
                        or bias_initializer.uint64_data
                    ),
                    raw=bool(bias_initializer.raw_data),
                )
                new_initializers.append(new_bias)
                used_initializers.add(Bias)
            else:
                # If we can't find the initializer, it might be an input or value_info
                # In this case, we should keep the original bias name
                new_bias_name = Bias

            # Create Add node with the new bias
            add_node = helper.make_node(
                "Add",
                inputs=[X, new_bias_name],
                outputs=[intermediate_output],
                name=f"{node.name}_Add",
            )

            # Create Gelu node
            gelu_node = helper.make_node(
                "Gelu",
                inputs=[intermediate_output],
                outputs=node.output,
                name=f"{node.name}_Gelu",
            )

            # Add shape information for the intermediate tensor
            # Try to find X's shape info
            X_shape = None
            X_type = 1  # Default to FLOAT

            # Look for X in inputs, outputs, value_info, or initializers
            for info in graph.input:
                if info.name == X:
                    X_shape = [
                        d.dim_value if d.HasField("dim_value") else -1
                        for d in info.type.tensor_type.shape.dim
                    ]
                    X_type = info.type.tensor_type.elem_type
                    break

            if X_shape is None:
                for info in graph.value_info:
                    if info.name == X:
                        X_shape = [
                            d.dim_value if d.HasField("dim_value") else -1
                            for d in info.type.tensor_type.shape.dim
                        ]
                        X_type = info.type.tensor_type.elem_type
                        break

            # Add value_info for intermediate tensor
            if X_shape:
                value_info = helper.make_tensor_value_info(intermediate_output, X_type, X_shape)
                graph.value_info.append(value_info)

            # Add the new nodes to our list
            new_nodes.extend([add_node, gelu_node])
        else:
            # Keep other nodes unchanged
            new_nodes.append(node)

    # Add the new initializers to the graph
    for initializer in new_initializers:
        graph.initializer.append(initializer)

    # Replace nodes in the graph
    graph.ClearField("node")
    graph.node.extend(new_nodes)

    # Rest of the function remains the same...
    # Create a copy of the model for selective shape inference
    safe_model = copy.deepcopy(model)

    # Remove Microsoft custom operators that might cause shape inference to fail
    ms_nodes = []
    for node in safe_model.graph.node:
        if node.domain == "com.microsoft":
            ms_nodes.append(node)

    if ms_nodes:
        print(
            f"⚠️ Found {len(ms_nodes)} Microsoft custom operators that might affect shape inference"
        )

        # Try to run shape inference on the modified model without MS operators
        try:
            # Create a temporary graph without Microsoft operators
            temp_model = copy.deepcopy(safe_model)
            temp_graph = temp_model.graph

            # Remove Microsoft custom operators
            temp_nodes = [node for node in temp_graph.node if node.domain != "com.microsoft"]
            temp_graph.ClearField("node")
            temp_graph.node.extend(temp_nodes)

            # Run shape inference on this simplified model
            inferred_model = shape_inference.infer_shapes(temp_model)

            # Collect inferred shapes for our new nodes
            inferred_value_infos = {}
            for value_info in inferred_model.graph.value_info:
                inferred_value_infos[value_info.name] = value_info

            # Update the original model with any newly inferred shapes
            for name, value_info in inferred_value_infos.items():
                # Skip if already exists
                if any(info.name == name for info in model.graph.value_info):
                    continue

                model.graph.value_info.append(value_info)

            print("✅ Partial shape inference completed for non-Microsoft operators")
        except Exception as e:
            print(f"⚠️ Partial shape inference failed: {e}")
    else:
        # No Microsoft operators, try regular shape inference
        try:
            model = shape_inference.infer_shapes(model)
            print("✅ Shape inference completed successfully")
        except Exception as e:
            print(f"⚠️ Shape inference failed: {e}")

    # Save the modified model
    onnx.save(model, onnx_out_file)

    if replaced_count > 0:
        print(f"✅ Successfully replaced {replaced_count} BiasGelu nodes with Add + GELU.")
    else:
        print("⚠️ No BiasGelu nodes were replaced.")

    return model


def optimize_softmax_axis(input_model_path: str, output_model_path: str):

    model = onnx.load(input_model_path)

    # Track if we made any changes
    optimized = False

    # Create a map of value_info by name for easy access
    value_info_map = {vi.name: vi for vi in model.graph.value_info}
    value_info_map.update({vi.name: vi for vi in model.graph.input})
    value_info_map.update({vi.name: vi for vi in model.graph.output})

    # Function to get shape from value_info
    def get_shape(tensor_name):
        if tensor_name in value_info_map:
            shape = []
            for dim in value_info_map[tensor_name].type.tensor_type.shape.dim:
                if dim.dim_param:
                    # Handle symbolic dimensions (set to -1 for dynamic dimension)
                    shape.append(-1)
                else:
                    shape.append(dim.dim_value)
            return shape
        return None

    # Track the names of nodes to be removed
    nodes_to_remove = []

    # Track new nodes and value_infos to be added
    new_nodes = []
    new_value_infos = []

    # For each node in the graph
    for i, node in enumerate(model.graph.node):
        if node.op_type == "Softmax":
            # Get the input and output names
            input_name = node.input[0]
            output_name = node.output[0]

            # Get the axis attribute
            axis = None
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = attr.i
                    break

            # If axis is not set, it defaults to 1 in ONNX
            if axis is None:
                axis = 1

            # Get the input shape
            input_shape = get_shape(input_name)
            if input_shape is None:
                print(f"Warning: Could not determine shape for {input_name}, skipping optimization")
                continue

            # Check if all dimensions after axis are 1
            all_ones_after_axis = (
                all(dim == 1 for dim in input_shape[axis + 1 :])
                if axis + 1 < len(input_shape)
                else True
            )

            # Only optimize if the axis is not the last dimension and all subsequent dimensions are 1
            if axis != len(input_shape) - 1 and all_ones_after_axis and axis >= 0:
                print(f"Optimizing Softmax node with input shape {input_shape} and axis={axis}")

                # Create unique names for intermediate tensors
                reshape_before_output = f"{input_name}_reshaped_before_softmax"
                softmax_output = f"{output_name}_after_softmax"

                # Calculate new shapes
                # Move the axis dimension to the end and flatten all the 1s
                new_shape_before = []
                for i in range(len(input_shape)):
                    if i < axis:
                        new_shape_before.append(input_shape[i])
                    elif i == axis:
                        continue
                    elif i > axis:
                        continue
                new_shape_before.append(input_shape[axis])

                # Create reshape node before softmax
                reshape_before_node = helper.make_node(
                    "Reshape",
                    inputs=[input_name, f"{input_name}_shape_before"],
                    outputs=[reshape_before_output],
                    name=f"Reshape_before_softmax_{output_name}",
                )

                # Create initializer for the shape tensor
                shape_tensor_before = numpy_helper.from_array(
                    np.array(new_shape_before, dtype=np.int64), name=f"{input_name}_shape_before"
                )

                # Create new softmax node with axis set to -1 (last dimension)
                new_softmax_node = helper.make_node(
                    "Softmax",
                    inputs=[reshape_before_output],
                    outputs=[softmax_output],
                    name=f"Softmax_optimized_{output_name}",
                    axis=-1,  # Use -1 to always target the last dimension
                )

                # Create reshape node after softmax to restore original shape
                reshape_after_node = helper.make_node(
                    "Reshape",
                    inputs=[softmax_output, f"{output_name}_shape_after"],
                    outputs=[output_name],
                    name=f"Reshape_after_softmax_{output_name}",
                )

                # Create initializer for the shape tensor
                shape_tensor_after = numpy_helper.from_array(
                    np.array(input_shape, dtype=np.int64), name=f"{output_name}_shape_after"
                )

                # Create value info for reshape_before_output
                reshape_before_vi = helper.make_tensor_value_info(
                    reshape_before_output,
                    value_info_map[input_name].type.tensor_type.elem_type,
                    new_shape_before,
                )

                # Create value info for softmax_output
                softmax_output_vi = helper.make_tensor_value_info(
                    softmax_output,
                    value_info_map[output_name].type.tensor_type.elem_type,
                    new_shape_before,  # Shape doesn't change after softmax
                )

                # Add all new nodes and value infos
                new_nodes.extend([reshape_before_node, new_softmax_node, reshape_after_node])
                new_value_infos.extend([reshape_before_vi, softmax_output_vi])
                model.graph.initializer.extend([shape_tensor_before, shape_tensor_after])

                # Mark the original node for removal
                nodes_to_remove.append(node)
                optimized = True

    # Remove the original nodes that were optimized
    for node in nodes_to_remove:
        model.graph.node.remove(node)

    # Add the new nodes and value infos
    model.graph.node.extend(new_nodes)
    model.graph.value_info.extend(new_value_infos)

    # Restore topological order: appending new_nodes to the end may break topo sort
    # (e.g. if downstream nodes appear before the inserted Reshape+Softmax nodes).
    # Re-sort via graphsurgeon when any change was made.
    if optimized:
        try:
            import onnx_graphsurgeon as gs

            g = gs.import_onnx(model)
            g.cleanup().toposort()
            model = gs.export_onnx(g)
        except Exception:
            pass  # If gs is unavailable fall back to unsorted (checker may still pass)

    # Save the optimized model
    print(f"Saving optimized model to {output_model_path}")
    onnx.save(model, output_model_path)

    print(f"Optimization complete. Modified {len(nodes_to_remove)} Softmax nodes.")


# ---------------------------------------------------------------------------
# ConcatTraining / SplitTraining → Concat / Split canonicalization
# ---------------------------------------------------------------------------


def _get_tensor_dim(graph: onnx.GraphProto, tensor_name: str, axis: int):
    """Return the size at *axis* for *tensor_name* by searching value_info/inputs."""
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if vi.name == tensor_name:
            t = vi.type.tensor_type
            if t.HasField("shape") and len(t.shape.dim) > axis:
                d = t.shape.dim[axis]
                if d.dim_value > 0:
                    return d.dim_value
    return None


def canonicalize_ms_training_ops(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Replace com.microsoft training-specific ops with standard ONNX equivalents.

    ConcatTraining(A, B, axis=a) → (C, per_input_length)
        becomes  Concat(A, B, axis=a) → C
        (per_input_length is compile-time constant; only needed by SplitTraining)

    SplitTraining(grad, per_input_length, axis=a) → (g_A, g_B)
        becomes  one Slice node per LIVE output (dead outputs are dropped):
            g_A = Slice(grad, start=0,    end=size_A, axes=[a])
            g_B = Slice(grad, start=size_A, end=size_A+size_B, axes=[a])
        Slice (not Split) is used because Deeploy supports Slice but not Split.
        Dead outputs (no downstream consumers) are silently omitted.

    Returns the modified model.
    """
    graph = model.graph

    # -------------------------------------------------------------------
    # Pass 1: Collect ConcatTraining info
    #   ct_map: per_input_length_name → {'inputs': [...], 'axis': int}
    # -------------------------------------------------------------------
    ct_map: dict = {}
    for node in graph.node:
        if node.op_type != "ConcatTraining":
            continue
        axis = 1  # default
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i
        main_out = node.output[0] if len(node.output) > 0 else None
        pil_name = node.output[1] if len(node.output) > 1 else None
        if pil_name:
            ct_map[pil_name] = {"inputs": list(node.input), "axis": axis, "main_out": main_out}

    n_concat = sum(1 for n in graph.node if n.op_type == "ConcatTraining")
    n_split = sum(1 for n in graph.node if n.op_type == "SplitTraining")
    if n_concat == 0 and n_split == 0:
        return model
    print(
        f"  canonicalize_ms_training_ops: found {n_concat} ConcatTraining, {n_split} SplitTraining"
    )

    # -------------------------------------------------------------------
    # Pass 2: Collect SplitTraining info.
    #   Key: first output name of the SplitTraining node (stable identifier).
    # -------------------------------------------------------------------
    # split_replacements: first_output_name → {"sizes": [...], "axis": int, "inputs": [...], "outputs": [...]}
    split_replacements: dict = {}
    for node in graph.node:
        if node.op_type != "SplitTraining":
            continue
        first_out = node.output[0] if len(node.output) > 0 else None
        if first_out is None:
            continue
        pil_name = node.input[1] if len(node.input) > 1 else None
        if pil_name is None:
            print(f"  WARNING: SplitTraining has no per_input_length input — skipping")
            continue
        axis = 1
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i
        ct_info = ct_map.get(pil_name)
        if ct_info is None:
            print(f"  WARNING: SplitTraining references unknown per_input_length '{pil_name}'")
            continue
        # Derive split sizes from ConcatTraining inputs
        sizes = []
        for inp_name in ct_info["inputs"]:
            sz = _get_tensor_dim(graph, inp_name, ct_info["axis"])
            if sz is None:
                print(f"  WARNING: Cannot infer dim for '{inp_name}' along axis={ct_info['axis']}")
                sz = 0
            sizes.append(sz)
        split_replacements[first_out] = {
            "sizes": sizes,
            "axis": axis,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "pil_name": pil_name,
        }

    # -------------------------------------------------------------------
    # Pass 3: Build replacement nodes and collect op-type/output sets
    #   to identify which original nodes to drop.
    #   Use output names as stable identifiers (not id(), which is
    #   unstable across protobuf field accesses).
    # -------------------------------------------------------------------
    new_nodes: list = []
    # Sets of first-output-names to drop from original graph.node list
    concat_training_main_outs: set = set()  # main outputs of ConcatTraining
    split_training_first_outs: set = set(split_replacements.keys())

    # Track per_input_length output names to remove from value_info later
    pil_names_to_remove: set = set(ct_map.keys())

    # Replace ConcatTraining → Concat
    for node in graph.node:
        if node.op_type != "ConcatTraining":
            continue
        axis = 1
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i
        new_node = helper.make_node(
            "Concat",
            inputs=list(node.input),
            outputs=[node.output[0]],  # drop per_input_length output
            axis=axis,
        )
        new_nodes.append(new_node)
        concat_training_main_outs.add(node.output[0])

    # Replace SplitTraining → one Slice per LIVE output.
    # Slice is in Deeploy's PULPOpen platform; Split is not.
    # Dead outputs (no consumers) are silently skipped.

    # Build consumer map: tensor_name → True/False (any consumer exists)
    consumers: set = set()
    for nd in graph.node:
        for inp in nd.input:
            if inp:
                consumers.add(inp)
    graph_outputs_set: set = {o.name for o in graph.output}

    for first_out, info in split_replacements.items():
        sizes = info["sizes"]
        axis = info["axis"]
        grad_input = info["inputs"][0]
        pil_name = info["pil_name"]
        outputs = info["outputs"]

        # Compute cumulative starts/ends for each output slice
        cumulative = 0
        for i, (out_name, sz) in enumerate(zip(outputs, sizes)):
            start = cumulative
            end = cumulative + sz
            cumulative += sz

            # Skip dead outputs (no downstream consumers and not a graph output)
            if out_name not in consumers and out_name not in graph_outputs_set:
                continue

            # Create initializers for starts, ends, axes (ONNX Slice opset 13+)
            uid = f"_slct_{pil_name.replace('/', '_')}_{i}"
            starts_name = uid + "_starts"
            ends_name = uid + "_ends"
            axes_name = uid + "_axes"
            graph.initializer.append(
                numpy_helper.from_array(np.array([start], dtype=np.int64), name=starts_name)
            )
            graph.initializer.append(
                numpy_helper.from_array(np.array([end], dtype=np.int64), name=ends_name)
            )
            graph.initializer.append(
                numpy_helper.from_array(np.array([axis], dtype=np.int64), name=axes_name)
            )
            new_node = helper.make_node(
                "Slice",
                inputs=[grad_input, starts_name, ends_name, axes_name],
                outputs=[out_name],
            )
            new_nodes.append(new_node)
            print(
                f"  SplitTraining[{i}] → Slice({grad_input!r}, {start}:{end}, axis={axis}) → {out_name!r}"
            )

        # per_input_length is now dead
        pil_names_to_remove.add(pil_name)

    # -------------------------------------------------------------------
    # Pass 4: Rebuild graph.node list — drop replaced nodes, add new ones
    # -------------------------------------------------------------------
    original_nodes = list(graph.node)
    del graph.node[:]
    for node in original_nodes:
        main_out = node.output[0] if len(node.output) > 0 else None
        if node.op_type == "ConcatTraining" and main_out in concat_training_main_outs:
            continue  # replaced
        if node.op_type == "SplitTraining" and main_out in split_training_first_outs:
            continue  # replaced
        graph.node.append(node)
    for node in new_nodes:
        graph.node.append(node)

    # -------------------------------------------------------------------
    # Pass 5: Remove orphaned per_input_length value_info entries
    # -------------------------------------------------------------------
    remaining_vi = [vi for vi in graph.value_info if vi.name not in pil_names_to_remove]
    del graph.value_info[:]
    for vi in remaining_vi:
        graph.value_info.append(vi)

    n_replaced = len(concat_training_main_outs) + len(split_training_first_outs)
    print(f"  Replaced {n_replaced} Microsoft training op(s) with standard ONNX equivalents.")
    return model


def canonicalize_ms_training_ops_file(input_path: str, output_path: str) -> None:
    """File-based wrapper for canonicalize_ms_training_ops."""
    model = onnx.load(input_path)
    model = canonicalize_ms_training_ops(model)
    onnx.save(model, output_path)
