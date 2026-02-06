# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Gradient optimization utilities for ONNX models.

This module contains functions for optimizing gradient-related operations in ONNX models,
including BiasGeluGrad removal, SoftmaxGrad renaming, and LayerNormGrad processing.
"""

import copy

import onnx
from onnx import helper


def run_optmization_remove_biasgelugrad(onnx_train_file: str, onnx_out_file: str):
    """
    Optimize ONNX model:
    1. Convert BiasGeluGrad_dX nodes to GeluGrad nodes
    2. Remove bias input (input 2)
    3. Delete subsequent ReduceSum nodes

    Args:
        onnx_train_file: Input ONNX model path
        onnx_out_file: Output ONNX model path
    """
    # Load ONNX model
    model = onnx.load(onnx_train_file)
    graph = model.graph
    modified_model = copy.deepcopy(model)
    modified_graph = modified_model.graph

    # Track nodes to be removed
    nodes_to_remove = set()

    # Map original output names to new output names
    output_mapping = {}

    # First pass: identify BiasGeluGrad_dX nodes and corresponding ReduceSum nodes
    for i, node in enumerate(graph.node):
        if node.op_type == "BiasGeluGrad_dX" or (
            node.domain == "com.microsoft" and "BiasGeluGrad_dX" in node.name
        ):
            # Create new GeluGrad node
            new_inputs = [node.input[0], node.input[1]]  # Remove bias input (input[2])
            new_node = helper.make_node(
                "GeluGrad", inputs=new_inputs, outputs=[node.output[0]], name=f"GeluGrad_{i}"
            )

            # Replace BiasGeluGrad_dX node with new GeluGrad node
            modified_graph.node.remove(modified_graph.node[i])
            modified_graph.node.insert(i, new_node)

            # Check for ReduceSum nodes that take this node's output as input
            for j, next_node in enumerate(graph.node):
                if next_node.op_type == "ReduceSum" and node.output[0] in next_node.input:
                    nodes_to_remove.add(j)
                    # Map ReduceSum output to new node output
                    output_mapping[next_node.output[0]] = node.output[0]

    # Second pass: remove identified ReduceSum nodes
    reduced_nodes = []
    for i, node in enumerate(modified_graph.node):
        if i not in nodes_to_remove:
            # Update inputs that reference outputs of removed nodes
            for j, input_name in enumerate(node.input):
                if input_name in output_mapping:
                    node.input[j] = output_mapping[input_name]
            reduced_nodes.append(node)

    # Replace nodes in the graph
    del modified_graph.node[:]
    modified_graph.node.extend(reduced_nodes)

    # Update model outputs if needed
    for output in modified_graph.output:
        if output.name in output_mapping:
            output.name = output_mapping[output.name]

    # Save modified model
    onnx.save(modified_model, onnx_out_file)
    print(f"Optimized model saved to {onnx_out_file}")


def rename_softmaxgrad_op(
    input_model_path: str,
    output_model_path: str,
    old_op_name: str = "SoftmaxGrad_13",
    new_op_name: str = "SoftmaxGrad",
):
    """
    Rename Microsoft's custom operator SoftmaxGrad_13 to SoftmaxGrad.

    Args:
        input_model_path: Path to the input ONNX model
        output_model_path: Path to save the converted ONNX model
        old_op_name: Original operator name (default: "SoftmaxGrad_13")
        new_op_name: New operator name (default: "SoftmaxGrad")
    """
    model = onnx.load(input_model_path)

    modified_nodes = []
    modified_count = 0

    # Process each node in the graph
    for node in model.graph.node:
        # Check if the node is the target Microsoft domain operator
        if node.op_type == old_op_name and node.domain == "com.microsoft":
            modified_count += 1

            # Create a new node with the updated op_type
            new_node = helper.make_node(
                op_type=new_op_name,
                inputs=list(node.input),
                outputs=list(node.output),
                name=node.name,
                domain=node.domain,  # Keep the original domain
            )

            # Copy all attributes from the original node
            for attr in node.attribute:
                new_node.attribute.append(attr)

            modified_nodes.append(new_node)
        else:
            # Keep all other nodes as they are
            modified_nodes.append(node)

    print(f"Modified {modified_count} {old_op_name} nodes to {new_op_name}")

    # Create a new graph with the modified nodes
    new_graph = helper.make_graph(
        nodes=modified_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=model.graph.initializer,
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


def remove_softmax_grad_loss_inputs(input_model_path: str, output_model_path: str):

    # Load the model
    model = onnx.load(input_model_path)
    graph = model.graph

    # Find SoftmaxCrossEntropyLossGrad nodes
    target_nodes = []
    for node in graph.node:
        if node.op_type == "SoftmaxCrossEntropyLossGrad":
            target_nodes.append(node)

    print(f"Found {len(target_nodes)} SoftmaxCrossEntropyLossGrad nodes")

    # Inputs to remove (first input)
    inputs_to_remove = []

    # Create new nodes with modified inputs
    new_nodes = []
    for node in graph.node:
        if node.op_type == "SoftmaxCrossEntropyLossGrad" and len(node.input) > 2:
            # Remove the first input and keep the rest
            first_input = node.input[0]
            inputs_to_remove.append(first_input)

            # Create a new node without the first input
            new_node = onnx.NodeProto()
            new_node.CopyFrom(node)

            # Keep only the second and third inputs
            remaining_inputs = list(node.input[1:])
            del new_node.input[:]
            new_node.input.extend(remaining_inputs)

            new_nodes.append(new_node)
        else:
            # Keep other nodes unchanged
            new_nodes.append(node)

    # Replace all nodes with the new set
    del graph.node[:]
    graph.node.extend(new_nodes)

    # Save the modified model
    onnx.save(model, output_model_path)
    print(
        f"Saved model with SoftmaxCrossEntropyLossGrad first input removed to: {output_model_path}"
    )


def process_layernormgrad_nodes(model_path: str, output_path: str, split_nodes: bool = False):
    """
    Process LayerNormalizationGrad nodes.

    Args:
        model_path: Path to the input ONNX model
        output_path: Path where the modified model will be saved
        split_nodes: If True, split into LayerNormGradX/W/B nodes.
                     If False, keep only dX output (original behavior).
    """
    model = onnx.load(model_path)
    graph = model.graph

    if not split_nodes:
        # Original behavior: keep only dX output
        for node in graph.node:
            if node.op_type == "LayerNormalizationGrad":
                inputs = list(node.input)
                outputs = list(node.output)

                if len(outputs) >= 1:
                    data_grad = outputs[0]
                    del node.output[:]
                    node.output.append(data_grad)

                # Find bias name
                bias_name = None
                if len(inputs) >= 3:
                    weight_name = inputs[2]
                    possible_bias_name = weight_name.replace("weight", "bias")
                    for initializer in graph.initializer:
                        if initializer.name == possible_bias_name:
                            bias_name = possible_bias_name
                            break

                if bias_name is None:
                    for n in graph.node:
                        if n.op_type == "LayerNormalization":
                            if len(n.input) >= 3:
                                bias_name = n.input[2]
                                break

                if bias_name is not None:
                    new_inputs = []
                    if len(inputs) >= 2:
                        new_inputs.extend([inputs[0], inputs[1]])
                    if len(inputs) >= 3:
                        new_inputs.append(inputs[2])
                    new_inputs.append(bias_name)
                    del node.input[:]
                    node.input.extend(new_inputs)
                else:
                    print(f"Warning: Bias input could not be found for node {node.name}")

        onnx.save(model, output_path)
        print(f"Modified model saved to {output_path}")
        return

    # Split mode: split LayerNormalizationGrad into X, W, B gradient nodes
    original_graph_outputs = set(out.name for out in graph.output)

    # Build a set of all tensor names that are used as inputs by other nodes
    used_tensors = set()
    for node in graph.node:
        for inp in node.input:
            used_tensors.add(inp)
    # Also add graph outputs as used
    used_tensors.update(original_graph_outputs)

    nodes_to_add = []
    nodes_to_remove = []

    for node in graph.node:
        if node.op_type == "LayerNormalizationGrad":
            inputs = list(node.input)
            outputs = list(node.output)

            # LayerNormGrad inputs:
            # [0] dY: upstream gradient
            # [1] X: forward input
            # [2] gamma: scale/weight
            # [3] mean: computed mean
            # [4] inv_std_dev: 1/sqrt(var + eps)
            dY = inputs[0] if len(inputs) > 0 else None
            X = inputs[1] if len(inputs) > 1 else None
            gamma = inputs[2] if len(inputs) > 2 else None
            mean = inputs[3] if len(inputs) > 3 else None
            inv_std = inputs[4] if len(inputs) > 4 else None

            # Copy attributes
            attrs = {attr.name: attr for attr in node.attribute}

            # LayerNormGrad outputs:
            # [0] dX: gradient w.r.t. input
            # [1] dGamma: gradient w.r.t. scale/weight
            # [2] dBeta: gradient w.r.t. bias

            # Create LayerNormGradX node (dX is always needed for backprop)
            if len(outputs) > 0 and outputs[0] != "":
                dX_output = outputs[0]
                grad_x_inputs = [i for i in [dY, X, gamma, mean, inv_std] if i is not None]

                layernormgrad_x = helper.make_node(
                    "LayerNormGradX",
                    inputs=grad_x_inputs,
                    outputs=[dX_output],
                    name=node.name + "_X",
                )
                for attr_name, attr in attrs.items():
                    layernormgrad_x.attribute.append(copy.deepcopy(attr))
                nodes_to_add.append(layernormgrad_x)
                print(f"✅ Created LayerNormGradX: {layernormgrad_x.name} -> {dX_output}")

            # Create LayerNormGradW node only if dW output is used
            if len(outputs) > 1 and outputs[1] != "" and outputs[1] in used_tensors:
                dW_output = outputs[1]
                grad_w_inputs = [i for i in [dY, X, mean, inv_std] if i is not None]

                layernormgrad_w = helper.make_node(
                    "LayerNormGradW",
                    inputs=grad_w_inputs,
                    outputs=[dW_output],
                    name=node.name + "_W",
                )
                for attr_name, attr in attrs.items():
                    layernormgrad_w.attribute.append(copy.deepcopy(attr))
                nodes_to_add.append(layernormgrad_w)
                print(f"✅ Created LayerNormGradW: {layernormgrad_w.name} -> {dW_output}")
            elif len(outputs) > 1 and outputs[1] != "":
                print(f"⏭️ Skipped LayerNormGradW: {outputs[1]} not used")

            # Create LayerNormGradB node only if dB output is used
            if len(outputs) > 2 and outputs[2] != "" and outputs[2] in used_tensors:
                dB_output = outputs[2]
                layernormgrad_b = helper.make_node(
                    "LayerNormGradB",
                    inputs=[dY],  # Only needs upstream gradient
                    outputs=[dB_output],
                    name=node.name + "_B",
                )
                for attr_name, attr in attrs.items():
                    layernormgrad_b.attribute.append(copy.deepcopy(attr))
                nodes_to_add.append(layernormgrad_b)
                print(f"✅ Created LayerNormGradB: {layernormgrad_b.name} -> {dB_output}")
            elif len(outputs) > 2 and outputs[2] != "":
                print(f"⏭️ Skipped LayerNormGradB: {outputs[2]} not used")

            nodes_to_remove.append(node)

    # Remove original nodes and add new split nodes
    for node in nodes_to_remove:
        graph.node.remove(node)
    graph.node.extend(nodes_to_add)

    onnx.save(model, output_path)
    print(
        f"✅ Split {len(nodes_to_remove)} LayerNormGrad node(s) into {len(nodes_to_add)} separate gradient nodes"
    )
    print(f"Modified model saved to {output_path}")
