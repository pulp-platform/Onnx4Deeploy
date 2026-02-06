# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Node utility functions for ONNX models.

This module contains utility functions for working with ONNX nodes,
including node name annotation, backward markers, and normalization conversions.
"""

import onnx
from onnx import TensorProto, helper, numpy_helper


def annotate_node_names_with_type(model):
    """
    Add type annotation to each node name in ONNX model

    Args:
        model: ONNX model object

    Returns:
        Modified ONNX model object
    """
    # Traverse all nodes in the model
    for i, node in enumerate(model.graph.node):
        # Get node operation type
        op_type = node.op_type

        # If node already has name, add type suffix
        if node.name:
            new_name = f"{node.name}_{op_type}"
        else:
            # If node has no name, create one based on index and type
            new_name = f"node_{i}_{op_type}"

        # Update node name
        node.name = new_name

    return model


def process_onnx_model_name_with_type(model_path: str, output_path: str = None):
    """
    Process ONNX model to add type annotations to node names

    Args:
        input_path: Input ONNX model file path
        output_path: Output ONNX model file path (optional)
    """
    model = onnx.load(model_path)

    # Traverse all nodes, add type to name
    for i, node in enumerate(model.graph.node):
        if node.name:
            node.name = f"{node.name}_{node.op_type}"
        else:
            node.name = f"node_{i}_{node.op_type}"

    # SaveModel（without validation）
    save_path = output_path if output_path else model_path
    onnx.save(model, save_path)

    return model


def add_backward_markers_to_nodes(input_model_path: str, output_model_path: str):
    """
    Add backward markers to node names based on various criteria:
    1. Node name contains 'grad' (case insensitive)
    2. Input/output tensor names contain 'grad' (case insensitive)
    3. Description contains 'Backward pass' (case insensitive)

    Args:
        input_model_path: Path to the input ONNX model
        output_model_path: Path to save the marked ONNX model
    """
    # Load the model
    model = onnx.load(input_model_path)

    print("🔍 Analyzing nodes for backward pass detection...")

    backward_nodes_found = 0
    already_marked_nodes = 0

    for node in model.graph.node:
        is_backward = False
        detection_reasons = []

        # Check 1: Node name contains 'grad'
        if node.name and "grad" in node.name.lower():
            is_backward = True
            detection_reasons.append(f"node name contains 'grad': {node.name}")

        # Check 2: Input tensor names contain 'grad'
        for input_tensor in node.input:
            if "grad" in input_tensor.lower():
                is_backward = True
                detection_reasons.append(f"input tensor contains 'grad': {input_tensor}")
                break  # Found one, no need to check more inputs

        # Check 3: Output tensor names contain 'grad'
        if not is_backward:  # Only check if not already detected
            for output_tensor in node.output:
                if "grad" in output_tensor.lower():
                    is_backward = True
                    detection_reasons.append(f"output tensor contains 'grad': {output_tensor}")
                    break  # Found one, no need to check more outputs

        # Check 4: Description contains 'Backward pass'
        if not is_backward:  # Only check if not already detected
            for attr in node.attribute:
                if attr.name == "description":
                    description = attr.s.decode("utf-8") if attr.s else ""
                    if "backward pass" in description.lower():
                        is_backward = True
                        detection_reasons.append(
                            f"description contains 'Backward pass': {description}"
                        )
                        break

        # If this is a backward node, modify its name
        if is_backward:
            # Check if already marked with 'backward'
            if "backward" in node.name.lower():
                already_marked_nodes += 1
                print(f"  ⚪ Already marked: {node.name}")
            else:
                # Add 'backward' to the node name
                if node.name:
                    new_name = f"{node.name}_backward"
                else:
                    new_name = f"backward_node_{backward_nodes_found}"

                old_name = node.name
                node.name = new_name
                backward_nodes_found += 1

                print(f"  🔴 Marked backward: {old_name} -> {new_name}")
                for reason in detection_reasons:
                    print(f"     Reason: {reason}")

                # Also add/update description attribute to mark as backward
                description_found = False
                for attr in node.attribute:
                    if attr.name == "description":
                        original_desc = attr.s.decode("utf-8") if attr.s else ""
                        if "backward pass" not in original_desc.lower():
                            new_desc = (
                                f"[Backward pass] {original_desc}"
                                if original_desc
                                else "[Backward pass]"
                            )
                            attr.s = new_desc.encode("utf-8")
                            print(f"     Updated description: {new_desc}")
                        description_found = True
                        break

                # If no description attribute exists, create one
                if not description_found:
                    desc_attr = helper.make_attribute("description", "[Backward pass]")
                    node.attribute.append(desc_attr)
                    print("     Added description: [Backward pass]")

    # Save the modified model
    onnx.save(model, output_model_path)

    # Summary
    total_nodes = len(model.graph.node)
    total_backward = backward_nodes_found + already_marked_nodes
    forward_nodes = total_nodes - total_backward

    print(f"\n{'='*60}")
    print("✅ BACKWARD NODE MARKING SUMMARY")
    print(f"{'='*60}")
    print(f"Total nodes in model: {total_nodes}")
    print(f"Forward nodes: {forward_nodes} ({forward_nodes/total_nodes*100:.1f}%)")
    print(f"Backward nodes (total): {total_backward} ({total_backward/total_nodes*100:.1f}%)")
    print(f"  - Newly marked: {backward_nodes_found}")
    print(f"  - Already marked: {already_marked_nodes}")
    print(f"Model saved to: {output_model_path}")
    print(f"{'='*60}")

    return model


def convert_layernorm_to_groupnorm(
    input_onnx: str, output_onnx: str, num_groups: int = 1, split_grad_nodes: bool = False
):
    """
    1. Convert LayerNorm to GroupNorm
    2. If split_grad_nodes is enabled, execute direct mode without Slice of Stat
    """
    model = onnx.load(input_onnx)
    graph = model.graph

    # --- Step 1: Basic operator conversion ---
    layernorm_params = {}
    for node in graph.node:
        if node.op_type == "LayerNormalization":
            # Record parameters for later shape adjustment
            for i in [1, 2]:  # weight, bias
                if len(node.input) > i:
                    param_name = node.input[i]
                    for init in graph.initializer:
                        if init.name == param_name:
                            array = numpy_helper.to_array(init)
                            layernorm_params[param_name] = {
                                "num_channels": array.shape[0],
                                "shape": array.shape,
                            }

            node.op_type = "GroupNormalization"
            # Temporarily clear attributes, will add them uniformly later
            epsilon = 1e-5
            for attr in node.attribute:
                if attr.name == "epsilon":
                    epsilon = attr.f
            del node.attribute[:]
            node.attribute.append(helper.make_attribute("epsilon", epsilon))
            node.attribute.append(helper.make_attribute("num_groups", num_groups))

        # Also convert original Grad operator name for easier later processing
        if node.op_type == "LayerNormalizationGrad":
            node.op_type = "GroupNormalizationGrad"
            epsilon = 1e-5
            for attr in node.attribute:
                if attr.name == "epsilon":
                    epsilon = attr.f
            del node.attribute[:]
            node.attribute.append(helper.make_attribute("epsilon", epsilon))
            node.attribute.append(helper.make_attribute("num_groups", num_groups))

    # --- Step 2: Adjust weight dimensions ---
    for init in graph.initializer:
        if init.name in layernorm_params and len(layernorm_params[init.name]["shape"]) > 1:
            array = numpy_helper.to_array(init)
            new_array = (
                array.reshape(layernorm_params[init.name]["num_channels"], -1)
                .mean(axis=1)
                .astype(array.dtype)
            )
            new_init = numpy_helper.from_array(new_array, init.name)
            init.CopyFrom(new_init)

    onnx.save(model, output_onnx)


def split_gn_to_single_stat_array(onnx_path: str, save_path: str):
    model = onnx.load(onnx_path)
    graph = model.graph

    # Helper function：from graph Get tensor Shape
    def get_shape(tensor_name):
        # Check graph input
        for vi in graph.input:
            if vi.name == tensor_name:
                return [d.dim_value for d in vi.type.tensor_type.shape.dim]
        # Check value_info
        for vi in graph.value_info:
            if vi.name == tensor_name:
                return [d.dim_value for d in vi.type.tensor_type.shape.dim]
        # Check initializer (sometimes shape is stored here)
        for init in graph.initializer:
            if init.name == tensor_name:
                return list(init.dims)
        return None

    # 1. Build global tensor reference table
    used_tensors = set(out.name for out in graph.output)
    for n in graph.node:
        for input_name in n.input:
            used_tensors.add(input_name)

    nodes_to_add = []
    nodes_to_remove = []
    x_to_stat_map = {}
    new_value_infos = []

    # --- Step 1：Forward processing ---
    for node in list(graph.node):
        if node.op_type == "GroupNormalization":
            X, gamma, beta = node.input[0], node.input[1], node.input[2]
            Y = node.output[0]
            stat_array_name = f"{X}_stat_combined"
            x_to_stat_map[X] = stat_array_name

            # Fix dict comprehension naming error here
            attrs = {a.name: a for a in node.attribute}
            num_groups = attrs["num_groups"].i
            epsilon = attrs["epsilon"].f if "epsilon" in attrs else 1e-5

            # Deduce并记录 stat 张量of Shape [N, G, 2]
            x_shape = get_shape(X)
            if x_shape and len(x_shape) > 0:
                N = x_shape[0]
                stat_shape = [N, num_groups, 2]
                stat_vi = helper.make_tensor_value_info(
                    stat_array_name, TensorProto.FLOAT, stat_shape
                )
                new_value_infos.append(stat_vi)

            # CreateForward node after splitting
            nodes_to_add.append(
                helper.make_node(
                    "GroupNormStats",
                    [X],
                    [stat_array_name],
                    name=f"{node.name}_stats",
                    num_groups=num_groups,
                    epsilon=epsilon,
                )
            )
            nodes_to_add.append(
                helper.make_node(
                    "GroupNormForward",
                    [X, gamma, beta, stat_array_name],
                    [Y],
                    name=f"{node.name}_fwd",
                    num_groups=num_groups,
                )
            )
            nodes_to_remove.append(node)

    # --- Step 2：按需拆分反向节点 ---
    for node in list(graph.node):
        if node.op_type == "GroupNormalizationGrad":
            dY, X, gamma = node.input[0], node.input[1], node.input[2]
            dX, dGamma, dBeta = node.output[0], node.output[1], node.output[2]

            # Get corresponding stat 张量名
            target_stat = x_to_stat_map.get(X, f"{X}_stat_combined")

            # Fix naming error again
            attrs = {a.name: a for a in node.attribute}
            num_groups = attrs["num_groups"].i

            # Only create gradient node when needed
            if dX != "" and dX in used_tensors:
                # Create grad_stat intermediate tensor name
                grad_stat_name = f"{dX}_grad_stat"

                # Deduce grad_stat shape [N, num_groups, 2]
                x_shape = get_shape(X)
                if x_shape and len(x_shape) > 0:
                    N = x_shape[0]
                    grad_stat_shape = [N, num_groups, 2]
                    grad_stat_vi = helper.make_tensor_value_info(
                        grad_stat_name, TensorProto.FLOAT, grad_stat_shape
                    )
                    new_value_infos.append(grad_stat_vi)

                # Step 1：Compute grad_stat (mean_gamma_dY, mean_gamma_dY_Xnorm)
                nodes_to_add.append(
                    helper.make_node(
                        "GroupNormGradXStat",
                        [dY, X, gamma, target_stat],
                        [grad_stat_name],
                        name=f"{node.name}_dX_stat",
                        num_groups=num_groups,
                    )
                )

                # Step 2：Use grad_stat Compute dX (Support HW tiling)
                nodes_to_add.append(
                    helper.make_node(
                        "GroupNormGradX",
                        [dY, X, gamma, target_stat, grad_stat_name],
                        [dX],
                        name=f"{node.name}_dX",
                        num_groups=num_groups,
                    )
                )

            if dGamma != "" and dGamma in used_tensors:
                nodes_to_add.append(
                    helper.make_node(
                        "GroupNormGradW",
                        [dY, X, target_stat],
                        [dGamma],
                        name=f"{node.name}_dW",
                        num_groups=num_groups,
                    )
                )
                # Fix dGamma shape：from LayerNorm of [C, H, W] 改for GroupNorm of [C]
                x_shape = get_shape(X)
                if x_shape and len(x_shape) >= 2:
                    C = x_shape[1]
                    # Delete old value_info
                    for vi in list(graph.value_info):
                        if vi.name == dGamma:
                            graph.value_info.remove(vi)
                    # Addcorrect shapeof value_info
                    dGamma_vi = helper.make_tensor_value_info(dGamma, TensorProto.FLOAT, [C])
                    new_value_infos.append(dGamma_vi)

            if dBeta != "" and dBeta in used_tensors:
                nodes_to_add.append(
                    helper.make_node("GroupNormGradB", [dY], [dBeta], name=f"{node.name}_dB")
                )
                # Fix dBeta shape：from LayerNorm of [C, H, W] 改for GroupNorm of [C]
                x_shape = get_shape(X)
                if x_shape and len(x_shape) >= 2:
                    C = x_shape[1]
                    # Delete old value_info
                    for vi in list(graph.value_info):
                        if vi.name == dBeta:
                            graph.value_info.remove(vi)
                    # Addcorrect shapeof value_info
                    dBeta_vi = helper.make_tensor_value_info(dBeta, TensorProto.FLOAT, [C])
                    new_value_infos.append(dBeta_vi)

            nodes_to_remove.append(node)

    # 3. Update graph structure
    for n in nodes_to_remove:
        graph.node.remove(n)
    graph.node.extend(nodes_to_add)

    graph.value_info.extend(new_value_infos)

    norm_param_shapes = {}
    for init in graph.initializer:
        if "layer_norm" in init.name and ("weight" in init.name or "bias" in init.name):
            norm_param_shapes[init.name] = list(init.dims)

    for out in graph.output:
        if "_updated" in out.name:

            param_name = out.name.replace("_updated", "")
            if param_name in norm_param_shapes:
                correct_shape = norm_param_shapes[param_name]

                out.type.tensor_type.shape.Clear()

                for dim_val in correct_shape:
                    out.type.tensor_type.shape.dim.add().dim_value = dim_val

    try:
        model = onnx.shape_inference.infer_shapes(model)
    except:
        pass

    onnx.save(model, save_path)
    print(f"✅ Conversion successful：Processed {onnx_path}")
