import onnx
import os
import re
import subprocess
import yaml
import torch
from onnx import helper, numpy_helper, shape_inference
import numpy as np
import onnxruntime.tools
from onnxruntime.tools import symbolic_shape_infer
import copy
from .fixShape import print_onnx_shapes
from .trainOptimization import *
import random
from onnx import TensorProto

def make_c_name(name, count=0):
    if name.lower() in ["input", "output"]:
        return name  # Keep 'input' and 'output' as is

    name = re.sub(r'input|output', '', name, flags=re.IGNORECASE)  # Remove 'input' and 'output' from other names
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if name is None or name == "":
        name = f'node_{count}'
    if name[0].isdigit() or name[0] == '_':
        name = f'node_{count}' + name
    return name

def rename_onnx_nodes(model):
    i_node = 0
    for node in model.graph.node:
        i_node += 1
        node.name = make_c_name(node.name, i_node)
        for i, input_name in enumerate(node.input):
            node.input[i] = make_c_name(input_name)
        for i, output_name in enumerate(node.output):
            node.output[i] = make_c_name(output_name)

    for input in model.graph.input:
        input.name = make_c_name(input.name)
    for output in model.graph.output:
        output.name = make_c_name(output.name)

    for init in model.graph.initializer:
        init.name = make_c_name(init.name)

    return model

def rename_and_save_onnx(input_onnx, output_onnx):
    model = onnx.load(input_onnx)
    model = rename_onnx_nodes(model)
    onnx.save(model, output_onnx)
    print(f"✅ Renamed ONNX model saved to {output_onnx}")

def run_onnx_optimization_infer(onnx_file, embedding_dim, num_heads, input_shape):
    
    batch_size, channels, height, width = input_shape  # Extract input dimensions
    try:
        print("🔹 Fixing dynamic shape...")
        subprocess.run([
            "python", "-m", "onnxruntime.tools.make_dynamic_shape_fixed",
            "--input_name", "input",
            "--input_shape", f"{batch_size},{channels},{height},{width}",
            onnx_file, onnx_file
        ], check=True)

        print("🔹 Running symbolic shape inference...")
        subprocess.run([
            "python", "-m", "onnxruntime.tools.symbolic_shape_infer",
            "--input", onnx_file, "--output", onnx_file, "--verbose", "3"
        ], check=True)

        print("🔹 Optimizing ONNX model for ViT...")
        subprocess.run([
            "python", "-m", "onnxruntime.transformers.optimizer",
            "--input", onnx_file, "--output", onnx_file,
            "--model_type", "vit",
            "--num_heads", str(num_heads),  # Controlled via config
            "--hidden_size", str(embedding_dim),  # Ensures hidden size = embedding_dim
            "--use_multi_head_attention",
            "--disable_bias_skip_layer_norm",
            "--disable_skip_layer_norm",
            "--disable_bias_gelu"
        ], check=True)

        print("✅ ONNX model optimization complete!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during ONNX optimization: {e}")

def run_onnx_optimization(onnx_file, embedding_dim, num_heads, input_shape):
    """ Run ONNX Runtime tools to optimize the model """

    batch_size, channels, height, width = input_shape  # Extract input dimensions

    try:
        print("🔹 Fixing dynamic shape...")
        subprocess.run([
            "python", "-m", "onnxruntime.tools.make_dynamic_shape_fixed",
            "--input_name", "input",
            "--input_shape", f"{batch_size},{channels},{height},{width}",
            onnx_file, onnx_file
        ], check=True)

        print("🔹 Running symbolic shape inference...")
        subprocess.run([
            "python", "-m", "onnxruntime.tools.symbolic_shape_infer",
            "--input", onnx_file, "--output", onnx_file, "--verbose", "3"
        ], check=True)

        print("🔹 Optimizing ONNX model for ViT...")
        subprocess.run([
            "python", "-m", "onnxruntime.transformers.optimizer",
            "--input", onnx_file, "--output", onnx_file,
            "--model_type", "vit",
            "--num_heads", str(num_heads),  # Controlled via config
            "--hidden_size", str(embedding_dim),  # Ensures hidden size = embedding_dim
            "--use_multi_head_attention",
            "--disable_bias_skip_layer_norm",
            "--disable_skip_layer_norm",
            "--disable_bias_gelu",
            "--disable_layer_norm",  # compatible with opset 15
        ], check=True)

        print("✅ ONNX model optimization complete!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during ONNX optimization: {e}")

    # fuse_matmul_add_to_gemm(onnx_file, onnx_file)
    # print(
    #     f"✅ Successfully fused MatMul and Add to Gemm nodes. Saved as {onnx_file}"
    # )
    

def load_config(config_filename="../config.yaml"):
    """Load and parse config.yaml, returning CCT-specific parameters in a single return statement."""
    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, config_filename)

    with open(config_file, "r") as f:
        config = yaml.safe_load(f).get("cct", {})

    return (
        config["pretrained"],
        config["img_size"],
        config["num_classes"],
        config["embedding_dim"],
        config["num_heads"],
        config["num_layers"],
        config["batch_size"],
        config.get("opset_version", 12)  # Default value for opset_version
    )

def load_train_config(config_filename="../config.yaml"):
    """Load and parse config.yaml, returning CCT-specific parameters in a single return statement."""
    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, config_filename)

    with open(config_file, "r") as f:
        config = yaml.safe_load(f).get("training", {})
    
    return config.get("learning_rate", 0.01)


def compare_onnx_models(model1_path, model2_path, name1="Model 1", name2="Model 2"):
    """
    Compare two ONNX models and report differences in nodes, inputs, outputs.
    """
    model1 = onnx.load(model1_path)
    model2 = onnx.load(model2_path)

    graph1 = model1.graph
    graph2 = model2.graph

    print(f"\n{'='*60}")
    print(f"Comparing {name1} vs {name2}")
    print(f"{'='*60}")

    # Compare node counts
    print(f"\n📊 Node Statistics:")
    print(f"  {name1}: {len(graph1.node)} nodes")
    print(f"  {name2}: {len(graph2.node)} nodes")
    print(f"  Difference: {len(graph2.node) - len(graph1.node):+d} nodes")

    # Compare node types
    types1 = {}
    types2 = {}
    for node in graph1.node:
        types1[node.op_type] = types1.get(node.op_type, 0) + 1
    for node in graph2.node:
        types2[node.op_type] = types2.get(node.op_type, 0) + 1

    all_types = set(types1.keys()) | set(types2.keys())

    print(f"\n📋 Node Type Changes:")
    for op_type in sorted(all_types):
        count1 = types1.get(op_type, 0)
        count2 = types2.get(op_type, 0)
        if count1 != count2:
            print(f"  {op_type}: {count1} → {count2} ({count2 - count1:+d})")

    # Compare graph outputs
    print(f"\n🎯 Graph Outputs:")
    print(f"  {name1}: {len(graph1.output)} outputs")
    outputs1_names = [out.name for out in graph1.output]
    for i, out in enumerate(outputs1_names[:10]):  # Show first 10
        print(f"    - {out}")
    if len(outputs1_names) > 10:
        print(f"    ... and {len(outputs1_names) - 10} more")

    print(f"  {name2}: {len(graph2.output)} outputs")
    outputs2_names = [out.name for out in graph2.output]
    for i, out in enumerate(outputs2_names[:10]):  # Show first 10
        print(f"    - {out}")
    if len(outputs2_names) > 10:
        print(f"    ... and {len(outputs2_names) - 10} more")

    # Check for missing outputs
    outputs1 = set(outputs1_names)
    outputs2 = set(outputs2_names)

    missing = outputs1 - outputs2
    added = outputs2 - outputs1

    if missing:
        print(f"\n⚠️  Missing outputs in {name2}: {len(missing)} outputs lost")
        for out in list(missing)[:20]:  # Show first 20
            print(f"    - {out}")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")

    if added:
        print(f"\n✨ New outputs in {name2}: {len(added)} outputs added")
        for out in list(added)[:20]:  # Show first 20
            print(f"    - {out}")
        if len(added) > 20:
            print(f"    ... and {len(added) - 20} more")

    print(f"{'='*60}\n")

    return {
        'node_diff': len(graph2.node) - len(graph1.node),
        'missing_outputs': list(missing),
        'added_outputs': list(added),
        'types_diff': {k: (types2.get(k, 0) - types1.get(k, 0)) for k in all_types if types1.get(k, 0) != types2.get(k, 0)}
    }


def split_convgrad_nodes(input_onnx, output_onnx):
    """
    Split ConvGrad nodes with 3 outputs into 3 separate nodes:
    ConvGradX, ConvGradW, and ConvGradB.
    
    Each node only receives the inputs it actually needs:
    - ConvGradX: [dY, W] - computes input gradient
    - ConvGradW: [dY, X] - computes weight gradient  
    - ConvGradB: [dY] - computes bias gradient
    """
    model = onnx.load(input_onnx)
    graph = model.graph

    # Store original graph outputs to preserve them
    original_graph_outputs = set(out.name for out in graph.output)

    nodes_to_add = []
    nodes_to_remove = []

    # Track all output tensors that should be preserved
    preserved_outputs = {}

    for node in graph.node:
        if node.op_type == "ConvGrad":
            # Get inputs: dY, X, W
            if len(node.input) < 3:
                continue

            dY = node.input[0]  # Output gradient
            X = node.input[1]   # Forward input
            W = node.input[2]   # Weight

            # Get outputs - keep original positions to avoid index misalignment
            # In ONNX, empty strings indicate unused outputs
            # node.output[0] = dX (gradient w.r.t. input)
            # node.output[1] = dW (gradient w.r.t. weight)
            # node.output[2] = dB (gradient w.r.t. bias, optional)

            # Check if we have enough outputs
            if len(node.output) < 2:
                continue

            # Copy attributes from original node
            attrs = {attr.name: attr for attr in node.attribute}

            # Create ConvGradX node if dX output exists (index 0)
            if len(node.output) > 0 and node.output[0] != '':
                dX_output = node.output[0]
                convgrad_x = helper.make_node(
                    'ConvGradX',
                    inputs=[dY, W],  # ConvTranspose-like operation
                    outputs=[dX_output],  # dX
                    name=node.name + '_X'
                )
                # Copy attributes
                for attr_name, attr in attrs.items():
                    convgrad_x.attribute.append(copy.deepcopy(attr))
                nodes_to_add.append(convgrad_x)

                # Track if this output is a graph output
                if dX_output in original_graph_outputs:
                    preserved_outputs[dX_output] = True

            # Create ConvGradW node if dW output exists (index 1)
            if len(node.output) > 1 and node.output[1] != '':
                dW_output = node.output[1]
                convgrad_w = helper.make_node(
                    'ConvGradW',
                    inputs=[dY, X],  # Correlation operation
                    outputs=[dW_output],  # dW
                    name=node.name + '_W'
                )
                # Copy attributes
                for attr_name, attr in attrs.items():
                    convgrad_w.attribute.append(copy.deepcopy(attr))
                nodes_to_add.append(convgrad_w)

                # Track if this output is a graph output
                if dW_output in original_graph_outputs:
                    preserved_outputs[dW_output] = True

            # Create ConvGradB node if dB output exists (index 2)
            if len(node.output) > 2 and node.output[2] != '':
                dB_output = node.output[2]
                convgrad_b = helper.make_node(
                    'ConvGradB',
                    inputs=[dY],  # Sum over spatial dimensions
                    outputs=[dB_output],  # dB
                    name=node.name + '_B'
                )
                # Copy attributes (though ConvGradB may not need all of them)
                for attr_name, attr in attrs.items():
                    convgrad_b.attribute.append(copy.deepcopy(attr))
                nodes_to_add.append(convgrad_b)

                # Track if this output is a graph output
                if dB_output in original_graph_outputs:
                    preserved_outputs[dB_output] = True

            # Mark original node for removal
            nodes_to_remove.append(node)

    # Remove original ConvGrad nodes
    for node in nodes_to_remove:
        graph.node.remove(node)

    # Add new split nodes
    graph.node.extend(nodes_to_add)

    # Save the modified model
    onnx.save(model, output_onnx)

    # Verify no graph outputs were lost
    new_model = onnx.load(output_onnx)
    new_graph_outputs = set(out.name for out in new_model.graph.output)
    lost_outputs = original_graph_outputs - new_graph_outputs

    if lost_outputs:
        print(f"⚠️  WARNING: Lost {len(lost_outputs)} graph outputs after ConvGrad split:")
        for out in list(lost_outputs)[:10]:
            print(f"    - {out}")
        if len(lost_outputs) > 10:
            print(f"    ... and {len(lost_outputs) - 10} more")
    else:
        print(f"✅ All {len(original_graph_outputs)} graph outputs preserved after ConvGrad split")

    print(f"✅ Split {len(nodes_to_remove)} ConvGrad node(s) into {len(nodes_to_add)} separate gradient nodes")
    if preserved_outputs:
        print(f"   {len(preserved_outputs)} ConvGrad outputs are graph outputs")


def run_train_onnx_optimization(onnx_train_file, onnx_output_file, split_layernormgrad=False):
    """
    Run optimization passes on the training ONNX model.

    Args:
        onnx_train_file: Path to the input training ONNX model
        onnx_output_file: Path to save the optimized model
        split_layernormgrad: If True, split LayerNormGrad into X/W/B nodes (for training norm params)
    """
    print(f"🔹 Running optimization for {onnx_train_file}...")

    fix_layernorm_output(onnx_train_file, onnx_output_file)
    print(
        f"✅ Successfully fixed LayerNormalization opset version. Saved as {onnx_output_file}"
    )
    optimize_softmax_axis(onnx_output_file, onnx_output_file)
    print(
        f"✅ Successfully optimized Softmax axis. Saved as {onnx_output_file}")

    optimize_reshape_fusion(onnx_output_file, onnx_output_file)
    print(
        f"✅ Successfully optimized Reshape nodes. Saved as {onnx_output_file}")

    modify_conflict_outputs(onnx_output_file, onnx_output_file)
    print(
        f"✅ Successfully removed all second outputs from Maxpool nodes. Saved as {onnx_output_file}"
    )

    convert_squeeze_unsqueeze_input_to_attr(onnx_output_file, onnx_output_file)
    print(
        f"✅ Successfully converted Squeeze inputs to attributes. Saved as {onnx_output_file}"
    )

    convert_reducesum_to_reshape(onnx_output_file, onnx_output_file)
    print(
        f"✅ Successfully converted ReduceSum to ReduceSum+Reshape. Saved as {onnx_output_file}"
    )

    remove_identity_nodes(onnx_output_file, onnx_output_file)
    print(
        f"✅ Successfully removed Identity nodes. Saved as {onnx_output_file}"
    )

    convert_reducesum_axes_to_attr(onnx_output_file, onnx_output_file)
    print(
        f"✅ Successfully converted ReduceSum axes to attributes. Saved as {onnx_output_file}"
    )

    convert_fusedmatmul_to_no_bias_gemm(onnx_output_file, onnx_output_file)
    print(
        f"✅ Successfully converted FusedMatMul to Gemm nodes. Saved as {onnx_output_file}"
    )

    convert_sum_to_add(onnx_output_file, onnx_output_file)
    print(
        f"✅ Successfully converted Sum to Add nodes. Saved as {onnx_output_file}"
    )

    rename_softmaxgrad_op(onnx_output_file, onnx_output_file)
    print(
        f"✅ Successfully renamed SoftmaxGrad nodes. Saved as {onnx_output_file}"
    )

    split_convgrad_nodes(onnx_output_file, onnx_output_file)
    print(
        f"✅ Successfully split ConvGrad nodes into ConvGradX/W/B. Saved as {onnx_output_file}"
    )

    remove_softmax_loss_outputs(onnx_output_file, onnx_output_file)
    print(f"✅ Successfully removed Softmax Loss outputs. Saved as {onnx_output_file}")

    remove_softmax_grad_loss_inputs(onnx_output_file, onnx_output_file)
    print(
        f"✅ Successfully removed Softmax Grad Loss inputs. Saved as {onnx_output_file}"
    )
    process_layernormgrad_nodes(
        onnx_output_file, onnx_output_file, split_nodes=split_layernormgrad
    )  # Process LayerNormGrad nodes
    if split_layernormgrad:
        print(f"✅ Successfully split LayerNormGrad into X/W/B nodes. Saved as {onnx_output_file}")
    else:
        print(f"✅ Successfully processed LayerNormGrad nodes. Saved as {onnx_output_file}")
    
    run_optmization_remove_biasgelu(onnx_output_file, onnx_output_file)
    print(f"✅ Successfully removed BiasGeluFusion. Saved as {onnx_output_file}")

    run_optmization_remove_biasgelugrad(onnx_output_file, onnx_output_file)
    print(
        f"✅ Successfully removed BiasGeluGradFusion. Saved as {onnx_output_file}"
    )


    # unify_gemm_input_dims(onnx_output_file, onnx_output_file)
    # print(f"✅ Successfully unified GEMM input dimensions. Saved as {onnx_output_file}")

    # Compare input and output models to check for any lost nodes/outputs
    print(f"\n🔍 Verifying optimization results...")
    compare_onnx_models(onnx_train_file, onnx_output_file,
                       name1="network_train.onnx", name2="network.onnx")

    fuse_matmul_add_to_gemm(onnx_output_file, onnx_output_file)
    print(
        f"✅ Successfully fused MatMul and Add to Gemm nodes. Saved as {onnx_output_file}"
    )
    process_onnx_model_name_with_type(onnx_output_file, onnx_output_file)
    print(
        f"✅ Successfully processed ONNX model name with type. Saved as {onnx_output_file}"
    )

    add_backward_markers_to_nodes(onnx_output_file, onnx_output_file)
    print(
        f"✅ Successfully added backward markers to nodes. Saved as {onnx_output_file}"
    )

    print_onnx_shapes(onnx_output_file)

def rename_nodes(model_path, output_path):
    """
    Rename nodes in an ONNX model by replacing all characters that are invalid
    for C variable names with underscores.
    
    Args:
        model_path: Path to the input ONNX model
        output_path: Path to save the renamed model
    """
    # Load the model
    model = onnx.load(model_path)

    # Create a map to store original to new name mappings
    name_map = {}

    # Helper function to replace invalid C variable name characters with underscores
    def clean_name(name):
        if name is None:
            return None
        # Replace any character that's not alphanumeric or underscore with underscore
        # Ensure name starts with a letter or underscore (C variable rule)
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        if cleaned and cleaned[0].isdigit():
            cleaned = '_' + cleaned
        return cleaned

    # Process graph inputs
    for input in model.graph.input:
        if input.name:
            new_name = clean_name(input.name)
            name_map[input.name] = new_name
            input.name = new_name

    # Process graph outputs
    for output in model.graph.output:
        if output.name:
            new_name = clean_name(output.name)
            name_map[output.name] = new_name
            output.name = new_name

    # Process initializers
    for initializer in model.graph.initializer:
        if initializer.name:
            new_name = clean_name(initializer.name)
            name_map[initializer.name] = new_name
            initializer.name = new_name

    # Process nodes
    for node in model.graph.node:
        # Rename node name if it exists
        if node.name:
            node.name = clean_name(node.name)

        # Rename node inputs
        for i, input_name in enumerate(node.input):
            if input_name in name_map:
                node.input[i] = name_map[input_name]
            else:
                new_name = clean_name(input_name)
                if new_name != input_name:
                    name_map[input_name] = new_name
                    node.input[i] = new_name

        # Rename node outputs
        for i, output_name in enumerate(node.output):
            if output_name in name_map:
                node.output[i] = name_map[output_name]
            else:
                new_name = clean_name(output_name)
                if new_name != output_name:
                    name_map[output_name] = new_name
                    node.output[i] = new_name

        # Rename attribute names if they contain node names
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                # Handle subgraphs if present (recursive call would be needed)
                pass
            elif attribute.type == onnx.AttributeProto.STRINGS:
                # Handle string attributes that might contain node names
                for i, s in enumerate(attribute.strings):
                    s_str = s.decode('utf-8') if isinstance(s, bytes) else s
                    if s_str in name_map:
                        attribute.strings[i] = name_map[s_str].encode('utf-8')

    # Check for any value_info that might need renaming
    for value_info in model.graph.value_info:
        if value_info.name:
            new_name = clean_name(value_info.name)
            name_map[value_info.name] = new_name
            value_info.name = new_name

    # Save the updated model
    onnx.save(model, output_path)

    return name_map

def randomize_layernorm_params(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            with torch.no_grad():
                module.weight.data = module.weight.data + torch.randn_like(module.weight.data) * 1e-6
                module.bias.data = module.bias.data + torch.randn_like(module.bias.data) * 1e-6

    return model

def fix_shared_initializers_by_node_name(model_path, output_path):
    """为每个bias添加使用它的节点名作为后缀，确保唯一性"""
    
    model = onnx.load(model_path)
    
    print("Fixing shared initializers by adding node names...")
    
    # 分析每个initializer被哪些节点使用
    initializer_node_mapping = {}  # init_name -> [node_names]
    
    for init in model.graph.initializer:
        init_name = init.name
        using_nodes = []
        
        # 查找使用这个initializer的节点
        for node in model.graph.node:
            if init_name in node.input:
                using_nodes.append(node.name)
        
        if len(using_nodes) > 1:
            # 如果被多个节点使用，记录下来
            initializer_node_mapping[init_name] = using_nodes
            print(f"Initializer '{init_name}' is used by {len(using_nodes)} nodes:")
            for node_name in using_nodes:
                print(f"  - {node_name}")
    
    if not initializer_node_mapping:
        print("No shared initializers detected. Nothing to fix.")
        return False
    
    # 为每个共享的initializer的每次使用创建独立副本
    new_initializers = []
    processed_inits = set()
    
    for init in model.graph.initializer:
        if init.name in initializer_node_mapping:
            # 这是一个被多个节点共享的initializer
            print(f"\nProcessing shared initializer: {init.name}")
            processed_inits.add(init.name)
            
            # 为每个使用的节点创建独立副本
            for node_name in initializer_node_mapping[init.name]:
                # 创建新的initializer名称：原名_节点名
                new_name = f"{init.name}_{node_name}"
                
                # 创建initializer的副本
                import copy
                new_init = copy.deepcopy(init)
                new_init.name = new_name
                new_initializers.append(new_init)
                print(f"  Created copy for node '{node_name}': {init.name} -> {new_name}")
        else:
            # 普通initializer，直接保留
            new_initializers.append(init)
    
    # 更新模型的initializer列表
    del model.graph.initializer[:]
    model.graph.initializer.extend(new_initializers)
    
    # 更新节点中的引用
    print("\nUpdating node references...")
    for node in model.graph.node:
        for j, input_name in enumerate(node.input):
            if input_name in processed_inits:
                # 这个节点使用的是共享的initializer，更新为专用版本
                old_name = input_name
                new_name = f"{input_name}_{node.name}"
                node.input[j] = new_name
                print(f"  Node '{node.name}': {old_name} -> {new_name}")
    
    # 保存修复后的模型
    onnx.save(model, output_path)
    print(f"\nFixed model saved to: {output_path}")
    
    return True



def randomize_onnx_initializers(model, seed=None, exclude_patterns=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if exclude_patterns is None:
        exclude_patterns = ["const", "shape", "Constant"]

    graph = model.graph

    modified_count = 0
    zero_count = 0
    skipped_count = 0

    for initializer in graph.initializer:
        # Skip initializers with specific patterns in their names
        if any(pattern in initializer.name for pattern in exclude_patterns):
            skipped_count += 1
            continue

        # Convert initializer to numpy array
        np_array = numpy_helper.to_array(initializer)

        # Check if array contains only zeros
        if np.all(np_array == 0):
            zero_count += 1

            # Determine appropriate scale based on tensor dimension and type
            if np_array.dtype == np.float32 or np_array.dtype == np.float64:
                # Use Kaiming/He initialization for weights
                if len(np_array.shape) > 1:
                    fan_in = np_array.shape[0]
                    scale = np.sqrt(2.0 / fan_in)
                    np_array = np.random.normal(0, scale, np_array.shape).astype(np_array.dtype)
                else:
                    # For bias terms or 1D tensors
                    np_array = np.random.uniform(-0.1, 0.1, np_array.shape).astype(np_array.dtype)
            elif np_array.dtype == np.int64 or np_array.dtype == np.int32:
                # For integer tensors (e.g., indices)
                max_val = min(100, 2**(np_array.itemsize*8 - 1) - 1)  # Avoid overflow
                np_array = np.random.randint(-max_val, max_val, np_array.shape).astype(np_array.dtype)

            # Create new tensor from modified numpy array
            new_tensor = numpy_helper.from_array(np_array, initializer.name)

            # Replace original initializer with new tensor
            initializer.CopyFrom(new_tensor)
            modified_count += 1


    print(f"Randomization complete:")
    print(f"- Total initializers: {len(graph.initializer)}")
    print(f"- Zero initializers found and randomized: {zero_count}")
    print(f"- Skipped initializers (based on patterns): {skipped_count}")
    print(f"- Modified initializers: {modified_count}")

    return model

def type_inference(input_model_path, output_model_path):
    """
    Perform type inference on ONNX model, setting float32 type for variables without explicit types.
    
    Args:
        input_model_path: Input ONNX model path
        output_model_path: Output ONNX model path
    """
    # Load the ONNX model
    model = onnx.load(input_model_path)
    graph = model.graph
    
    # Process input tensors
    for input_tensor in graph.input:
        if not input_tensor.type.tensor_type.elem_type:
            print(f"Setting input variable {input_tensor.name} type to FLOAT")
            input_tensor.type.tensor_type.elem_type = TensorProto.FLOAT
    
    # Process output tensors
    for output_tensor in graph.output:
        if not output_tensor.type.tensor_type.elem_type:
            print(f"Setting output variable {output_tensor.name} type to FLOAT")
            output_tensor.type.tensor_type.elem_type = TensorProto.FLOAT
    
    # Process intermediate variables
    for value_info in graph.value_info:
        if not value_info.type.tensor_type.elem_type:
            print(f"Setting intermediate variable {value_info.name} type to FLOAT")
            value_info.type.tensor_type.elem_type = TensorProto.FLOAT
    
    # Check for any tensors mentioned in nodes but missing type info
    processed_tensors = set([tensor.name for tensor in graph.input] + 
                           [tensor.name for tensor in graph.output] + 
                           [tensor.name for tensor in graph.value_info])
    
    missing_tensors = set()
    for node in graph.node:
        for input_name in node.input:
            if input_name not in processed_tensors and input_name:
                missing_tensors.add(input_name)
        for output_name in node.output:
            if output_name not in processed_tensors and output_name:
                missing_tensors.add(output_name)
    
    # Add missing tensors to value_info with FLOAT type (keeping existing shapes)
    for tensor_name in missing_tensors:
        # Create a basic ValueInfo for the tensor (shape will be inferred by ONNX)
        tensor_value_info = onnx.helper.make_tensor_value_info(
            name=tensor_name,
            elem_type=TensorProto.FLOAT,
            shape=None  # Let ONNX infer the shape
        )
        graph.value_info.append(tensor_value_info)
        print(f"Added missing tensor {tensor_name} with FLOAT type")
    
    # Save the modified model
    onnx.save(model, output_model_path)
    print(f"Successfully saved type-inferred model to {output_model_path}")

def ensure_all_tensor_shapes(model_path, output_path):
    """
    Ensure all tensors in the ONNX model have shape annotations.
    This function infers shapes for missing tensors, especially gradient intermediates.
    """
    print("🔍 Checking and fixing tensor shapes...")
    
    # Load the model
    onnx_model = onnx.load(model_path)
    graph = onnx_model.graph
    
    # First, run standard shape inference
    try:
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    except Exception as e:
        print(f"⚠️  Standard shape inference warning: {e}")
    
    # Collect all tensor shapes from value_info
    known_shapes = {}
    
    # Get shapes from inputs
    for input_tensor in graph.input:
        if input_tensor.type.tensor_type.HasField('shape'):
            shape = [dim.dim_value if dim.HasField('dim_value') else -1 
                    for dim in input_tensor.type.tensor_type.shape.dim]
            known_shapes[input_tensor.name] = shape
    
    # Get shapes from outputs
    for output_tensor in graph.output:
        if output_tensor.type.tensor_type.HasField('shape'):
            shape = [dim.dim_value if dim.HasField('dim_value') else -1 
                    for dim in output_tensor.type.tensor_type.shape.dim]
            known_shapes[output_tensor.name] = shape
    
    # Get shapes from value_info
    for value_info in graph.value_info:
        if value_info.type.tensor_type.HasField('shape'):
            shape = [dim.dim_value if dim.HasField('dim_value') else -1 
                    for dim in value_info.type.tensor_type.shape.dim]
            known_shapes[value_info.name] = shape
    
    # Get shapes from initializers
    for init in graph.initializer:
        known_shapes[init.name] = list(init.dims)
    
    # Find all tensor names used in the graph
    all_tensor_names = set()
    for node in graph.node:
        all_tensor_names.update(node.input)
        all_tensor_names.update(node.output)
    
    # Find missing tensors
    missing_tensors = all_tensor_names - set(known_shapes.keys())
    
    if missing_tensors:
        print(f"⚠️  Found {len(missing_tensors)} tensors with missing shapes")
        print(f"Missing tensors: {list(missing_tensors)[:10]}...")  # Show first 10
        
        # Try to infer shapes from connected nodes
        for tensor_name in missing_tensors:
            # Find nodes that produce this tensor
            producer_nodes = [n for n in graph.node if tensor_name in n.output]
            consumer_nodes = [n for n in graph.node if tensor_name in n.input]
            
            inferred_shape = None
            
            # Try to infer from producer
            if producer_nodes:
                producer = producer_nodes[0]
                
                # For gradient accumulation nodes, use input shape
                if 'AccumulateGrad' in producer.op_type or 'Gradient' in producer.op_type:
                    for input_name in producer.input:
                        if input_name in known_shapes:
                            inferred_shape = known_shapes[input_name]
                            break
                
                # For element-wise ops, inherit from input
                elif producer.op_type in ['Add', 'Sub', 'Mul', 'Div', 'Relu', 'Identity']:
                    for input_name in producer.input:
                        if input_name in known_shapes:
                            inferred_shape = known_shapes[input_name]
                            break
            
            # Try to infer from consumer
            if inferred_shape is None and consumer_nodes:
                consumer = consumer_nodes[0]
                for input_name in consumer.input:
                    if input_name != tensor_name and input_name in known_shapes:
                        inferred_shape = known_shapes[input_name]
                        break
            
            # If we found a shape, add it to value_info
            if inferred_shape is not None:
                # Create value_info with inferred shape
                tensor_type = onnx.TensorProto.FLOAT  # Default to FLOAT
                value_info = helper.make_tensor_value_info(
                    tensor_name,
                    tensor_type,
                    inferred_shape
                )
                graph.value_info.append(value_info)
                known_shapes[tensor_name] = inferred_shape
                print(f"✅ Inferred shape for {tensor_name}: {inferred_shape}")
    
    # Final shape inference pass
    try:
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        print("✅ Final shape inference completed")
    except Exception as e:
        print(f"⚠️  Final shape inference warning: {e}")
    
    # Verify all tensors now have shapes
    remaining_missing = []
    for value_info in graph.value_info:
        if not value_info.type.tensor_type.HasField('shape'):
            remaining_missing.append(value_info.name)
    
    if remaining_missing:
        print(f"⚠️  Warning: {len(remaining_missing)} tensors still missing shapes")
        print(f"Remaining missing: {remaining_missing[:10]}")
    else:
        print("✅ All tensors have shape annotations")
    
    # Save the model
    onnx.save(onnx_model, output_path)
    print(f"✅ Saved model with complete shapes to {output_path}")
    
    return onnx_model