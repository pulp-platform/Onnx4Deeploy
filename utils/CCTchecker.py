import copy
import numpy as np
import onnx
import onnxruntime as ort
import argparse
from onnx import TensorProto, helper
import sys
import os

def trim_onnx_model(model_path, output_node_name, save_path="trimmed_network.onnx"):
    """Extract a subgraph from the model up to and including the target node."""
    model = onnx.load(model_path)
    nodes = model.graph.node
    target_node_idx = None

    # Find the target node
    for idx, node in enumerate(nodes):
        if node.name == output_node_name:
            target_node_idx = idx
            break

    if target_node_idx is None:
        raise ValueError(f"Cannot find {output_node_name} in the model.")

    # Get the outputs of the target node
    target_node = nodes[target_node_idx]
    target_outputs = list(target_node.output)

    # Get original model inputs
    input_names = [inp.name for inp in model.graph.input]

    print(f"\nExtracting subgraph:")
    print(f"  Input names: {input_names}")
    print(f"  Output names: {target_outputs}")

    # Use ONNX's extract_model utility to properly handle dependencies
    try:
        trimmed_model = onnx.utils.extract_model(
            model_path,
            save_path,
            input_names=input_names,
            output_names=target_outputs
        )
        print(f"Saved: {save_path}")
        return trimmed_model
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        print("Falling back to manual extraction...")

        # Fallback: manual extraction with proper dependency handling
        trimmed_model = _manual_extract(model, target_node_idx, save_path)
        return trimmed_model

def _manual_extract(model, target_node_idx, save_path):
    """Manually extract subgraph by keeping only nodes needed for target output."""
    trimmed_model = copy.deepcopy(model)
    nodes = list(model.graph.node)
    target_node = nodes[target_node_idx]

    # Build dependency graph
    required_nodes = set()
    to_process = [target_node_idx]

    while to_process:
        current_idx = to_process.pop()
        if current_idx in required_nodes:
            continue
        required_nodes.add(current_idx)

        current_node = nodes[current_idx]
        # Find nodes that produce inputs for current node
        for input_name in current_node.input:
            for idx, node in enumerate(nodes):
                if idx < current_idx and input_name in node.output:
                    if idx not in required_nodes:
                        to_process.append(idx)

    # Keep only required nodes in order
    new_nodes = [nodes[i] for i in sorted(required_nodes)]

    # Update graph
    trimmed_model.graph.ClearField('node')
    trimmed_model.graph.node.extend(new_nodes)

    # Set new outputs
    target_outputs = list(target_node.output)
    new_outputs = []
    for output in target_outputs:
        tensor_type = helper.make_tensor_type_proto(
            elem_type=TensorProto.FLOAT,
            shape=None
        )
        output_value_info = helper.make_value_info(name=output, type_proto=tensor_type)
        new_outputs.append(output_value_info)

    trimmed_model.graph.ClearField('output')
    trimmed_model.graph.output.extend(new_outputs)

    try:
        onnx.checker.check_model(trimmed_model)
    except Exception as e:
        print(f"Warning: {str(e)}")
        print("Trying to save model anyway...")

    onnx.save(trimmed_model, save_path)
    print(f"Saved: {save_path}")

    return trimmed_model

def infer_model(model_path, input_data):
    """Run inference on the model, handling both standard and training graphs."""
    model = onnx.load(model_path)
    input_names = [input.name for input in model.graph.input]

    # Check if this is a training graph
    has_training_ops = any(
        op.domain in ['ai.onnx.training', 'ai.onnx.preview.training', 'com.microsoft']
        for op in model.opset_import
    )

    if has_training_ops:
        print("\nWARNING: This model contains training-specific operators.")
        print("Standard ONNX Runtime may not support all operations.")
        print("Attempting inference anyway...\n")

    # Try to use the model directly first
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        ort_inputs = {name: input_data[name] for name in input_names if name in input_data}

        # Add missing inputs with dummy values if needed
        for session_input in session.get_inputs():
            if session_input.name not in ort_inputs:
                print(f"WARNING: Missing input '{session_input.name}', using zeros")
                shape = session_input.shape
                shape = [dim if isinstance(dim, int) else 1 for dim in shape]
                ort_inputs[session_input.name] = np.zeros(shape, dtype=np.float32)

        outputs = session.run(None, ort_inputs)
        output_names = [output.name for output in session.get_outputs()]
        return dict(zip(output_names, outputs))

    except Exception as e:
        print(f"\nError during inference: {str(e)}")
        print("\nThis is likely because:")
        print("1. The model uses training-specific operators not supported in standard ONNX Runtime")
        print("2. Required inputs are missing from the input data")
        print("\nFor training graphs, consider:")
        print("- Using network_infer.onnx for forward pass nodes")
        print("- Using ONNX Runtime Training API for gradient nodes")
        print("- Extracting intermediate outputs during actual training")
        raise

def add_intermediate_outputs_to_model(model_path, output_path, intermediate_node_names):
    """Add intermediate node outputs as model outputs for debugging.

    This allows you to see intermediate values during training by adding
    them as additional outputs to the model.

    Args:
        model_path: Path to the original ONNX model
        output_path: Path to save the modified model
        intermediate_node_names: List of node names whose outputs should be added
    """
    model = onnx.load(model_path)
    nodes = {node.name: node for node in model.graph.node}

    # Collect outputs to add
    outputs_to_add = []
    for node_name in intermediate_node_names:
        if node_name not in nodes:
            print(f"Warning: Node '{node_name}' not found in model")
            continue

        node = nodes[node_name]
        for output in node.output:
            # Check if already an output
            existing_outputs = [o.name for o in model.graph.output]
            if output not in existing_outputs:
                # Create output value info
                tensor_type = helper.make_tensor_type_proto(
                    elem_type=TensorProto.FLOAT,
                    shape=None
                )
                output_value_info = helper.make_value_info(name=output, type_proto=tensor_type)
                outputs_to_add.append(output_value_info)
                print(f"Adding output: {output} (from node {node_name})")

    # Add new outputs
    model.graph.output.extend(outputs_to_add)

    # Save modified model
    onnx.save(model, output_path)
    print(f"\nSaved model with {len(outputs_to_add)} additional outputs to: {output_path}")
    return model

def create_training_checkpoint_model(train_model_path, output_path, checkpoint_nodes=None):
    """Create a modified training model that outputs intermediate values at checkpoint nodes.

    This is useful for debugging training by examining intermediate gradients and activations.

    Args:
        train_model_path: Path to the training ONNX model
        output_path: Path to save the checkpoint model
        checkpoint_nodes: List of node names to checkpoint (default: all backward nodes)
    """
    model = onnx.load(train_model_path)

    if checkpoint_nodes is None:
        # Default: checkpoint all backward nodes
        checkpoint_nodes = [node.name for node in model.graph.node
                          if is_backward_node(node.name)]
        print(f"Auto-selected {len(checkpoint_nodes)} backward nodes for checkpointing")

    return add_intermediate_outputs_to_model(train_model_path, output_path, checkpoint_nodes)

def generate_pytorch_instrumentation_code(model_path, target_node_name, output_file=None):
    """Generate PyTorch training code instrumentation to capture intermediate values.

    This generates code snippets you can insert into your PyTorch training loop
    to capture and save intermediate tensor values.

    Args:
        model_path: Path to the ONNX model
        target_node_name: Name of the target node
        output_file: Optional file to save the code (default: print to stdout)
    """
    model = onnx.load(model_path)
    nodes = list(model.graph.node)

    target_idx = None
    for idx, node in enumerate(nodes):
        if node.name == target_node_name:
            target_idx = idx
            break

    if target_idx is None:
        print(f"Error: Node '{target_node_name}' not found")
        return

    target_node = nodes[target_idx]
    is_backward = is_backward_node(target_node_name)

    code = f"""
# Instrumentation code for node: {target_node_name}
# Node type: {'BACKWARD/GRADIENT' if is_backward else 'FORWARD'}
# Inputs: {', '.join(target_node.input)}
# Outputs: {', '.join(target_node.output)}

# Add this to your PyTorch training code:

import torch
import numpy as np

# Dictionary to store intermediate values
intermediate_values = {{}}

"""

    if is_backward:
        code += f"""
# For backward pass, register a hook on the corresponding forward tensor
# You need to identify the corresponding forward tensor for: {target_node.input[0] if target_node.input else 'N/A'}

def save_gradient_hook(name):
    def hook(grad):
        intermediate_values[name] = grad.detach().cpu().numpy()
        print(f"Captured gradient at {{name}}: shape={{grad.shape}}")
        return grad
    return hook

# Example: Register hook on a forward tensor
# Replace 'your_tensor' with the actual tensor variable
# your_tensor.register_hook(save_gradient_hook('{target_node_name}'))

# After backward pass:
# np.savez('intermediate_grads.npz', **intermediate_values)
"""
    else:
        code += f"""
# For forward pass, save the activation directly
# You need to identify the corresponding tensor in your model for: {target_node.output[0] if target_node.output else 'N/A'}

# Example: After computing the activation
# Replace 'your_activation' with the actual tensor variable
# intermediate_values['{target_node_name}'] = your_activation.detach().cpu().numpy()

# After forward pass:
# np.savez('intermediate_activations.npz', **intermediate_values)
"""

    code += f"""
# To compare with ONNX model output:
# 1. Save the intermediate value as shown above
# 2. Run CCTchecker.py to get ONNX intermediate output:
#    python CCTchecker.py {target_node_name} {'--use-infer-model' if not is_backward else '-y'}
# 3. Compare the saved numpy arrays
"""

    if output_file:
        with open(output_file, 'w') as f:
            f.write(code)
        print(f"Instrumentation code saved to: {output_file}")
    else:
        print(code)

    return code

def is_backward_node(node_name):
    """Check if a node is a backward/gradient node."""
    backward_keywords = ['Grad', '_backward', 'sgd', 'adam', 'optimizer']
    return any(keyword in node_name for keyword in backward_keywords)

def suggest_model_path(node_name, base_dir):
    """Suggest the appropriate model file based on the node type."""
    if is_backward_node(node_name):
        # For backward nodes, use the training model
        return f"{base_dir}/network.onnx"
    else:
        # For forward nodes, prefer inference model if available
        infer_path = f"{base_dir}/network_infer.onnx"
        import os
        if os.path.exists(infer_path):
            return infer_path
        return f"{base_dir}/network.onnx"

def main(output_node_name, model_path="/app/Onnx4Deeploy/Tests/Models/MI-BMInet/onnx/MI_BMINet_train_C8_T2000_F1_8_D2_N4/network.onnx", input_path="/app/Onnx4Deeploy/Tests/Models/MI-BMInet/onnx/MI_BMINet_train_C8_T2000_F1_8_D2_N4/inputs.npz", save_path="trimmed_network.onnx", skip_confirm=False):

    print(f"Target node: {output_node_name}")
    print(f"Is backward node: {is_backward_node(output_node_name)}")

    # Load and display all nodes
    model = onnx.load(model_path)
    print(f"\nAll nodes in {model_path.split('/')[-1]}:")
    for node in model.graph.node:
        prefix = "  [BACKWARD]" if is_backward_node(node.name) else "  [FORWARD] "
        print(f"{prefix} {node.name}")

    # Check if target node exists
    node_names = [node.name for node in model.graph.node]
    if output_node_name not in node_names:
        print(f"\nERROR: Node '{output_node_name}' not found in model!")
        print("\nAvailable nodes:")
        for name in node_names:
            print(f"  {name}")
        return

    # Warn if trying to extract backward node
    if is_backward_node(output_node_name) and not skip_confirm:
        print("\n" + "="*70)
        print("WARNING: You are trying to extract a BACKWARD/GRADIENT node!")
        print("="*70)
        print("Backward nodes require:")
        print("  1. Forward pass outputs as inputs")
        print("  2. ONNX Runtime Training (not standard inference)")
        print("\nRecommendations:")
        print("  - For debugging forward pass: Use network_infer.onnx instead")
        print("  - For backward pass: Instrument training code to save intermediates")
        print("  - Alternatively: Use ONNX Runtime Training Session")
        print("="*70)
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    elif is_backward_node(output_node_name):
        print("\nWARNING: Extracting backward node (confirmation skipped with --yes)")
        print("This will likely fail during inference!")

    # Extract subgraph
    print(f"\nExtracting subgraph to '{save_path}'...")
    trimmed_model = trim_onnx_model(model_path, output_node_name, save_path)

    # Load input data
    input_data = np.load(input_path)
    print(f"\nLoaded inputs: {list(input_data.keys())}")

    # Run inference
    try:
        result_dict = infer_model(save_path, input_data)

        # Display results
        np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)

        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)

        print("\nInput:")
        if 'input' in input_data:
            print(f"  Shape: {input_data['input'].shape}")
            print(f"  Data:\n{input_data['input']}")
        else:
            print(f"  Available: {list(input_data.keys())}")

        print("\nOutput:")
        for name, output in result_dict.items():
            print(f"  {name}:")
            print(f"    Shape: {output.shape}")
            print(f"    Data:\n{output}")

    except Exception as e:
        print(f"\nFailed to run inference: {e}")
        print("\nThe extracted model was saved but cannot be executed.")
        print("See error messages above for details.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim ONNX model and run inference.")
    parser.add_argument("output_node", type=str, nargs='?', default=None,
                       help="Name of the output node.")
    parser.add_argument("--model", type=str,
                       default="/app/Onnx4Deeploy/Tests/Models/MI-BMInet/onnx/MI_BMINet_train_C8_T2000_F1_8_D2_N4/network.onnx",
                       help="Path to the ONNX model")
    parser.add_argument("--input", type=str,
                       default="/app/Onnx4Deeploy/Tests/Models/MI-BMInet/onnx/MI_BMINet_train_C8_T2000_F1_8_D2_N4/inputs.npz",
                       help="Path to the input data (.npz)")
    parser.add_argument("--output", type=str, default="trimmed_network.onnx",
                       help="Path to save the trimmed model")
    parser.add_argument("--use-infer-model", action="store_true",
                       help="Use network_infer.onnx for forward nodes (if available)")
    parser.add_argument("-y", "--yes", action="store_true",
                       help="Skip confirmation prompts")
    parser.add_argument("--list-nodes", action="store_true",
                       help="List all nodes and exit")
    parser.add_argument("--add-outputs", type=str, nargs='+',
                       help="Add intermediate node outputs to model (saves modified model)")
    parser.add_argument("--create-checkpoint-model", type=str,
                       help="Create a checkpoint model with intermediate outputs at specified path")
    parser.add_argument("--generate-instrumentation", action="store_true",
                       help="Generate PyTorch training instrumentation code for the target node")
    parser.add_argument("--instrumentation-output", type=str,
                       help="File to save instrumentation code (default: print to stdout)")

    args = parser.parse_args()

    # Handle list nodes
    if args.list_nodes:
        model = onnx.load(args.model)
        print(f"Nodes in {args.model}:")
        for node in model.graph.node:
            prefix = "[BACKWARD]" if is_backward_node(node.name) else "[FORWARD] "
            print(f"  {prefix} {node.name}")
        exit(0)

    # Handle add outputs
    if args.add_outputs:
        if not args.output:
            parser.error("--output is required when using --add-outputs")
        print(f"Adding intermediate outputs to model...")
        add_intermediate_outputs_to_model(args.model, args.output, args.add_outputs)
        exit(0)

    # Handle create checkpoint model
    if args.create_checkpoint_model:
        print(f"Creating checkpoint model...")
        create_training_checkpoint_model(args.model, args.create_checkpoint_model)
        exit(0)

    # Check if output_node is provided
    if args.output_node is None:
        parser.error("output_node is required (unless using --list-nodes, --add-outputs, or --create-checkpoint-model)")

    # Handle generate instrumentation
    if args.generate_instrumentation:
        print(f"Generating instrumentation code for node: {args.output_node}\n")
        generate_pytorch_instrumentation_code(
            args.model,
            args.output_node,
            args.instrumentation_output
        )
        exit(0)

    # Auto-select inference model for forward nodes if requested
    model_path = args.model
    if args.use_infer_model and not is_backward_node(args.output_node):
        import os
        base_dir = os.path.dirname(args.model)
        infer_path = os.path.join(base_dir, "network_infer.onnx")
        if os.path.exists(infer_path):
            print(f"Using inference model: {infer_path}")
            model_path = infer_path

    main(args.output_node, model_path, args.input, args.output, skip_confirm=args.yes)
