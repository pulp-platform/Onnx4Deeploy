import numpy as np
import onnxruntime as ort
import onnx
import os
import torch
import torchvision
from torchvision import transforms
from .utils import *

def preprocess_mnist(batch_size, image_size):
    """
    Preprocess MNIST dataset with configurable image size.
    
    Args:
        batch_size: Number of images to process
        image_size: Size to resize images to (will be used as both height and width)
        
    Returns:
        Tuple of (images, labels) as numpy arrays
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    
    dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    indices = np.random.choice(len(dataset), batch_size, replace=False)
    images = torch.stack([dataset[i][0] for i in indices])
    labels = np.array([dataset[i][1] for i in indices], dtype=np.int64)
    
    return images.numpy(), labels

def run_original_onnx_model(input_data, labels, model_path):
    """
    Run inference on original ONNX model to get gradients.
    
    Args:
        input_data: Input data for the model
        labels: Labels for the model
        model_path: Path to the original ONNX model (without SGD)
        
    Returns:
        Dictionary of model outputs (gradients)
    """
    ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    
    output_names = [output.name for output in ort_session.get_outputs()]
    print(f"Model has {len(output_names)} outputs: {output_names}")
    
    outputs = ort_session.run(None, {"input": input_data, "labels": labels})
    
    output_dict = {}
    for i, name in enumerate(output_names):
        output_dict[name] = outputs[i]
    
    return output_dict

def get_initializer_from_onnx(model_path, initializer_name):
    """
    Extract initializer tensor from ONNX model.
    
    Args:
        model_path: Path to the ONNX model
        initializer_name: Name of the initializer to extract
        
    Returns:
        Numpy array of the initializer tensor
    """
    model = onnx.load(model_path)
    for initializer in model.graph.initializer:
        if initializer.name == initializer_name:
            # Convert ONNX tensor to numpy array
            from onnx import numpy_helper
            return numpy_helper.to_array(initializer)
    
    raise ValueError(f"Initializer {initializer_name} not found in model")

def apply_sgd_update(weight, gradient, learning_rate=0.01):
    """
    Manually apply SGD update to weights.
    
    Args:
        weight: Current weight tensor
        gradient: Gradient tensor
        learning_rate: Learning rate for SGD
        
    Returns:
        Updated weight tensor
    """
    return weight - learning_rate * gradient

def create_test_input_output():
    """
    Create test input and output files with manual SGD implementation.
    Updated to handle all parameters with gradients, not just fc weights and bias.
    """
    # Load config
    config = load_config()
    if isinstance(config, tuple):
        pretrained, img_size, num_classes, embedding_dim, num_heads, num_layers, batch_size, opset_version = config
    else:
        img_size = config.get("img_size", 16)
        batch_size = config.get("batch_size", 8)
        embedding_dim = config.get("embedding_dim", 384)
        num_heads = config.get("num_heads", 6)
        num_layers = config.get("num_layers", 7)
    
    print(f"Using image size: {img_size}, batch size: {batch_size}")
    
    folder_name = f"CCT_train_{img_size}_{embedding_dim}_{num_heads}_{num_layers}"
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder_path = os.path.join(base_dir, "onnx", folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Path to original training network
    network_path = os.path.join(folder_path, "network_train.onnx")
    input_path = os.path.join(folder_path, "inputs.npz")
    output_path = os.path.join(folder_path, "outputs.npz")
    
    if not os.path.exists(network_path):
        raise FileNotFoundError(f"ONNX model file not found: {network_path}")
    
    # Create input data with the specified image size
    input_data, labels = preprocess_mnist(batch_size, img_size)
    np.savez(input_path, input=input_data, labels=labels)
    print(f"✅ Input saved to inputs.npz (image size: {img_size}x{img_size}, batch size: {batch_size})")
    
    # Run the original model to get gradients
    outputs_dict = run_original_onnx_model(input_data, labels, model_path=network_path)
    
    # Get all output names from the model
    output_names = list(outputs_dict.keys())
    
    # Load the model to get all parameter names
    model = onnx.load(network_path)
    param_names = [init.name for init in model.graph.initializer]
    
    # Special mappings for known gradient patterns
    special_mappings = {
        # Format: 'gradient_pattern': 'parameter_name'
        "classifier_fc_gemm_grad_dc_reduced": "classifier_fc_bias",
    }
    
    # Find all gradient outputs and map them to parameters
    grad_to_param_map = {}
    
    for output_name in output_names:
        if "grad" in output_name.lower():
            # Try various strategies to find the corresponding parameter
            
            # 1. Check special mappings first
            param_found = False
            for pattern, param in special_mappings.items():
                if pattern in output_name.lower():
                    if param in param_names:
                        grad_to_param_map[param] = output_name
                        param_found = True
                        print(f"Mapped gradient {output_name} to parameter {param} (special mapping)")
                        break
            
            # 2. Direct match (e.g., param_name_grad -> param_name)
            if not param_found:
                potential_param = re.sub(r'_grad.*$', '', output_name, flags=re.IGNORECASE)
                if potential_param in param_names:
                    grad_to_param_map[potential_param] = output_name
                    param_found = True
                    print(f"Mapped gradient {output_name} to parameter {potential_param} (direct match)")
            
            # 3. Check for node prefix
            if not param_found and "node_" in output_name:
                no_prefix = re.sub(r'^node_\d+_', '', output_name)
                potential_param = re.sub(r'_grad.*$', '', no_prefix, flags=re.IGNORECASE)
                if potential_param in param_names:
                    grad_to_param_map[potential_param] = output_name
                    param_found = True
                    print(f"Mapped gradient {output_name} to parameter {potential_param} (node prefix removed)")
            
            # 4. For gemm gradients that might be bias gradients
            if not param_found and "gemm_grad" in output_name.lower():
                base_part = output_name.split("_Gemm_Grad")[0]
                if "node_" in base_part:
                    base_part = re.sub(r'^node_\d+_', '', base_part)
                
                bias_name = f"{base_part}_bias"
                if bias_name in param_names:
                    grad_to_param_map[bias_name] = output_name
                    param_found = True
                    print(f"Mapped gradient {output_name} to parameter {bias_name} (gemm bias match)")
            
            # 5. Substring match as last resort
            if not param_found:
                for param in param_names:
                    if len(param) > 3 and param.lower() in output_name.lower():
                        grad_to_param_map[param] = output_name
                        param_found = True
                        print(f"Mapped gradient {output_name} to parameter {param} (substring match)")
                        break
            
            if not param_found:
                print(f"⚠️ Could not find matching parameter for gradient: {output_name}")
    
    # Apply SGD manually to all mapped parameters
    learning_rate = load_train_config()
    
    sgd_outputs = {}
    
    for param_name, grad_name in grad_to_param_map.items():
        try:
            # Get original parameter
            param_value = get_initializer_from_onnx(network_path, param_name)
            
            # Get gradient
            grad_value = outputs_dict[grad_name]
            
            # Apply SGD update
            param_updated = apply_sgd_update(param_value, grad_value, learning_rate)
            
            # Store updated parameter
            updated_name = f"{param_name}_updated"
            sgd_outputs[updated_name] = param_updated
            
            print(f"✅ Successfully applied SGD update to {param_name}")
            print(f"  Original shape: {param_value.shape}, Gradient shape: {grad_value.shape}")
            print(f"  Updated parameter saved as: {updated_name}")
            
        except Exception as e:
            print(f"❌ Error updating {param_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save updated parameters to output file
    if sgd_outputs:
        np.savez(output_path, **sgd_outputs)
        print(f"✅ Updated parameters saved to {output_path}")
        
        # Print output shapes
        print("Final output shapes:")
        for name, arr in sgd_outputs.items():
            print(f"  {name}: {arr.shape}")
    else:
        print("❌ No parameters were updated. Cannot save outputs.")
        
    return sgd_outputs

def check_network_consistency(original_model_path, optimized_model_path, input_data, tolerance=1e-4):
    """
    Check if two ONNX models produce consistent outputs for the same input data.
    
    Args:
        original_model_path: Path to the original ONNX model
        optimized_model_path: Path to the optimized ONNX model
        input_data: Dictionary of input data to use for inference
        tolerance: Maximum allowable difference between outputs
        
    Returns:
        bool: True if models are consistent, False otherwise
    """
    # Load and run original model
    orig_session = ort.InferenceSession(original_model_path, providers=["CPUExecutionProvider"])
    orig_outputs = orig_session.run(None, input_data)
    
    # Load and run optimized model
    opt_session = ort.InferenceSession(optimized_model_path, providers=["CPUExecutionProvider"])
    opt_outputs = opt_session.run(None, input_data)
    
    # Check if both models have same number of outputs
    if len(orig_outputs) != len(opt_outputs):
        print(f"❌ Models have different number of outputs: Original {len(orig_outputs)}, Optimized {len(opt_outputs)}")
        return False
    
    # Get output names for better error reporting
    orig_output_names = [output.name for output in orig_session.get_outputs()]
    opt_output_names = [output.name for output in opt_session.get_outputs()]
    
    # Print output names for debugging
    print(f"Original model outputs: {orig_output_names}")
    print(f"Optimized model outputs: {opt_output_names}")
    
    # Check each output tensor
    all_consistent = True
    for i, (orig, opt) in enumerate(zip(orig_outputs, opt_outputs)):
        orig_name = orig_output_names[i] if i < len(orig_output_names) else f"output_{i}"
        opt_name = opt_output_names[i] if i < len(opt_output_names) else f"output_{i}"
        
        # Check shape
        if orig.shape != opt.shape:
            print(f"❌ Output shape mismatch for output {i} ({orig_name} vs {opt_name}):")
            print(f"  Original: {orig.shape}, Optimized: {opt.shape}")
            all_consistent = False
            continue
        
        # Check values
        max_diff = np.max(np.abs(orig - opt))
        if max_diff > tolerance:
            print(f"❌ Output value mismatch for output {i} ({orig_name} vs {opt_name}):")
            print(f"  Max difference: {max_diff}, Tolerance: {tolerance}")
            all_consistent = False
        else:
            print(f"✅ Output {i} ({orig_name} vs {opt_name}) consistent, Max diff: {max_diff}")
    
    if all_consistent:
        print(f"✅ All outputs consistent within tolerance {tolerance}")
    
    return all_consistent

def check_graph_structure_consistency(original_model_path, optimized_model_path):
    """
    Check if two ONNX models have consistent graph structures.
    
    Args:
        original_model_path: Path to the original ONNX model
        optimized_model_path: Path to the optimized ONNX model
        
    Returns:
        bool: True if graph structures are consistent, False otherwise
    """
    # Load models
    orig_model = onnx.load(original_model_path)
    opt_model = onnx.load(optimized_model_path)
    
    # Check input consistency
    orig_inputs = {input.name: input for input in orig_model.graph.input}
    opt_inputs = {input.name: input for input in opt_model.graph.input}
    
    if len(orig_inputs) != len(opt_inputs):
        print(f"❌ Models have different number of inputs: Original {len(orig_inputs)}, Optimized {len(opt_inputs)}")
        return False
    
    # Check if optimized model has all original inputs
    missing_inputs = [name for name in orig_inputs if name not in opt_inputs]
    if missing_inputs:
        print(f"❌ Optimized model is missing inputs: {missing_inputs}")
        return False
    
    # Check initializer consistency
    orig_initializers = {init.name: init for init in orig_model.graph.initializer}
    opt_initializers = {init.name: init for init in opt_model.graph.initializer}
    
    # Check if optimized model has all original initializers (except SGD ones)
    missing_initializers = []
    for name in orig_initializers:
        if name not in opt_initializers and not name.endswith("_sgd"):
            missing_initializers.append(name)
    
    if missing_initializers:
        print(f"❌ Optimized model is missing initializers: {missing_initializers}")
        return False
    
    # Count node types for both models
    def count_node_types(model):
        counter = {}
        for node in model.graph.node:
            op_type = node.op_type
            counter[op_type] = counter.get(op_type, 0) + 1
        return counter
    
    orig_node_types = count_node_types(orig_model)
    opt_node_types = count_node_types(opt_model)
    
    # Print node type counts for comparison
    print("Node type counts:")
    print("Original model:")
    for op_type, count in sorted(orig_node_types.items()):
        print(f"  {op_type}: {count}")
    
    print("Optimized model:")
    for op_type, count in sorted(opt_node_types.items()):
        print(f"  {op_type}: {count}")
    
    # Check if all original op types exist in optimized model
    missing_op_types = [op for op in orig_node_types if op not in opt_node_types]
    if missing_op_types:
        print(f"❌ Optimized model is missing operator types: {missing_op_types}")
        return False
    
    # Additional op types in optimized model (may be OK if related to SGD)
    new_op_types = [op for op in opt_node_types if op not in orig_node_types]
    if new_op_types:
        print(f"⚠️ Optimized model has new operator types: {new_op_types}")
        # This might be OK if the new ops are related to SGD updates
    
    print("✅ Basic graph structure consistency check passed")
    return True
