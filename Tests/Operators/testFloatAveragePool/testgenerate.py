import onnx
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper, shape_inference

def generate_averagepool_onnx_and_data(save_path=None):
    """ Generate ONNX model for a single AveragePool operator with test data """
    
    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")
    
    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config.get("sgd", {})
    weight_shape = tuple(model_config.get("weight_shape", (10, 8)))
    learning_rate = float(model_config.get("learning_rate", 0.01))
    
    print(f"Generating SGD model with:")
    print(f"  Weight shape: {weight_shape}")
    print(f"  Learning rate: {learning_rate}")
    
    # Set default save path if not provided
    base_path = save_path if save_path else os.path.join(script_dir, "onnx", "sgd_test")
    
    # Define standard filenames
    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")
    
    # Ensure the save directory exists
    os.makedirs(base_path, exist_ok=True)
    
    # Generate random input data
    # Initialize weights and gradient
    weights = np.random.randn(*weight_shape).astype(np.float32)
    gradient = np.random.randn(*weight_shape).astype(np.float32) * 0.1
    
    # Save input data
    np.savez(input_file, weights=weights, gradient=gradient)
    print(f"✅ Input data saved to {input_file}")
    
    # Define ONNX tensors for the model
    weights_tensor = helper.make_tensor_value_info("weights", TensorProto.FLOAT, weight_shape)
    gradient_tensor = helper.make_tensor_value_info("gradient", TensorProto.FLOAT, weight_shape)
    
    # For a custom op like SGD, we should explicitly set the output shape
    # since standard ONNX shape inference won't work
    updated_weights_tensor = helper.make_tensor_value_info(
        "updated_weights", 
        TensorProto.FLOAT, 
        weight_shape  # Explicitly set the shape to match input weights
    )
    
    # Create SGD node with learning_rate as attribute
    sgd_node = helper.make_node(
        "SGD",  # Op name
        inputs=["weights", "gradient"],
        outputs=["updated_weights"],
        name="sgd_node",
        domain="com.mydomain",  # Use custom domain for custom operator
        lr=float(learning_rate)  # Set learning rate as attribute
    )
    
    # Create graph with the SGD node
    graph_def = helper.make_graph(
        [sgd_node], 
        "sgd_graph",
        [weights_tensor, gradient_tensor],
        [updated_weights_tensor]
    )
    
    # Include both standard and custom domains
    model_def = helper.make_model(
        graph_def, 
        producer_name="sgd_model",
        opset_imports=[
            helper.make_opsetid("", 13),  # Standard ONNX domain
            helper.make_opsetid("com.mydomain", 1)  # Custom domain for SGD
        ]
    )
    
    # Skip standard shape inference since SGD is a custom operator
    # Instead, manually set the output shape to match the input shape
    # Extract output shape info from the graph
    output_tensor = model_def.graph.output[0]
    
    # Get the shape from input weights
    input_tensor = model_def.graph.input[0]  # weights tensor
    if input_tensor.type.tensor_type.shape.dim:
        # Copy shape from input weights to output
        output_tensor.type.tensor_type.shape.CopyFrom(input_tensor.type.tensor_type.shape)
        print("✅ Manually set output shape to match input weights shape")
    else:
        print("⚠️ Could not determine input shape for manual shape setting")
    
    # Save ONNX model
    onnx.save(model_def, onnx_file)
    print(f"✅ ONNX model saved to {onnx_file}")
    
    # Compute the expected output manually
    updated_weights = weights - learning_rate * gradient
    
    # Save output data
    np.savez(output_file, updated_weights=updated_weights)
    
    print(f"✅ Expected output data saved to {output_file}")
    print("Note: This script computes expected output manually.")

    # Save a reference implementation aligned with the ONNX model
    code_file = os.path.join(base_path, "sgd_reference.py")
    with open(code_file, "w") as f:
        f.write('''
# Reference implementation of SGD operator
import numpy as np

def sgd_update(param, grad, learning_rate=0.01):
    """
    Simple SGD update: param = param - learning_rate * grad
    
    Args:
        param: Parameter tensor to update
        grad: Gradient tensor
        learning_rate: Learning rate (default: 0.01)
        
    Returns:
        Updated parameter tensor
    """
    return param - learning_rate * grad

# Example usage
if __name__ == "__main__":
    import numpy as np
    import os
    
    # Load test data if available
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "inputs.npz")
    
    if os.path.exists(input_file):
        data = np.load(input_file)
        weights = data["weights"]
        gradient = data["gradient"]
        
        # Default learning rate matches the model
        learning_rate = 0.01
        
        # Compute update
        updated_weights = sgd_update(weights, gradient, learning_rate)
        
        print(f"Input weights shape: {weights.shape}")
        print(f"Gradient shape: {gradient.shape}")
        print(f"Updated weights shape: {updated_weights.shape}")
        print(f"Learning rate: {learning_rate}")
        
        # Example showing norm of change
        weight_change = np.linalg.norm(updated_weights - weights)
        print(f"Norm of weight change: {weight_change}")
''')
    print(f"✅ Reference implementation saved to {code_file}")

if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_sgd_onnx_and_data(save_path)