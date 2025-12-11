import onnx
from onnx import helper, numpy_helper
from onnx import TensorProto
import re

def add_sgd_nodes(input_model_path, output_model_path, learning_rate=0.01):
    """
    Add SGD optimizer nodes to the ONNX model for automatic weight updates.
    Includes proper shape inference for output tensors.
    
    Args:
        input_model_path: Path to the input ONNX model
        output_model_path: Path to save the updated ONNX model
        learning_rate: Learning rate for SGD optimizer
        
    Returns:
        List of updated parameter names
    """
    model = onnx.load(input_model_path)
    graph = model.graph
    
    # Get all output names from the model
    output_names = [output.name for output in graph.output]
    
    # Get all initializer names (parameters)
    param_names = [init.name for init in graph.initializer]
    param_shapes = {init.name: numpy_helper.to_array(init).shape for init in graph.initializer}
    
    # Find all gradient outputs
    grad_to_param_map = {}

    for output_name in output_names:
        if "grad" in output_name.lower():
            print(f"Found gradient output: {output_name}")

            # Special case for *_Gemm_Grad_dC_reduced -> *_bias
            # This handles Gemm bias gradients like:
            # - node_15_fc_Gemm_Grad_dC_reduced -> fc_bias
            # - classifier_fc_Gemm_Grad_dC_reduced -> classifier_fc_bias
            if "gemm_grad_dc_reduced" in output_name.lower():
                # Extract the layer name from the gradient output
                # e.g., "node_15_fc_Gemm_Grad_dC_reduced" -> "fc"
                parts = output_name.lower().split("_gemm_grad_dc_reduced")[0]
                # Remove "node_XX_" prefix if present
                if parts.startswith("node_"):
                    # Find the layer name after node_XX_
                    parts = "_".join(parts.split("_")[2:])  # Skip "node" and number

                # Try to match with *_bias parameters
                bias_name = f"{parts}_bias"
                matched = False
                for param_name in param_names:
                    if param_name.lower() == bias_name.lower():
                        grad_to_param_map[param_name] = output_name
                        print(f"  ✅ Matched {output_name} -> {param_name} (Gemm bias)")
                        matched = True
                        break

                if matched:
                    continue
                else:
                    print(f"  ⚠️  Could not match Gemm bias gradient: {output_name}")

            # Regular case: param_name_grad -> param_name
            # Remove _grad suffix and match exactly
            if output_name.lower().endswith("_grad"):
                param_name_candidate = output_name[:-5]  # Remove "_grad"

                # Try exact match first
                if param_name_candidate in param_names:
                    grad_to_param_map[param_name_candidate] = output_name
                    print(f"  ✅ Matched {output_name} -> {param_name_candidate}")
                    continue

                # Try case-insensitive exact match
                for param_name in param_names:
                    if param_name.lower() == param_name_candidate.lower():
                        grad_to_param_map[param_name] = output_name
                        print(f"  ✅ Matched {output_name} -> {param_name}")
                        break
                else:
                    print(f"  ⚠️  No matching parameter found for {output_name}")
    
    # Add SGD update nodes for each parameter with a gradient
    new_nodes = []
    new_outputs = []
    updated_params = []
    
    for param_name, grad_name in grad_to_param_map.items():
        # Get original parameter shape
        if param_name not in param_shapes:
            print(f"⚠️ Could not find shape for parameter: {param_name}")
            continue
            
        param_shape = param_shapes[param_name]
        updated_param_name = f"{param_name}_updated"
        updated_params.append(updated_param_name)
        
        # Create SGD node with learning_rate as attribute
        sgd_node = helper.make_node(
            op_type="SGD",
            inputs=[
                param_name,  # weights to update
                grad_name,   # gradient
            ],
            outputs=[updated_param_name],
            name=f"{param_name}_sgd",
            domain="",
            lr=float(learning_rate)  # Set learning_rate as attribute
        )
        new_nodes.append(sgd_node)
        
        # Add output tensor info WITH SHAPE information
        output_tensor = helper.make_tensor_value_info(
            updated_param_name,
            TensorProto.FLOAT,
            param_shape  # Use original parameter shape
        )
        new_outputs.append(output_tensor)
        
        print(f"Added SGD node: {param_name} -> {updated_param_name} (shape: {param_shape})")
    
    # Add all new nodes to the graph
    graph.node.extend(new_nodes)
    
    # Create a new outputs list with only the updated parameters
    new_graph_outputs = []
    for output in new_outputs:
        new_graph_outputs.append(output)
    
    # Replace the graph outputs
    del graph.output[:]
    graph.output.extend(new_graph_outputs)
    
    # Save the updated model
    onnx.save(model, output_model_path)
    
    print(f"✅ Added SGD nodes for {len(updated_params)} parameters with proper shape information")
    
    return updated_params

def get_output_shape(graph, output_name):
    """
    Get the shape of an output tensor by name.
    
    Args:
        graph: ONNX graph
        output_name: Name of the output tensor
        
    Returns:
        List representing the tensor shape
    """
    # First check among graph outputs
    for output in graph.output:
        if output.name == output_name:
            return [dim.dim_value for dim in output.type.tensor_type.shape.dim]
    
    # If not found in outputs, look for value_info
    for value_info in graph.value_info:
        if value_info.name == output_name:
            return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
    
    raise ValueError(f"Cannot find shape for output {output_name}")

def get_value_info_type(tensor):
    """
    Get the data type of a tensor.
    
    Args:
        tensor: ONNX tensor
        
    Returns:
        ONNX data type
    """
    return tensor.data_type