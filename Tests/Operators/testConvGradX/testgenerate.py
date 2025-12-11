import onnx
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper, shape_inference

def generate_convgradx_onnx_and_data(save_path=None):
    """ Generate ONNX model for ConvGradX operator with test data """

    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")

    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model_config = config.get("convgradx", {})
    output_grad_shape = tuple(model_config.get("input_grad_shape", [1, 8, 8, 8]))
    weight_shape = tuple(model_config.get("weight_shape", [8, 16, 3, 3]))
    input_shape = tuple(model_config.get("input_data_shape", [1, 16, 8, 8]))
    kernel_shape = model_config.get("kernel_shape", [3, 3])
    strides = model_config.get("strides", [1, 1])
    pads = model_config.get("pads", [1, 1, 1, 1])
    dilations = model_config.get("dilations", [1, 1])
    group = model_config.get("group", 1)
    opset_version = model_config.get("opset_version", 13)

    print(f"Generating ConvGradX model with:")
    print(f"  Output gradient shape: {output_grad_shape}")
    print(f"  Weight shape: {weight_shape}")
    print(f"  Input gradient shape (output): {input_shape}")
    print(f"  Kernel shape: {kernel_shape}")
    print(f"  Strides: {strides}")
    print(f"  Pads: {pads}")
    print(f"  Dilations: {dilations}")
    print(f"  Group: {group}")

    # Set default save path if not provided
    base_path = save_path if save_path else os.path.join(script_dir, "onnx")

    # Define standard filenames
    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")

    # Ensure the save directory exists
    os.makedirs(base_path, exist_ok=True)

    # Generate random gradient data
    output_grad = np.random.randn(*output_grad_shape).astype(np.float32)
    weight = np.random.randn(*weight_shape).astype(np.float32)

    # Save input data (only output_grad, weight is a constant in the graph)
    np.savez(input_file, output_grad=output_grad)
    print(f"✅ Input data saved to {input_file}")

    # Define ONNX tensors
    output_grad_tensor = helper.make_tensor_value_info("output_grad", TensorProto.FLOAT, output_grad_shape)
    
    # Weight as Constant
    weight_constant = helper.make_tensor(
        name="weight",
        data_type=TensorProto.FLOAT,
        dims=weight_shape,
        vals=weight.flatten().tolist()
    )
    
    input_grad_tensor = helper.make_tensor_value_info("input_grad", TensorProto.FLOAT, input_shape)

    # Create ConvGradX node
    convgradx_node = helper.make_node(
        "ConvGradX",
        inputs=["output_grad", "weight"],
        outputs=["input_grad"],
        name="convgradx_node",
        kernel_shape=kernel_shape,
        strides=strides,
        pads=pads,
        dilations=dilations,
        group=group
    )

    # Create graph (weight is now an initializer, not an input)
    graph_def = helper.make_graph(
        [convgradx_node],
        "convgradx_graph",
        [output_grad_tensor],  # Only output_grad is input
        [input_grad_tensor],
        [weight_constant]  # Weight is a constant initializer
    )

    # Create model
    model_def = helper.make_model(
        graph_def,
        producer_name="convgradx_model",
        opset_imports=[helper.make_opsetid("", opset_version)]
    )

    # Manually set output shape
    output_tensor = model_def.graph.output[0]
    del output_tensor.type.tensor_type.shape.dim[:]
    for dim in input_shape:
        output_tensor.type.tensor_type.shape.dim.add().dim_value = dim
    print("✅ Manually set output shape for ConvGradX")

    # Save ONNX model
    onnx.save(model_def, onnx_file)
    print(f"✅ ONNX model saved to {onnx_file}")

    # Compute the expected input gradient
    batch, out_ch, out_h, out_w = output_grad_shape
    out_ch_w, in_ch_per_group_from_weight, kernel_h, kernel_w = weight_shape
    batch_in, in_ch, in_h, in_w = input_shape  # Get actual in_ch from input_shape
    stride_h, stride_w = strides
    dilation_h, dilation_w = dilations
    pad_top, pad_left, pad_bottom, pad_right = pads

    input_grad = np.zeros(input_shape, dtype=np.float32)

    # Calculate channels per group
    in_ch_per_group = in_ch // group
    out_ch_per_group = out_ch // group

    for b in range(batch):
        for g in range(group):
            # Calculate channel ranges for this group
            ic_start = g * in_ch_per_group
            ic_end = (g + 1) * in_ch_per_group
            oc_start = g * out_ch_per_group
            oc_end = (g + 1) * out_ch_per_group
            
            for ic in range(ic_start, ic_end):
                for ih in range(in_h):
                    for iw in range(in_w):
                        
                        grad_sum = 0.0
                        
                        # Only iterate over output channels in the same group
                        for oc in range(oc_start, oc_end):
                            for kh in range(kernel_h):
                                for kw in range(kernel_w):
                                    
                                    oh_candidate = ih + pad_top - kh * dilation_h
                                    ow_candidate = iw + pad_left - kw * dilation_w
                                    
                                    if oh_candidate % stride_h == 0 and ow_candidate % stride_w == 0:
                                        oh = oh_candidate // stride_h
                                        ow = ow_candidate // stride_w
                                        
                                        if 0 <= oh < out_h and 0 <= ow < out_w:
                                            # Weight indexing for grouped convolution
                                            # weight shape: [out_ch, in_ch_per_group, kh, kw]
                                            ic_in_group = ic - ic_start
                                            grad_sum += output_grad[b, oc, oh, ow] * weight[oc, ic_in_group, kh, kw]
                        
                        input_grad[b, ic, ih, iw] = grad_sum

    # Save output data
    np.savez(output_file, input_grad=input_grad)
    print(f"✅ Expected output data saved to {output_file}")

if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_convgradx_onnx_and_data(save_path)
