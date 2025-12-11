import onnx
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper

def generate_convgradw_onnx_and_data(save_path=None):
    """ Generate ONNX model for ConvGradW operator with test data """

    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")

    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model_config = config.get("convgradw", {})
    output_grad_shape = tuple(model_config.get("output_grad_shape", [1, 2, 5, 5]))
    input_shape = tuple(model_config.get("input_shape", [1, 1, 2, 2]))
    weight_shape = tuple(model_config.get("weight_grad_shape", [2, 1, 2, 2]))
    kernel_shape = model_config.get("kernel_shape", [2, 2])
    strides = model_config.get("strides", [1, 1])
    pads = model_config.get("pads", [1, 1, 1, 1])
    dilations = model_config.get("dilations", [1, 1])
    group = model_config.get("group", 1)
    opset_version = model_config.get("opset_version", 13)

    print(f"Generating ConvGradW model with:")
    print(f"  Output gradient shape (NCHW): {output_grad_shape}")
    print(f"  Input shape (NCHW): {input_shape}")
    print(f"  Weight gradient shape (NCHW): {weight_shape}")
    print(f"  Kernel shape: {kernel_shape}")
    print(f"  Strides: {strides}")
    print(f"  Pads: {pads}")

    # Set default save path if not provided
    base_path = save_path if save_path else os.path.join(script_dir, "onnx")

    # Define standard filenames
    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")

    # Ensure the save directory exists
    os.makedirs(base_path, exist_ok=True)

    # Generate random data
    output_grad = np.random.randn(*output_grad_shape).astype(np.float32)
    input_data = np.random.randn(*input_shape).astype(np.float32)

    # Save BOTH inputs (output_grad and input_data)
    np.savez(input_file, output_grad=output_grad, input_data=input_data)
    print(f"✅ Input data saved to {input_file}")

    # Define ONNX tensors - BOTH are inputs now
    output_grad_tensor = helper.make_tensor_value_info("output_grad", TensorProto.FLOAT, output_grad_shape)
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
    weight_grad_tensor = helper.make_tensor_value_info("weight_grad", TensorProto.FLOAT, weight_shape)

    # Create ConvGradW node
    convgradw_node = helper.make_node(
        "ConvGradW",
        inputs=["output_grad", "input"],  # Two inputs
        outputs=["weight_grad"],
        name="convgradw_node",
        kernel_shape=kernel_shape,
        strides=strides,
        pads=pads,
        dilations=dilations,
        group=group
    )

    # Create graph with TWO inputs
    graph_def = helper.make_graph(
        [convgradw_node],
        "convgradw_graph",
        [output_grad_tensor, input_tensor],  # Both are inputs
        [weight_grad_tensor]
    )

    model_def = helper.make_model(
        graph_def,
        producer_name="convgradw_model",
        opset_imports=[helper.make_opsetid("", opset_version)]
    )

    # Manually set output shape
    output_tensor = model_def.graph.output[0]
    del output_tensor.type.tensor_type.shape.dim[:]
    for dim in weight_shape:
        output_tensor.type.tensor_type.shape.dim.add().dim_value = dim
    print("✅ Manually set output shape for ConvGradW")

    onnx.save(model_def, onnx_file)
    print(f"✅ ONNX model saved to {onnx_file}")

    # Compute expected weight gradient
    batch, c_out, out_h, out_w = output_grad_shape
    _, c_in, in_h, in_w = input_shape
    out_ch_total, in_ch_per_group, kernel_h, kernel_w = weight_shape
    stride_h, stride_w = strides
    dilation_h, dilation_w = dilations
    pad_top, pad_left, pad_bottom, pad_right = pads

    # Validate group configuration
    assert c_out % group == 0, f"Output channels {c_out} must be divisible by group {group}"
    assert c_in % group == 0, f"Input channels {c_in} must be divisible by group {group}"
    assert c_out == out_ch_total, f"Output channels mismatch: {c_out} vs {out_ch_total}"
    assert c_in // group == in_ch_per_group, f"Input channels per group mismatch: {c_in // group} vs {in_ch_per_group}"

    weight_grad = np.zeros(weight_shape, dtype=np.float32)

    # Python reference implementation for grouped convolution
    # For each group, output channels [g*c_out_per_group : (g+1)*c_out_per_group]
    # connect to input channels [g*c_in_per_group : (g+1)*c_in_per_group]
    c_out_per_group = c_out // group
    c_in_per_group = c_in // group
    
    for b in range(batch):
        for g in range(group):
            # Output channels for this group
            oc_start = g * c_out_per_group
            oc_end = (g + 1) * c_out_per_group
            
            # Input channels for this group
            ic_start = g * c_in_per_group
            ic_end = (g + 1) * c_in_per_group
            
            for oc_idx, oc in enumerate(range(oc_start, oc_end)):
                for ic_idx, ic in enumerate(range(ic_start, ic_end)):
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            
                            grad_sum = 0.0
                            
                            for oh in range(out_h):
                                for ow in range(out_w):
                                    
                                    ih = oh * stride_h + kh * dilation_h - pad_top
                                    iw = ow * stride_w + kw * dilation_w - pad_left
                                    
                                    if 0 <= ih < in_h and 0 <= iw < in_w:
                                        grad_sum += output_grad[b, oc, oh, ow] * input_data[b, ic, ih, iw]
                            
                            # Weight layout: [out_ch, in_ch_per_group, kH, kW]
                            # oc is global output channel index, we need local index within weight tensor
                            weight_grad[oc, ic_idx, kh, kw] = grad_sum

    np.savez(output_file, weight_grad=weight_grad)
    print(f"✅ Expected output data saved to {output_file}")

if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_convgradw_onnx_and_data(save_path)
