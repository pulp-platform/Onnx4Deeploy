import onnx
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper


def generate_groupnormgradw_onnx_and_data(save_path=None):
    """Generate ONNX model for GroupNormGradW operator with test data"""

    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")

    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model_config = config.get("groupnormgradw", {})
    input_shape = tuple(model_config.get("input_shape", [1, 8, 1, 64]))
    num_groups = model_config.get("num_groups", 1)
    epsilon = model_config.get("epsilon", 0.001)
    opset_version = model_config.get("opset_version", 13)

    N, C, H, W = input_shape
    channels_per_group = C // num_groups

    print(f"Generating GroupNormGradW model with:")
    print(f"  Input shape: {input_shape}")
    print(f"  Num groups: {num_groups}")
    print(f"  Channels per group: {channels_per_group}")
    print(f"  Output shape (dGamma): [{C}]")
    print(f"  Epsilon: {epsilon}")

    # Set default save path if not provided
    base_path = save_path if save_path else os.path.join(script_dir, "onnx")

    # Define standard filenames
    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")

    # Ensure the save directory exists
    os.makedirs(base_path, exist_ok=True)

    # Generate random data
    # dY: upstream gradient (same shape as input)
    dY = np.random.randn(*input_shape).astype(np.float32)
    # X: forward input
    X = np.random.randn(*input_shape).astype(np.float32)

    # Compute mean and inv_std for forward pass
    X_reshaped = X.reshape(N, num_groups, channels_per_group, H, W)
    mean = X_reshaped.mean(axis=(2, 3, 4), keepdims=True)
    var = X_reshaped.var(axis=(2, 3, 4), keepdims=True)
    inv_std = 1.0 / np.sqrt(var + epsilon)

    # Flatten mean and inv_std to [N, G]
    mean_flat = mean.reshape(N, num_groups).astype(np.float32)
    inv_std_flat = inv_std.reshape(N, num_groups).astype(np.float32)

    # Save input data
    np.savez(input_file, dY=dY, X=X, mean=mean_flat, inv_std=inv_std_flat)
    print(f"Input data saved to {input_file}")

    # Define ONNX tensors
    dY_tensor = helper.make_tensor_value_info("dY", TensorProto.FLOAT, input_shape)
    X_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)
    mean_tensor = helper.make_tensor_value_info("mean", TensorProto.FLOAT, [N, num_groups])
    inv_std_tensor = helper.make_tensor_value_info("inv_std", TensorProto.FLOAT, [N, num_groups])

    dGamma_tensor = helper.make_tensor_value_info("dGamma", TensorProto.FLOAT, [C])

    # Create GroupNormGradW node
    groupnormgradw_node = helper.make_node(
        "GroupNormGradW",
        inputs=["dY", "X", "mean", "inv_std"],
        outputs=["dGamma"],
        name="groupnormgradw_node",
        epsilon=epsilon,
        num_groups=num_groups,
    )

    # Create graph
    graph_def = helper.make_graph(
        [groupnormgradw_node],
        "groupnormgradw_graph",
        [dY_tensor, X_tensor, mean_tensor, inv_std_tensor],
        [dGamma_tensor],
    )

    # Create model
    model_def = helper.make_model(
        graph_def,
        producer_name="groupnormgradw_model",
        opset_imports=[helper.make_opsetid("", opset_version)],
    )

    # Manually set output shape
    output_tensor = model_def.graph.output[0]
    del output_tensor.type.tensor_type.shape.dim[:]
    output_tensor.type.tensor_type.shape.dim.add().dim_value = C

    # Save ONNX model
    onnx.save(model_def, onnx_file)
    print(f"ONNX model saved to {onnx_file}")

    # Compute expected dGamma using numpy
    dGamma = compute_groupnormgradw_numpy(dY, X, num_groups, epsilon)

    # Save expected output
    np.savez(output_file, dGamma=dGamma)
    print(f"Expected output data saved to {output_file}")


def compute_groupnormgradw_numpy(dY, X, num_groups, epsilon):
    """
    Compute GroupNormGradW: gradient w.r.t. gamma (scale parameter)

    GroupNorm: Y = gamma * X_norm + beta
    where X_norm = (X - mean) / std

    dGamma = sum over (N, H, W) of (dY * X_norm)
    For each channel c: dGamma[c] = sum_{n,h,w} dY[n,c,h,w] * X_norm[n,c,h,w]
    """
    N, C, H, W = X.shape
    channels_per_group = C // num_groups

    # Reshape for group computation
    X_reshaped = X.reshape(N, num_groups, channels_per_group, H, W)
    dY_reshaped = dY.reshape(N, num_groups, channels_per_group, H, W)

    # Compute statistics
    mean = X_reshaped.mean(axis=(2, 3, 4), keepdims=True)
    var = X_reshaped.var(axis=(2, 3, 4), keepdims=True)
    std = np.sqrt(var + epsilon)

    # Normalized input
    X_norm = (X_reshaped - mean) / std

    # Compute dGamma: sum over (N, H, W) dimensions
    # dGamma has shape [C]
    dGamma_reshaped = (dY_reshaped * X_norm).sum(axis=(0, 3, 4))  # [G, C/G]
    dGamma = dGamma_reshaped.reshape(C).astype(np.float32)

    return dGamma


if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_groupnormgradw_onnx_and_data(save_path)
