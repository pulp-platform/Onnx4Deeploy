import onnx
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper


def generate_groupnormgradx_onnx_and_data(save_path=None):
    """Generate ONNX model for GroupNormGradX operator with test data"""

    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")

    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model_config = config.get("groupnormgradx", {})
    input_shape = tuple(model_config.get("input_shape", [1, 8, 1, 64]))
    num_groups = model_config.get("num_groups", 1)
    epsilon = model_config.get("epsilon", 0.001)
    opset_version = model_config.get("opset_version", 13)

    N, C, H, W = input_shape
    channels_per_group = C // num_groups

    print(f"Generating GroupNormGradX model with:")
    print(f"  Input shape: {input_shape}")
    print(f"  Num groups: {num_groups}")
    print(f"  Channels per group: {channels_per_group}")
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
    # gamma: scale parameter (shape: [C])
    gamma = np.random.randn(C).astype(np.float32)

    # Compute mean and inv_std for forward pass (needed by GroupNormGradX)
    X_reshaped = X.reshape(N, num_groups, channels_per_group, H, W)
    mean = X_reshaped.mean(axis=(2, 3, 4), keepdims=True)  # [N, G, 1, 1, 1]
    var = X_reshaped.var(axis=(2, 3, 4), keepdims=True)
    inv_std = 1.0 / np.sqrt(var + epsilon)  # [N, G, 1, 1, 1]

    # Flatten mean and inv_std to [N, G]
    mean_flat = mean.reshape(N, num_groups).astype(np.float32)
    inv_std_flat = inv_std.reshape(N, num_groups).astype(np.float32)

    # Save input data
    np.savez(input_file, dY=dY, X=X, gamma=gamma, mean=mean_flat, inv_std=inv_std_flat)
    print(f"Input data saved to {input_file}")

    # Define ONNX tensors
    dY_tensor = helper.make_tensor_value_info("dY", TensorProto.FLOAT, input_shape)
    X_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)
    gamma_tensor = helper.make_tensor_value_info("gamma", TensorProto.FLOAT, [C])
    mean_tensor = helper.make_tensor_value_info("mean", TensorProto.FLOAT, [N, num_groups])
    inv_std_tensor = helper.make_tensor_value_info("inv_std", TensorProto.FLOAT, [N, num_groups])

    dX_tensor = helper.make_tensor_value_info("dX", TensorProto.FLOAT, input_shape)

    # Create GroupNormGradX node
    groupnormgradx_node = helper.make_node(
        "GroupNormGradX",
        inputs=["dY", "X", "gamma", "mean", "inv_std"],
        outputs=["dX"],
        name="groupnormgradx_node",
        epsilon=epsilon,
        num_groups=num_groups,
    )

    # Create graph
    graph_def = helper.make_graph(
        [groupnormgradx_node],
        "groupnormgradx_graph",
        [dY_tensor, X_tensor, gamma_tensor, mean_tensor, inv_std_tensor],
        [dX_tensor],
    )

    # Create model
    model_def = helper.make_model(
        graph_def,
        producer_name="groupnormgradx_model",
        opset_imports=[helper.make_opsetid("", opset_version)],
    )

    # Manually set output shape
    output_tensor = model_def.graph.output[0]
    del output_tensor.type.tensor_type.shape.dim[:]
    for dim in input_shape:
        output_tensor.type.tensor_type.shape.dim.add().dim_value = dim

    # Save ONNX model
    onnx.save(model_def, onnx_file)
    print(f"ONNX model saved to {onnx_file}")

    # Compute expected dX using numpy
    dX = compute_groupnormgradx_numpy(dY, X, gamma, num_groups, epsilon)

    # Save expected output
    np.savez(output_file, dX=dX)
    print(f"Expected output data saved to {output_file}")


def compute_groupnormgradx_numpy(dY, X, gamma, num_groups, epsilon):
    """
    Compute GroupNormGradX: gradient w.r.t. input X

    GroupNorm: Y = gamma * (X - mean) / sqrt(var + eps) + beta

    dX = (1/std) * (gamma * dY - mean(gamma * dY) - X_norm * mean(gamma * dY * X_norm))
    where X_norm = (X - mean) / std
    """
    N, C, H, W = X.shape
    channels_per_group = C // num_groups

    # Reshape for group computation
    X_reshaped = X.reshape(N, num_groups, channels_per_group, H, W)
    dY_reshaped = dY.reshape(N, num_groups, channels_per_group, H, W)
    gamma_reshaped = gamma.reshape(1, num_groups, channels_per_group, 1, 1)

    # Compute statistics
    mean = X_reshaped.mean(axis=(2, 3, 4), keepdims=True)
    var = X_reshaped.var(axis=(2, 3, 4), keepdims=True)
    std = np.sqrt(var + epsilon)
    inv_std = 1.0 / std

    # Normalized input
    X_norm = (X_reshaped - mean) * inv_std

    # Number of elements per group
    group_size = channels_per_group * H * W

    # Weighted gradient
    gamma_dY = gamma_reshaped * dY_reshaped

    # Mean of gamma * dY over the group
    mean_gamma_dY = gamma_dY.mean(axis=(2, 3, 4), keepdims=True)

    # Mean of gamma * dY * X_norm over the group
    mean_gamma_dY_Xnorm = (gamma_dY * X_norm).mean(axis=(2, 3, 4), keepdims=True)

    # Compute dX
    dX_reshaped = inv_std * (gamma_dY - mean_gamma_dY - X_norm * mean_gamma_dY_Xnorm)

    # Reshape back
    dX = dX_reshaped.reshape(N, C, H, W).astype(np.float32)

    return dX


if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_groupnormgradx_onnx_and_data(save_path)
