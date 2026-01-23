import onnx
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper


def generate_groupnorm_onnx_and_data(save_path=None):
    """Generate ONNX model for GroupNormalization operator with test data"""

    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")

    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model_config = config.get("groupnorm", {})
    input_shape = tuple(model_config.get("input_shape", [1, 8, 1, 64]))
    num_groups = model_config.get("num_groups", 1)
    epsilon = model_config.get("epsilon", 0.001)
    opset_version = model_config.get("opset_version", 13)

    N, C, H, W = input_shape
    channels_per_group = C // num_groups

    print(f"Generating GroupNormalization model with:")
    print(f"  Input shape: {input_shape}")
    print(f"  Num groups: {num_groups}")
    print(f"  Channels per group: {channels_per_group}")
    print(f"  Output shape (Y): {input_shape}")
    print(f"  Stat shape: [{N}, {num_groups}, 2] (stat[:,:,0]=mean, stat[:,:,1]=inv_std)")
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
    # X: input tensor
    X = np.random.randn(*input_shape).astype(np.float32)
    # gamma: scale parameter (per channel)
    gamma = np.random.randn(C).astype(np.float32)
    # beta: bias parameter (per channel)
    beta = np.random.randn(C).astype(np.float32)

    # Save input data
    np.savez(input_file, X=X, gamma=gamma, beta=beta)
    print(f"Input data saved to {input_file}")

    # Define ONNX input tensors
    X_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)
    gamma_tensor = helper.make_tensor_value_info("gamma", TensorProto.FLOAT, [C])
    beta_tensor = helper.make_tensor_value_info("beta", TensorProto.FLOAT, [C])

    # Define ONNX output tensors
    Y_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, input_shape)
    # stat tensor: [N, G, 2] where stat[:,:,0]=mean, stat[:,:,1]=inv_std
    stat_tensor = helper.make_tensor_value_info("stat", TensorProto.FLOAT, [N, num_groups, 2])

    # Create GroupNormalizationStat node (computes mean and inv_std)
    # Input: X
    # Output: stat [N, G, 2] where stat[:,:,0]=mean, stat[:,:,1]=inv_std
    groupnorm_stat_node = helper.make_node(
        "GroupNormalizationStat",
        inputs=["X"],
        outputs=["stat"],
        name="groupnorm_stat_node",
        epsilon=epsilon,
        num_groups=num_groups,
    )

    # Create GroupNormalization node (applies normalization)
    # Input: X, gamma, beta, stat
    # Output: Y
    groupnorm_node = helper.make_node(
        "GroupNormalization",
        inputs=["X", "gamma", "beta", "stat"],
        outputs=["Y"],
        name="groupnorm_node",
        epsilon=epsilon,
        num_groups=num_groups,
    )

    # Create graph with two nodes
    graph_def = helper.make_graph(
        [groupnorm_stat_node, groupnorm_node],
        "groupnorm_graph",
        [X_tensor, gamma_tensor, beta_tensor],
        [Y_tensor, stat_tensor],
    )

    # Create model
    model_def = helper.make_model(
        graph_def,
        producer_name="groupnorm_model",
        opset_imports=[helper.make_opsetid("", opset_version)],
    )

    # Manually set output shapes
    # Y output shape
    Y_output = model_def.graph.output[0]
    del Y_output.type.tensor_type.shape.dim[:]
    for dim in input_shape:
        Y_output.type.tensor_type.shape.dim.add().dim_value = dim

    # stat output shape [N, G, 2]
    stat_output = model_def.graph.output[1]
    del stat_output.type.tensor_type.shape.dim[:]
    stat_output.type.tensor_type.shape.dim.add().dim_value = N
    stat_output.type.tensor_type.shape.dim.add().dim_value = num_groups
    stat_output.type.tensor_type.shape.dim.add().dim_value = 2

    # Save ONNX model
    onnx.save(model_def, onnx_file)
    print(f"ONNX model saved to {onnx_file}")

    # Compute expected outputs using numpy
    Y, stat = compute_groupnorm_numpy(X, gamma, beta, num_groups, epsilon)

    # Save expected output
    np.savez(output_file, Y=Y, stat=stat)
    print(f"Expected output data saved to {output_file}")


def compute_groupnorm_numpy(X, gamma, beta, num_groups, epsilon):
    """
    Compute GroupNormalization forward pass

    GroupNormalization: Y = gamma * X_norm + beta
    where X_norm = (X - mean) / std

    Also returns stat [N, G, 2] where stat[:,:,0]=mean, stat[:,:,1]=inv_std
    """
    N, C, H, W = X.shape
    channels_per_group = C // num_groups

    # Reshape for group computation
    X_reshaped = X.reshape(N, num_groups, channels_per_group, H, W)

    # Compute statistics per group
    mean = X_reshaped.mean(axis=(2, 3, 4), keepdims=True)
    var = X_reshaped.var(axis=(2, 3, 4), keepdims=True)
    std = np.sqrt(var + epsilon)
    inv_std = 1.0 / std

    # Normalize
    X_norm = (X_reshaped - mean) / std

    # Reshape back to [N, C, H, W]
    X_norm = X_norm.reshape(N, C, H, W)

    # Apply scale and shift (gamma and beta are per-channel)
    # gamma and beta have shape [C], need to broadcast to [1, C, 1, 1]
    gamma_reshaped = gamma.reshape(1, C, 1, 1)
    beta_reshaped = beta.reshape(1, C, 1, 1)

    Y = gamma_reshaped * X_norm + beta_reshaped

    # Flatten mean and inv_std to [N, G]
    mean_flat = mean.reshape(N, num_groups)
    inv_std_flat = inv_std.reshape(N, num_groups)

    # Combine into stat [N, G, 2] where stat[:,:,0]=mean, stat[:,:,1]=inv_std
    stat = np.stack([mean_flat, inv_std_flat], axis=-1).astype(np.float32)

    return Y.astype(np.float32), stat


if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_groupnorm_onnx_and_data(save_path)
