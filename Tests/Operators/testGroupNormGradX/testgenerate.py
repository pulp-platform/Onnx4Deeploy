import onnx
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper

def generate_groupnormgradx_onnx_and_data(save_path=None):
    """Generate ONNX model for GroupNormGradX operator with combined stat array [N, G, 2]"""

    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")

    # Load configuration
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    model_config = config.get("groupnormgradx", {})
    input_shape = tuple(model_config.get("input_shape", [1, 8, 1, 64]))
    num_groups = model_config.get("num_groups", 1)
    epsilon = model_config.get("epsilon", 0.001)
    opset_version = model_config.get("opset_version", 13)

    N, C, H, W = input_shape
    print(f"Generating GroupNormGradX model (Combined Stat):")
    print(f"  Input shape: {input_shape}")
    print(f"  Stat shape: [{N}, {num_groups}, 2]")

    # Set default save path
    base_path = save_path if save_path else os.path.join(script_dir, "onnx")
    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")
    os.makedirs(base_path, exist_ok=True)

    # 1. Generate random data
    dY = np.random.randn(*input_shape).astype(np.float32)
    X = np.random.randn(*input_shape).astype(np.float32)
    gamma = np.random.randn(C).astype(np.float32)

    # 2. Compute combined stat array [N, G, 2]
    channels_per_group = C // num_groups
    X_reshaped = X.reshape(N, num_groups, channels_per_group, H, W)
    mean = X_reshaped.mean(axis=(2, 3, 4)) # [N, G]
    var = X_reshaped.var(axis=(2, 3, 4))   # [N, G]
    inv_std = 1.0 / np.sqrt(var + epsilon) # [N, G]
    
    # Combined stat: [N, G, 0]=mean, [N, G, 1]=inv_std
    stat = np.stack([mean, inv_std], axis=-1).astype(np.float32)

    # Save inputs (now including 'stat' instead of separate mean/inv_std)
    np.savez(input_file, dY=dY, X=X, gamma=gamma, stat=stat)
    print(f"Input data saved to {input_file}")

    # 3. Define ONNX tensors
    dY_vi = helper.make_tensor_value_info("dY", TensorProto.FLOAT, input_shape)
    X_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)
    gamma_vi = helper.make_tensor_value_info("gamma", TensorProto.FLOAT, [C])
    # 这里的 stat 必须是单输入
    stat_vi = helper.make_tensor_value_info("stat", TensorProto.FLOAT, [N, num_groups, 2])
    
    dX_vi = helper.make_tensor_value_info("dX", TensorProto.FLOAT, input_shape)

    # 4. Create GroupNormGradX node (Inputs: dY, X, gamma, stat)
    groupnormgradx_node = helper.make_node(
        "GroupNormGradX",
        inputs=["dY", "X", "gamma", "stat"], # 4个输入
        outputs=["dX"],
        name="groupnormgradx_node",
        epsilon=epsilon,
        num_groups=num_groups,
    )

    # Create graph
    graph_def = helper.make_graph(
        [groupnormgradx_node],
        "groupnormgradx_graph",
        [dY_vi, X_vi, gamma_vi, stat_vi],
        [dX_vi],
        value_info=[stat_vi] # 注入 shape 信息防止 Codegen 识别为 scalar
    )

    # Create model
    model_def = helper.make_model(
        graph_def,
        producer_name="groupnormgradx_model",
        opset_imports=[helper.make_opsetid("", opset_version)],
    )

    # Save ONNX
    onnx.save(model_def, onnx_file)
    print(f"ONNX model saved to {onnx_file}")

    # 5. Compute expected dX using updated numpy logic
    dX = compute_groupnormgradx_numpy_combined(dY, X, gamma, stat, num_groups)

    # Save expected output
    np.savez(output_file, dX=dX)
    print(f"Expected output data saved to {output_file}")


def compute_groupnormgradx_numpy_combined(dY, X, gamma, stat, num_groups):
    """
    Updated Numpy implementation to handle [N, G, 2] stat array
    """
    N, C, H, W = X.shape
    G = num_groups
    C_p_G = C // G

    # 从 stat 数组解析 mean 和 inv_std
    mean = stat[:, :, 0]    # [N, G]
    inv_std = stat[:, :, 1] # [N, G]

    # Reshape for computation
    X_reshaped = X.reshape(N, G, C_p_G, H, W)
    dY_reshaped = dY.reshape(N, G, C_p_G, H, W)
    gamma_reshaped = gamma.reshape(1, G, C_p_G, 1, 1)

    # Broadcast mean/inv_std to [N, G, 1, 1, 1]
    mean_b = mean.reshape(N, G, 1, 1, 1)
    inv_std_b = inv_std.reshape(N, G, 1, 1, 1)

    # X_norm = (X - mean) * inv_std
    X_norm = (X_reshaped - mean_b) * inv_std_b

    # Weighted gradient
    gamma_dY = gamma_reshaped * dY_reshaped

    # Mean of (gamma * dY) and (gamma * dY * X_norm) over group
    mean_gamma_dY = gamma_dY.mean(axis=(2, 3, 4), keepdims=True)
    mean_gamma_dY_Xnorm = (gamma_dY * X_norm).mean(axis=(2, 3, 4), keepdims=True)

    # dX = inv_std * (gamma_dY - mean_gamma_dY - X_norm * mean_gamma_dY_Xnorm)
    dX_reshaped = inv_std_b * (gamma_dY - mean_gamma_dY - X_norm * mean_gamma_dY_Xnorm)

    return dX_reshaped.reshape(N, C, H, W).astype(np.float32)


if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_groupnormgradx_onnx_and_data(save_path)