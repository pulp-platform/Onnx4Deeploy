import onnx
import onnxruntime as ort
import numpy as np
import sys
import os
import torch
import yaml
from mi_bminet_model.mi_bminet import *
from utils.utils import *


def load_config():
    """Load configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    mi_bminet_config = config["mi_bminet"]
    training_config = config["training"]

    return (
        mi_bminet_config["pretrained"],
        mi_bminet_config["img_size"],
        mi_bminet_config["num_classes"],
        mi_bminet_config["embedding_dim"],
        mi_bminet_config["num_heads"],
        mi_bminet_config["num_layers"],
        mi_bminet_config["batch_size"],
        mi_bminet_config["opset_version"],
    )


def generate_mi_bminet_onnx_and_data(save_path=None):
    """Generate ONNX model for MI-BMInet based on config, with optional save path"""

    pretrained, img_size, num_classes, embedding_dim, num_heads, num_layers, batch_size, opset_version = load_config()
    print(f"✅ Loaded config: img_size={img_size}, embedding_dim={embedding_dim}, num_heads={num_heads}, num_layers={num_layers}, opset_version={opset_version}")

    input_shape = (1, 3, img_size, img_size)

    folder_name = f"MI_BMInet_infer_{img_size}_{embedding_dim}_{num_heads}_{num_layers}"

    base_path = save_path if save_path else os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx", folder_name)

    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")

    os.makedirs(base_path, exist_ok=True)

    # Create model
    model = mi_bminet_small(
        pretrained=pretrained,
        img_size=img_size,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    model.eval()
    model = randomize_layernorm_params(model)

    # Generate random input data
    input_data = np.random.randn(*input_shape).astype(np.float32)
    np.savez(input_file, input=input_data)

    input_tensor = torch.tensor(input_data)

    # Export to ONNX
    torch.onnx.export(
        model,
        input_tensor,
        onnx_file,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    onnx_model = onnx.load(onnx_file)
    onnx_model = randomize_onnx_initializers(onnx_model)
    print(f"✅ ONNX model saved to {onnx_file}")
    rename_and_save_onnx(onnx_file, onnx_file)

    # Run optimization
    run_onnx_optimization_infer(onnx_file, embedding_dim, num_heads, input_shape)
    rename_and_save_onnx(onnx_file, onnx_file)

    # Run inference to generate output data
    ort_session = ort.InferenceSession(onnx_file)
    output_data = ort_session.run(None, {"input": input_data})[0]

    np.savez(output_file, output=output_data)
    print(f"✅ Output data saved to {output_file}")

    # Unify GEMM input dimensions
    unify_gemm_input_dims(onnx_file, onnx_file)
    print(f"✅ Successfully unified GEMM input dimensions. Saved as {onnx_file}")


if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_mi_bminet_onnx_and_data(save_path)
