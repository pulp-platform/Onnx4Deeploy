import onnx
from onnx import helper
import torch
import os
import sys
import io
import shutil
import yaml
from mi_bminet_model.mi_bminet import *
from onnxruntime.training import artifacts
from utils.fixShape import *
from utils.utils import *
from utils.checkNetworkstructure import *
from utils.appendOptimizer import *


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


def load_train_config():
    """Load training configuration"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["training"]["learning_rate"]


def generate_mi_bminet_training_onnx(save_path=None):
    """Generate ONNX training model for MI-BMInet based on config, with optional save path"""

    (
        pretrained,
        img_size,
        num_classes,
        embedding_dim,
        num_heads,
        num_layers,
        batch_size,
        opset_version,
    ) = load_config()

    input_shape = (batch_size, 3, img_size, img_size)

    folder_name = f"MI_BMInet_train_{img_size}_{embedding_dim}_{num_heads}_{num_layers}"

    base_path = (
        save_path
        if save_path
        else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "onnx", folder_name
        )
    )
    os.makedirs(base_path, exist_ok=True)

    onnx_infer_file = os.path.join(base_path, "network_infer.onnx")
    onnx_train_file = os.path.join(base_path, "network_train.onnx")
    onnx_output_file = os.path.join(base_path, "network.onnx")
    onnx_train_optim = os.path.join(base_path, "network_train_optim.onnx")

    # Create model
    model = mi_bminet_small(
        pretrained=pretrained,
        img_size=img_size,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    model.train()
    model = randomize_layernorm_params(model)

    # Generate random input data for export
    input_tensor = torch.randn(*input_shape, dtype=torch.float32)

    # Export model to ONNX in training mode
    f = io.BytesIO()
    torch.onnx.export(
        model,
        input_tensor,
        f,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset_version,
        do_constant_folding=False,
        export_params=True,
        keep_initializers_as_inputs=False,
    )

    # Load ONNX model from buffer and save it as network_infer.onnx
    onnx_model = onnx.load_model_from_string(f.getvalue())
    onnx_model = randomize_onnx_initializers(onnx_model)

    onnx.save(onnx_model, onnx_infer_file)
    print(f"✅ Inference ONNX model saved to {onnx_infer_file}")

    fix_shared_initializers_by_node_name(onnx_infer_file, onnx_infer_file)
    rename_and_save_onnx(onnx_infer_file, onnx_infer_file)
    run_onnx_optimization(onnx_infer_file, embedding_dim, num_heads, input_shape)
    print_onnx_shapes(onnx_infer_file)
    onnx_model = onnx.load(onnx_infer_file)

    # Get all parameter names
    all_param_names = [init.name for init in onnx_model.graph.initializer]
    print(f"📋 All Parameters: {all_param_names}")

    # Define which parameters require gradients
    # Modify this list based on which layers you want to train
    requires_grad = [
        name
        for name in all_param_names
        if any(keyword in name for keyword in [
            "classifier",  # Train classifier layers
            # Add more keywords to train more layers
        ])
    ]

    frozen_params = [name for name in all_param_names if name not in requires_grad]
    print(f"❄️ Frozen Parameters: {frozen_params}")
    print(f"🔹 Training Only: {requires_grad}")

    # Generate artifacts for training
    artifacts.generate_artifacts(
        onnx_model,
        optimizer=artifacts.OptimType.SGD,
        loss=artifacts.LossType.CrossEntropyLoss,
        requires_grad=requires_grad,
        frozen_params=frozen_params,
        artifact_directory=base_path,
    )

    training_model_path = os.path.join(base_path, "training_model.onnx")
    if os.path.exists(training_model_path):
        os.rename(training_model_path, onnx_train_file)
        print(f"✅ Final Training ONNX model saved as {onnx_train_file}")

    # Load the training model
    onnx_model = onnx.load(onnx_train_file)
    graph = onnx_model.graph
    grad_tensor_names = [name + "_grad" for name in requires_grad]

    # Add gradient outputs
    for grad_name in grad_tensor_names:
        if not any(output.name == grad_name for output in graph.output):
            grad_output = helper.make_tensor_value_info(
                grad_name, onnx.TensorProto.FLOAT, None
            )
            graph.output.append(grad_output)

    onnx.save(onnx_model, onnx_train_optim)
    onnx.save(onnx_model, onnx_train_file)

    onnx_output_file = os.path.join(base_path, "network.onnx")
    run_train_onnx_optimization(onnx_train_optim, onnx_output_file)
    infer_shapes_with_custom_ops(onnx_output_file, onnx_output_file)
    rename_nodes(onnx_output_file, onnx_output_file)
    print_onnx_shapes(onnx_output_file)

    pre_sgd_model_path = os.path.join(base_path, "network_pre_sgd.onnx")
    shutil.copy(onnx_output_file, pre_sgd_model_path)

    print(f"✅ Training ONNX model saved to {onnx_output_file}")

    # SGD Append
    # Workaround: create_test_input_output() expects files in CCT path structure
    # Create a temporary copy in the expected location
    cct_folder_name = f"CCT_train_{img_size}_{embedding_dim}_{num_heads}_{num_layers}"
    cct_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx", cct_folder_name)
    os.makedirs(cct_base_path, exist_ok=True)
    cct_network_train = os.path.join(cct_base_path, "network_train.onnx")
    shutil.copy(onnx_train_file, cct_network_train)

    create_test_input_output()
    print(f"✅ Created test input and output data")

    # Copy generated input/output back to MI-BMInet path
    cct_input_file = os.path.join(cct_base_path, "inputs.npz")
    cct_output_file = os.path.join(cct_base_path, "outputs.npz")
    mi_input_file = os.path.join(base_path, "inputs.npz")
    mi_output_file = os.path.join(base_path, "outputs.npz")
    if os.path.exists(cct_input_file):
        shutil.copy(cct_input_file, mi_input_file)
    if os.path.exists(cct_output_file):
        shutil.copy(cct_output_file, mi_output_file)
    learning_rate = load_train_config()
    add_sgd_nodes(onnx_output_file, onnx_output_file, learning_rate=learning_rate)

    # Adjust Shape and Type Inference
    infer_shapes_with_custom_ops(onnx_output_file, onnx_output_file)
    type_inference(onnx_output_file, onnx_output_file)
    print(f"✅ Added SGD nodes to {onnx_output_file}")
    ensure_all_tensor_shapes(onnx_output_file, onnx_output_file)

    print(f"✅ Training graph generation completed successfully!")


if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_mi_bminet_training_onnx(save_path)
