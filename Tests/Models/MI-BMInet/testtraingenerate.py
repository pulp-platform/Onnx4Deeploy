import onnx
from onnx import helper
import torch
import os
import sys
import io
import shutil
import yaml
from mi_bminet_model.mi_bminet import MIBMINetDeploy
from onnxruntime.training import artifacts
from utils.fixShape import *
from utils.utils import *
from utils.checkNetworkstructure import *
from utils.appendOptimizer import *
from utils.trainOptimization import split_gn_to_single_stat_array


def load_config(config_filename="config.yaml"):
    """Load and parse config.yaml, returning MI-BMINet-specific parameters"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, config_filename)

    with open(config_file, "r") as f:
        config = yaml.safe_load(f).get("mi_bminet", {})

    return (
        config.get("pretrained", False),
        config.get("F1", 8),
        config.get("D", 2),
        config.get("F2", None),
        config.get("C", 8),
        config.get("T", 2000),
        config.get("N", 4),
        config.get("Nf", 64),
        config.get("Nf2", 16),
        config.get("activation", "relu"),
        config.get("batch_size", 1),
        config.get("opset_version", 12)
    )


def load_train_config(config_filename="config.yaml"):
    """Load training configuration from config.yaml"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, config_filename)

    with open(config_file, "r") as f:
        config = yaml.safe_load(f).get("training", {})

    return config.get("learning_rate", 0.01)


def create_test_input_output(base_path, batch_size, C, T, N):
    """Create test input/output data for MI-BMINet training"""
    import numpy as np
    import onnxruntime as ort
    import onnx
    from onnx import numpy_helper
    import re
    
    # Generate input data: EEG signals (batch_size, 1, C, T)
    input_data = np.random.randn(batch_size, 1, C, T).astype(np.float32)
    
    # Generate labels: class indices (batch_size,) for CrossEntropyLoss
    labels = np.random.randint(0, N, size=(batch_size,)).astype(np.int64)
    
    # Save inputs
    input_file = os.path.join(base_path, "inputs.npz")
    np.savez(input_file, input=input_data, labels=labels)
    print(f"✅ Test input data saved to {input_file}")
    print(f"   - Input shape: {input_data.shape}")
    print(f"   - Labels shape: {labels.shape}")
    
    # Run the training model to get outputs (gradients)
    network_train_path = os.path.join(base_path, "network_train.onnx")
    output_file = os.path.join(base_path, "outputs.npz")
    
    if not os.path.exists(network_train_path):
        print(f"⚠️ Training model not found: {network_train_path}")
        print("   Outputs will be generated after model creation")
        return input_file
    
    # Run inference to get gradients
    try:
        ort_session = ort.InferenceSession(network_train_path, providers=["CPUExecutionProvider"])
        output_names = [output.name for output in ort_session.get_outputs()]
        print(f"Training model has {len(output_names)} outputs")
        
        outputs = ort_session.run(None, {"input": input_data, "labels": labels})
        outputs_dict = {name: outputs[i] for i, name in enumerate(output_names)}
        
        # Load model to get parameter names
        model = onnx.load(network_train_path)
        param_names = [init.name for init in model.graph.initializer]
        
        # Map gradients to parameters
        learning_rate = load_train_config()
        grad_to_param_map = {}
        
        for output_name in output_names:
            if "grad" in output_name.lower():
                # Try to match gradient to parameter
                potential_param = re.sub(r'_grad.*$', '', output_name, flags=re.IGNORECASE)
                if potential_param in param_names:
                    grad_to_param_map[potential_param] = output_name
                    print(f"Mapped gradient {output_name} to parameter {potential_param}")
        
        # Apply SGD updates to parameters
        sgd_outputs = {}
        for param_name, grad_name in grad_to_param_map.items():
            # Get original parameter
            param_value = None
            for initializer in model.graph.initializer:
                if initializer.name == param_name:
                    param_value = numpy_helper.to_array(initializer)
                    break
            
            if param_value is not None:
                grad_value = outputs_dict[grad_name]
                # Apply SGD: param = param - lr * grad
                param_updated = param_value - learning_rate * grad_value
                
                updated_name = f"{param_name}_updated"
                sgd_outputs[updated_name] = param_updated
                print(f"✅ Applied SGD update to {param_name}")
        
        if sgd_outputs:
            np.savez(output_file, **sgd_outputs)
            print(f"✅ Updated parameters saved to {output_file}")
        else:
            print("⚠️ No parameters were updated")
            
    except Exception as e:
        print(f"⚠️ Error running training model: {e}")
        print("   Outputs will be generated later")
    
    return input_file


def generate_mi_bminet_training_onnx(save_path=None):
    """Generate ONNX training model for MI-BMINet based on config, with optional save path"""

    (
        pretrained,
        F1,
        D,
        F2,
        C,
        T,
        N,
        Nf,
        Nf2,
        activation,
        batch_size,
        opset_version,
    ) = load_config()

    # Input shape for EEG data: (batch_size, 1, channels, time_samples)
    input_shape = (batch_size, 1, C, T)

    folder_name = f"MI_BMINet_train_C{C}_T{T}_F1_{F1}_D{D}_N{N}"

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

    # Create deployment-optimized MI-BMINet model
    model = MIBMINetDeploy(
        F1=F1,
        D=D,
        F2=F2,
        C=C,
        T=T,
        N=N,
        Nf=Nf,
        Nf2=Nf2,
        activation=activation,
    )
    model.train()

    # Generate random input data for export (EEG signals)
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
    print("Randomizing initializers in inference model...")
    onnx_model = randomize_onnx_initializers(onnx_model)

    onnx.save(onnx_model, onnx_infer_file)
    print(f" Inference ONNX model saved to {onnx_infer_file}")

    # Fix shared initializers and rename
    fix_shared_initializers_by_node_name(onnx_infer_file, onnx_infer_file)
    rename_and_save_onnx(onnx_infer_file, onnx_infer_file)

    print_onnx_shapes(onnx_infer_file)
    onnx_model = onnx.load(onnx_infer_file)

    # Get all parameter names
    all_param_names = [init.name for init in onnx_model.graph.initializer]
    print(f"=All Parameters: {all_param_names}")
    # Define trainable parameters
    # For MI-BMINet, typically we want to train all layers
    # You can customize this list based on your fine-tuning strategy
    requires_grad = [
        name
        for name in all_param_names
        if name
        in [
            # Uncomment the parameters you want to train
            # Example: Train only the final classifier
            "fc_weight",
            "fc_bias",
            # "layer_norm3_weight",  # Fixed: removed leading space
            # "layer_norm3_bias",
            "conv1_weight",
            "layer_norm3_weight",
            # "layer_norm1_weight",
            # "layer_norm3_weight",
            "layer_norm2_weight",
            "layer_norm1_weight",
            "conv2_weight",
            "sep_conv1_weight",
            "sep_conv2_weight",
           
        ]
    ]

    frozen_params = [name for name in all_param_names if name not in requires_grad]

    print(f"=9 Training Parameters: {requires_grad}")
    print(f"=9 Frozen Parameters: {frozen_params}")

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
        print(f" Final Training ONNX model saved as {onnx_train_file}")

    # Load the training model and check gradient outputs
    onnx_model = onnx.load(onnx_train_file)
    graph = onnx_model.graph

    # Debug: Print all outputs from training model
    print(f"\n📋 Training model outputs ({len(graph.output)}):")
    for out in graph.output:
        print(f"   - {out.name}")

    # Debug: Find LayerNormalizationGrad nodes and their outputs
    print(f"\n📋 LayerNormalizationGrad nodes:")
    for node in graph.node:
        if "LayerNorm" in node.op_type and "Grad" in node.op_type:
            print(f"   - {node.name} ({node.op_type})")
            print(f"     Inputs: {list(node.input)}")
            print(f"     Outputs: {list(node.output)}")

    # Add gradient outputs that are expected but missing
    grad_tensor_names = [name + "_grad" for name in requires_grad]

    for grad_name in grad_tensor_names:
        if not any(output.name == grad_name for output in graph.output):
            # Check if this gradient is actually produced by any node
            is_produced = False
            for node in graph.node:
                if grad_name in node.output:
                    is_produced = True
                    break

            if is_produced:
                grad_output = helper.make_tensor_value_info(
                    grad_name, onnx.TensorProto.FLOAT, None
                )
                graph.output.append(grad_output)
                print(f"✅ Added gradient output: {grad_name}")
            else:
                print(f"⚠️ Gradient {grad_name} not produced by any node!")

    onnx.save(onnx_model, onnx_train_optim)
    onnx.save(onnx_model, onnx_train_file)

    # Check if any norm parameters need training (to determine if we split LayerNormGrad)
    train_norm_params = any("layer_norm" in name or "norm" in name.lower() for name in requires_grad)

    # Optimize training model
    onnx_output_file = os.path.join(base_path, "network.onnx")
    run_train_onnx_optimization(onnx_train_optim, onnx_output_file, split_layernormgrad=train_norm_params)
    infer_shapes_with_custom_ops(onnx_output_file, onnx_output_file)
    rename_nodes(onnx_output_file, onnx_output_file)
    print_onnx_shapes(onnx_output_file)

    # Save pre-SGD version
    pre_sgd_model_path = os.path.join(base_path, "network_pre_sgd.onnx")
    shutil.copy(onnx_output_file, pre_sgd_model_path)

    print(f"📊 Training ONNX model saved to {onnx_output_file}")

    # Create test input and output data
    create_test_input_output_pytorch(base_path, batch_size, C, T, N, model, requires_grad)
    print(f"✅ Created test input and output data")
    
    # Add SGD optimizer nodes
    learning_rate = load_train_config()
    add_sgd_nodes(onnx_output_file, onnx_output_file, learning_rate=learning_rate)

    # Adjust Shape and Type Inference
    infer_shapes_with_custom_ops(onnx_output_file, onnx_output_file)
    type_inference(onnx_output_file, onnx_output_file)
    print(f" Added SGD nodes to {onnx_output_file}")
    ensure_all_tensor_shapes(onnx_output_file, onnx_output_file)

    # Convert LayerNorm to GroupNorm for deployment
    convert_layernorm_to_groupnorm(onnx_output_file, onnx_output_file, num_groups=1)
    print(f"✅ Converted LayerNorm to GroupNorm in {onnx_output_file}")

    # Split GroupNorm into GroupNormalizationStat + GroupNormalization
    # This makes forward stat available for backward pass
    split_gn_to_single_stat_array(onnx_output_file, onnx_output_file)
    print(f"✅ Split GroupNorm into Stat + Norm in {onnx_output_file}")
    # infer_shapes_with_custom_ops(onnx_output_file, onnx_output_file)

    print(f"\n🎉 MI-BMINet training model generation complete!")
    print(f"📁 Output directory: {base_path}")
    print(f"📄 Files generated:")
    print(f"   - network_infer.onnx: Inference model")
    print(f"   - network_train.onnx: Training model (before optimization)")
    print(f"   - network_pre_sgd.onnx: Training model (before SGD)")
    print(f"   - network.onnx: Final training model with SGD")


if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_mi_bminet_training_onnx(save_path)
