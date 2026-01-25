import onnx
from onnx import helper
import torch
import os
import sys
import io
import shutil
import yaml
from epidenet_model.epidenet import EpiDeNetDeploy
from onnxruntime.training import artifacts

# Add CCT utils to path
cct_utils_path = os.path.join(os.path.dirname(__file__), '..', 'CCT')
sys.path.insert(0, cct_utils_path)
from utils.fixShape import *
from utils.utils import *
from utils.checkNetworkstructure import *
from utils.appendOptimizer import *
from utils.trainOptimization import *


def load_config(config_filename="config.yaml"):
    """Load and parse config.yaml, returning EpiDeNet-specific parameters"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, config_filename)

    with open(config_file, "r") as f:
        config = yaml.safe_load(f).get("epidenet", {})

    return (
        config.get("pretrained", False),
        config.get("C", 16),
        config.get("T", 1000),
        config.get("N", 11),
        config.get("batch_size", 1),
        config.get("opset_version", 12)
    )


def load_train_config(config_filename="config.yaml"):
    """Load training configuration from config.yaml"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, config_filename)

    with open(config_file, "r") as f:
        config = yaml.safe_load(f).get("training", {})

    return config.get("learning_rate", 0.001)


def create_test_input_output(base_path, batch_size, C, T, N):
    """Create test input/output data for EpiDeNet training"""
    import numpy as np
    import onnxruntime as ort
    import onnx
    from onnx import numpy_helper
    import re

    seed = 42
    np.random.seed(seed)

    # Generate input data: EOG signals (batch_size, 1, C, T)
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


def create_test_input_output_pytorch(base_path, batch_size, C, T, N, model_pytorch, requires_grad_list):
    """
    Generate reference test inputs and outputs using GroupNorm model for gradient compatibility.
    Uses EpiDeNetDeployGroupNorm internally so that gradient shapes match Deeploy's GroupNormGradW.
    Loads weights from ONNX file to ensure randomized biases match between ONNX and PyTorch.
    """
    import numpy as np
    import torch.nn as nn
    import onnx
    from onnx import numpy_helper
    from epidenet_model.epidenet import EpiDeNetDeployGroupNorm

    # DEBUG: Storage for layer_norm3 inputs/outputs
    debug_data = {
        'layer_norm3_input': None,      # X - input to layer_norm3
        'layer_norm3_grad_output': None,  # dY - upstream gradient
        # Relu_2 debug data (relu after layer_norm3, before pool3)
        'relu_2_data_in': None,         # relu input (layer_norm3 output)
        'relu_2_grad_out': None,        # gradient from pool3 (relu grad_output)
        'relu_2_grad_in': None,         # gradient to layer_norm3 (relu grad_input)
        # Pool3 debug data (AvgPoolGrad)
        'pool3_data_in': None,          # pool3 input (relu_2 output)
        'pool3_data_out': None,         # pool3 output (conv4 input)
        'pool3_grad_out': None,         # gradient INTO pool3 from conv4 (AvgPoolGrad input)
        'pool3_grad_in': None,          # gradient OUT of pool3 to relu_2 (AvgPoolGrad output)
        # Conv4 debug data (ConvGradX)
        'conv4_grad_in': None,          # gradient INTO conv4 (from layer_norm4 backward)
        'conv4_grad_out': None,         # gradient OUT of conv4 to pool3 (ConvGradX output)
    }

    def layer_norm3_forward_hook(module, input, output):
        """Capture input to layer_norm3 (X) and output (relu input)"""
        debug_data['layer_norm3_input'] = input[0].detach().clone()
        # layer_norm3 output = relu_2 input
        debug_data['relu_2_data_in'] = output.detach().clone()

    def layer_norm3_backward_hook(module, grad_input, grad_output):
        """Capture upstream gradient (dY) to layer_norm3 = relu_2 grad_in"""
        debug_data['layer_norm3_grad_output'] = grad_output[0].detach().clone()
        # grad_output to layer_norm3 = grad_input from relu_2
        debug_data['relu_2_grad_in'] = grad_output[0].detach().clone()

    def pool3_forward_hook(module, input, output):
        """pool3 forward: capture input (relu_2 output) and output (conv4 input)"""
        debug_data['pool3_data_in'] = input[0].detach().clone()
        debug_data['pool3_data_out'] = output.detach().clone()

    def pool3_backward_hook(module, grad_input, grad_output):
        """pool3 backward:
           - grad_input[0] = gradient flowing OUT of pool3 to relu_2 (AvgPoolGrad output)
           - grad_output[0] = gradient flowing INTO pool3 from conv4 (AvgPoolGrad input)
        """
        debug_data['relu_2_grad_out'] = grad_input[0].detach().clone()
        debug_data['pool3_grad_in'] = grad_input[0].detach().clone()  # same as relu_2_grad_out
        debug_data['pool3_grad_out'] = grad_output[0].detach().clone()  # from conv4

    def conv4_backward_hook(module, grad_input, grad_output):
        """conv4 backward:
           - grad_input[0] = gradient w.r.t. conv4's input (ConvGradX output) = pool3's grad_output
           - grad_output[0] = gradient w.r.t. conv4's output (from layer_norm4 backward)
        """
        debug_data['conv4_grad_out'] = grad_input[0].detach().clone() if grad_input[0] is not None else None
        debug_data['conv4_grad_in'] = grad_output[0].detach().clone()

    # 1. Prepare data
    input_data = torch.randn(batch_size, 1, C, T)
    labels = torch.randint(0, N, (batch_size,))

    # Save inputs
    input_file = os.path.join(base_path, "inputs.npz")
    np.savez(input_file, input=input_data.numpy(), labels=labels.numpy())
    print(f"✅ Input data saved to {input_file}")

    # 2. Load weights from ONNX file (including randomized biases)
    onnx_infer_path = os.path.join(base_path, "network_infer.onnx")
    print(f"📥 Loading weights from ONNX: {onnx_infer_path}")
    onnx_model = onnx.load(onnx_infer_path)

    # Convert ONNX weights to dict
    onnx_weights = {}
    for init in onnx_model.graph.initializer:
        onnx_weights[init.name] = numpy_helper.to_array(init)

    # Debug: Print all available weight names
    print(f"📋 Available ONNX weights: {list(onnx_weights.keys())}")

    # 3. Create GroupNorm version of model and load ONNX weights
    model_gn = EpiDeNetDeployGroupNorm(
        C=model_pytorch.C,
        T=model_pytorch.T,
        output_classes=model_pytorch.output_classes,
    )

    # Load ONNX weights into GroupNorm model
    # Conv weights copy directly, LayerNorm weights need reshape and mean
    model_gn.conv1.weight.data = torch.from_numpy(onnx_weights['conv1_weight'].copy())
    model_gn.conv2.weight.data = torch.from_numpy(onnx_weights['conv2_weight'].copy())
    model_gn.conv3.weight.data = torch.from_numpy(onnx_weights['conv3_weight'].copy())
    model_gn.conv4.weight.data = torch.from_numpy(onnx_weights['conv4_weight'].copy())
    model_gn.conv5.weight.data = torch.from_numpy(onnx_weights['conv5_weight'].copy())
    model_gn.fcn.weight.data = torch.from_numpy(onnx_weights['fcn_weight'].copy())
    model_gn.fcn.bias.data = torch.from_numpy(onnx_weights['fcn_bias'].copy())

    # LayerNorm weights [C, H, W] -> GroupNorm weights [C] via mean
    # Note: In ONNX, LayerNorm exported from PyTorch may have different naming
    for i in range(1, 6):
        ln_name = f'layer_norm{i}'
        gn_module = getattr(model_gn, ln_name)

        # Try different possible naming conventions
        weight_key = None
        bias_key = None
        for key in onnx_weights.keys():
            if f'layer_norm{i}' in key and 'weight' in key:
                weight_key = key
            if f'layer_norm{i}' in key and 'bias' in key:
                bias_key = key

        if weight_key and bias_key:
            w = onnx_weights[weight_key]  # [C, H, W]
            b = onnx_weights[bias_key]    # [C, H, W]
            # Convert to [C] by averaging over spatial dimensions
            gn_module.weight.data = torch.from_numpy(w.reshape(w.shape[0], -1).mean(axis=1).copy())
            gn_module.bias.data = torch.from_numpy(b.reshape(b.shape[0], -1).mean(axis=1).copy())
            print(f"✅ Loaded {ln_name} from {weight_key}, {bias_key}")
        else:
            print(f"⚠️ Could not find weights for {ln_name}, using random initialization")

    print(f"✅ Loaded weights from ONNX into GroupNorm model")

    # 3. Execute PyTorch computation (using GroupNorm model)
    model_gn.train()
    criterion = nn.CrossEntropyLoss()
    learning_rate = load_train_config()

    # Register hooks for layer_norm3 debugging
    hook_fwd = model_gn.layer_norm3.register_forward_hook(layer_norm3_forward_hook)
    hook_bwd = model_gn.layer_norm3.register_full_backward_hook(layer_norm3_backward_hook)
    # Register hooks for pool3 to capture relu_2 grad_out
    hook_pool3_fwd = model_gn.pool3.register_forward_hook(pool3_forward_hook)
    hook_pool3_bwd = model_gn.pool3.register_full_backward_hook(pool3_backward_hook)
    # Register hooks for conv4 to capture ConvGradX output
    hook_conv4_bwd = model_gn.conv4.register_full_backward_hook(conv4_backward_hook)

    model_gn.zero_grad()
    output = model_gn(input_data)
    loss = criterion(output, labels)
    loss.backward()

    # Remove hooks
    hook_fwd.remove()
    hook_bwd.remove()
    hook_pool3_fwd.remove()
    hook_pool3_bwd.remove()
    hook_conv4_bwd.remove()

    # DEBUG: Print layer_norm3 inputs/outputs for comparison with Deeploy
    print("\n" + "="*60)
    print("=== DEBUG: layer_norm3 INPUTS/OUTPUTS (PyTorch) ===")
    print("="*60)

    if debug_data['layer_norm3_input'] is not None:
        X = debug_data['layer_norm3_input']
        # GroupNorm input is NCHW, need to transpose to match Deeploy's NCHW layout
        # In Deeploy, data layout after transpose is [N, C, H, W]
        X_flat = X.numpy().flatten()
        print(f"X shape: {X.shape}")
        print(f"X_DATA = [", end="")
        for i, val in enumerate(X_flat[:640]):  # Print first 640 elements
            print(f"{val:.8f}", end="")
            if i < min(639, len(X_flat)-1):
                print(", ", end="")
            if (i+1) % 10 == 0:
                print()
        print("]")

        # Compute mean and inv_std for GroupNorm (num_groups=1)
        # GroupNorm computes stats over (C, H, W) for each N
        N_batch = X.shape[0]
        num_groups = 1
        C_ch = X.shape[1]
        group_size = C_ch * X.shape[2] * X.shape[3]

        for n in range(N_batch):
            x_group = X[n].flatten()
            mean = x_group.mean().item()
            var = x_group.var(unbiased=False).item()
            inv_std = 1.0 / np.sqrt(var + 0.001)  # eps=0.001
            print(f"stat_mean={mean:.8f} stat_inv_std={inv_std:.8f}")

    if debug_data['layer_norm3_grad_output'] is not None:
        dY = debug_data['layer_norm3_grad_output']
        dY_flat = dY.numpy().flatten()
        print(f"dY shape: {dY.shape}")
        print(f"dY_DATA = [", end="")
        for i, val in enumerate(dY_flat[:640]):  # Print first 640 elements
            print(f"{val:.8f}", end="")
            if i < min(639, len(dY_flat)-1):
                print(", ", end="")
            if (i+1) % 10 == 0:
                print()
        print("]")

    # Print dGamma (layer_norm3.weight.grad)
    if model_gn.layer_norm3.weight.grad is not None:
        dGamma = model_gn.layer_norm3.weight.grad.detach().numpy()
        print(f"dGamma shape: {dGamma.shape}")
        print(f"dGamma_PYTORCH = [", end="")
        for i, val in enumerate(dGamma):
            print(f"{val:.8f}", end="")
            if i < len(dGamma)-1:
                print(", ", end="")
        print("]")

    # ============================================================
    # DEBUG: Relu_2 (relu after layer_norm3, before pool3)
    # ============================================================
    print("\n" + "="*60)
    print("=== DEBUG: Relu_2 (node_41) INPUTS/OUTPUTS (PyTorch) ===")
    print("="*60)

    # relu_2 data_in = layer_norm3 output
    if debug_data['relu_2_data_in'] is not None:
        data_in = debug_data['relu_2_data_in']
        data_in_flat = data_in.numpy().flatten()
        print(f"relu_2_data_in shape: {data_in.shape}")
        print(f"data_in = [", end="")
        for i, val in enumerate(data_in_flat):
            print(f"{val:.8f}", end="")
            if i < len(data_in_flat) - 1:
                print(", ", end="")
            if (i + 1) % 10 == 0:
                print()
        print("]")

    # relu_2 grad_out = gradient from pool3
    if debug_data['relu_2_grad_out'] is not None:
        grad_out = debug_data['relu_2_grad_out']
        grad_out_flat = grad_out.numpy().flatten()
        print(f"relu_2_grad_out shape: {grad_out.shape}")
        print(f"grad_out = [", end="")
        for i, val in enumerate(grad_out_flat):
            print(f"{val:.8f}", end="")
            if i < len(grad_out_flat) - 1:
                print(", ", end="")
            if (i + 1) % 10 == 0:
                print()
        print("]")

    # relu_2 grad_in = gradient to layer_norm3
    if debug_data['relu_2_grad_in'] is not None:
        grad_in = debug_data['relu_2_grad_in']
        grad_in_flat = grad_in.numpy().flatten()
        print(f"relu_2_grad_in shape: {grad_in.shape}")
        print(f"grad_in = [", end="")
        for i, val in enumerate(grad_in_flat):
            print(f"{val:.8f}", end="")
            if i < len(grad_in_flat) - 1:
                print(", ", end="")
            if (i + 1) % 10 == 0:
                print()
        print("]")

    # ============================================================
    # DEBUG: Conv4 (node_43 - ConvGradX) INPUTS/OUTPUTS
    # ============================================================
    print("\n" + "="*60)
    print("=== DEBUG: Conv4 (node_43) ConvGradX INPUTS/OUTPUTS (PyTorch) ===")
    print("="*60)

    # conv4 grad_in = gradient from layer_norm4 backward (input to ConvGradX)
    if debug_data['conv4_grad_in'] is not None:
        grad_in = debug_data['conv4_grad_in']
        grad_in_flat = grad_in.numpy().flatten()
        print(f"conv4_grad_in shape: {grad_in.shape}  (ConvGradX INPUT, NCHW)")
        print(f"convgradx_grad_in = [", end="")
        for i, val in enumerate(grad_in_flat):
            print(f"{val:.8f}", end="")
            if i < len(grad_in_flat) - 1:
                print(", ", end="")
            if (i + 1) % 10 == 0:
                print()
        print("]")

    # conv4 grad_out = gradient to pool3 (output of ConvGradX)
    if debug_data['conv4_grad_out'] is not None:
        grad_out = debug_data['conv4_grad_out']
        grad_out_flat = grad_out.numpy().flatten()
        print(f"\nconv4_grad_out shape: {grad_out.shape}  (ConvGradX OUTPUT, NCHW)")
        print(f"convgradx_grad_out = [", end="")
        for i, val in enumerate(grad_out_flat):
            print(f"{val:.8f}", end="")
            if i < len(grad_out_flat) - 1:
                print(", ", end="")
            if (i + 1) % 10 == 0:
                print()
        print("]")

    # ============================================================
    # DEBUG: Pool3 (node_42 - AvgPoolGrad) INPUTS/OUTPUTS
    # ============================================================
    print("\n" + "="*60)
    print("=== DEBUG: Pool3 (node_42) AvgPoolGrad INPUTS/OUTPUTS (PyTorch) ===")
    print("="*60)

    # pool3 forward data (for reference)
    if debug_data['pool3_data_in'] is not None:
        data_in = debug_data['pool3_data_in']
        print(f"pool3_data_in shape: {data_in.shape}  (H_in=2, W_in=20, C=16 in HWC)")

    if debug_data['pool3_data_out'] is not None:
        data_out = debug_data['pool3_data_out']
        print(f"pool3_data_out shape: {data_out.shape}  (H_out=2, W_out=5, C=16 in HWC)")

    # pool3 grad_out = gradient from conv4 backward = AvgPoolGrad INPUT
    if debug_data['pool3_grad_out'] is not None:
        grad_out = debug_data['pool3_grad_out']
        grad_out_flat = grad_out.numpy().flatten()
        print(f"\npool3_grad_out shape: {grad_out.shape}  (AvgPoolGrad INPUT from conv4)")
        print(f"avgpool_grad_in = [", end="")
        for i, val in enumerate(grad_out_flat):
            print(f"{val:.8f}", end="")
            if i < len(grad_out_flat) - 1:
                print(", ", end="")
            if (i + 1) % 10 == 0:
                print()
        print("]")

    # pool3 grad_in = gradient to relu_2 = AvgPoolGrad OUTPUT
    if debug_data['pool3_grad_in'] is not None:
        grad_in = debug_data['pool3_grad_in']
        grad_in_flat = grad_in.numpy().flatten()
        print(f"\npool3_grad_in shape: {grad_in.shape}  (AvgPoolGrad OUTPUT to relu_2)")
        print(f"avgpool_grad_out = [", end="")
        for i, val in enumerate(grad_in_flat):
            print(f"{val:.8f}", end="")
            if i < len(grad_in_flat) - 1:
                print(", ", end="")
            if (i + 1) % 10 == 0:
                print()
        print("]")

    print("="*60)
    print("=== END DEBUG ===")
    print("="*60 + "\n")

    # 4. Selectively extract gradients and execute SGD updates
    import numpy as np
    sgd_outputs = {}

    # Map PyTorch parameter names to ONNX name patterns
    param_map = {
        'conv1.weight': 'conv1_weight',
        'conv2.weight': 'conv2_weight',
        'conv3.weight': 'conv3_weight',
        'conv4.weight': 'conv4_weight',
        'conv5.weight': 'conv5_weight',
        'fcn.weight': 'fcn_weight',
        'fcn.bias': 'fcn_bias',
        'layer_norm1.weight': 'layer_norm1_weight',
        'layer_norm2.weight': 'layer_norm2_weight',
        'layer_norm3.weight': 'layer_norm3_weight',
        'layer_norm4.weight': 'layer_norm4_weight',
        'layer_norm5.weight': 'layer_norm5_weight',
    }

    for pytorch_name, param in model_gn.named_parameters():
        base_onnx_name = param_map.get(pytorch_name)
        if base_onnx_name is None:
            continue

        # Find matching ONNX name in requires_grad_list (may have suffix due to shared initializers)
        matching_onnx_names = [n for n in requires_grad_list if n == base_onnx_name or n.startswith(base_onnx_name + "_")]

        if matching_onnx_names and param.grad is not None:
            grad = param.grad.detach().numpy()
            param_value = param.detach().numpy()
            param_updated = param_value - learning_rate * grad

            # Apply update to all matching ONNX parameters (handles shared weights)
            for onnx_name in matching_onnx_names:
                updated_name = f"{onnx_name}_updated"
                sgd_outputs[updated_name] = param_updated
                print(f"✅ Applied SGD update to {onnx_name}")

    output_file = os.path.join(base_path, "outputs.npz")
    if sgd_outputs:
        np.savez(output_file, **sgd_outputs)
        print(f"✅ Updated parameters saved to {output_file}")
    else:
        print("⚠️ No parameters were updated")

    return input_file


def generate_epidenet_training_onnx(save_path=None):
    """Generate ONNX training model for EpiDeNet based on config"""

    (
        pretrained,
        C,
        T,
        N,
        batch_size,
        opset_version,
    ) = load_config()

    # Input shape for EOG data: (batch_size, 1, channels, time_samples)
    input_shape = (batch_size, 1, C, T)

    folder_name = f"EPIDENET_train_C{C}_T{T}_N{N}"

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

    # Create deployment-optimized EpiDeNet model
    model = EpiDeNetDeploy(
        C=C,
        T=T,
        output_classes=N,
    )
    model.train()

    # Generate random input data for export (EOG signals)
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
    print(f"✅ Inference ONNX model saved to {onnx_infer_file}")

    # Fix shared initializers and rename
    fix_shared_initializers_by_node_name(onnx_infer_file, onnx_infer_file)
    rename_and_save_onnx(onnx_infer_file, onnx_infer_file)

    print_onnx_shapes(onnx_infer_file)
    onnx_model = onnx.load(onnx_infer_file)

    # Get all parameter names
    all_param_names = [init.name for init in onnx_model.graph.initializer]
    print(f"📋 All Parameters: {all_param_names}")

    # Define trainable parameters
# For EpiDeNet, you can customize which layers to train
    # By default, we train all convolutional and fully connected layers
    # Note: layer_norm4 and layer_norm5 share weights (same shape), so ONNX splits them
    trainable_param_patterns = [
        # Classifier
        "fcn_weight",
        "fcn_bias",

        # Conv layers
        "conv1_weight",
        "conv2_weight",
        "conv3_weight",
        "conv4_weight",
        "conv5_weight",

        # LayerNorm weights (only train layers after pool3 to avoid ReluGrad size mismatch)
        "layer_norm1_weight",
        # "layer_norm1_bias",
        "layer_norm2_weight",
        # "layer_norm2_bias"
        "layer_norm3_weight",  # Disabled: causes ReluGrad input/output size mismatch through pool3
        # "layer_norm3_bias",
        "layer_norm4_weight",
        # "layer_norm4_bias",
        "layer_norm5_weight",
        # "layer_norm5_bias"
    ]

    requires_grad = []
    for name in all_param_names:
        for pattern in trainable_param_patterns:
            if name == pattern or name.startswith(pattern + "_"):
                requires_grad.append(name)
                break

    frozen_params = [name for name in all_param_names if name not in requires_grad]

    print(f"🔧 Training Parameters: {requires_grad}")
    print(f"❄️  Frozen Parameters: {frozen_params}")

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

    # Load the training model and add gradient outputs
    onnx_model = onnx.load(onnx_train_file)
    graph = onnx_model.graph
    grad_tensor_names = [name + "_grad" for name in requires_grad]

    for grad_name in grad_tensor_names:
        if not any(output.name == grad_name for output in graph.output):
            grad_output = helper.make_tensor_value_info(
                grad_name, onnx.TensorProto.FLOAT, None
            )
            graph.output.append(grad_output)

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

    # Convert LayerNorm to GroupNorm for deployment FIRST
    # This must happen before add_sgd_nodes so that SGD nodes use correct [C] shapes
    # instead of original [C, H, W] shapes
    convert_layernorm_to_groupnorm(onnx_output_file, onnx_output_file, num_groups=1)
    print(f"✅ Converted LayerNorm to GroupNorm in {onnx_output_file}")

    # Split GroupNorm into GroupNormalizationStat + GroupNormalization
    # This makes forward stat available for backward pass
    split_gn_to_single_stat_array(onnx_output_file, onnx_output_file)
    print(f"✅ Split GroupNorm into Stat + Norm in {onnx_output_file}")

    # Now add SGD optimizer nodes (after conversion, so shapes are correct)
    learning_rate = load_train_config()
    add_sgd_nodes(onnx_output_file, onnx_output_file, learning_rate=learning_rate)

    # Adjust Shape and Type Inference
    infer_shapes_with_custom_ops(onnx_output_file, onnx_output_file)
    type_inference(onnx_output_file, onnx_output_file)
    print(f"✅ Added SGD nodes to {onnx_output_file}")
    ensure_all_tensor_shapes(onnx_output_file, onnx_output_file)

    print(f"\n🎉 EpiDeNet training model generation complete!")
    print(f"📁 Output directory: {base_path}")
    print(f"📄 Files generated:")
    print(f"   - network_infer.onnx: Inference model")
    print(f"   - network_train.onnx: Training model (before optimization)")
    print(f"   - network_pre_sgd.onnx: Training model (before SGD)")
    print(f"   - network.onnx: Final training model with SGD")


if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_epidenet_training_onnx(save_path)
