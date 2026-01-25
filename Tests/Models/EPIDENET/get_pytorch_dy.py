"""
Script to extract dy (gradient) from PyTorch at any specified layer
Uses the same inputs, weights, and configurations as testtraingenerate.py

Usage:
    python get_pytorch_dy.py [base_path] [layer_name]

Examples:
    python get_pytorch_dy.py                                      # Uses layer_norm3 by default
    python get_pytorch_dy.py onnx/EPIDENET_train_C16_T1000_N11 conv3
    python get_pytorch_dy.py onnx/EPIDENET_train_C16_T1000_N11 layer_norm4
    python get_pytorch_dy.py onnx/EPIDENET_train_C16_T1000_N11 relu_2  # For Relu after layer_norm3
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import yaml
import onnx
from onnx import numpy_helper
from epidenet_model.epidenet import EpiDeNetDeployGroupNorm


def load_config(config_filename="config.yaml"):
    """Load and parse config.yaml, returning EpiDeNet-specific parameters"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, config_filename)

    with open(config_file, "r") as f:
        config = yaml.safe_load(f).get("epidenet", {})

    return (
        config.get("C", 16),
        config.get("T", 1000),
        config.get("N", 11),
        config.get("batch_size", 1),
    )


def load_train_config(config_filename="config.yaml"):
    """Load training configuration from config.yaml"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, config_filename)

    with open(config_file, "r") as f:
        config = yaml.safe_load(f).get("training", {})

    return config.get("learning_rate", 0.001)


def get_module_by_name(model, layer_name):
    """Get a module from the model by name, including special handling for ReLU

    Note: ReLU activations in this model are F.relu() calls, not modules.
    For ReLU layers, we hook the previous layer (layer_norm) and track the ReLU through hooks.

    relu_0 -> after layer_norm1
    relu_1 -> after layer_norm2
    relu_2 -> after layer_norm3  (this is what you want for node_41)
    relu_3 -> after layer_norm4
    relu_4 -> after layer_norm5
    """
    # Map ReLU names to the layer_norm before them
    # We'll hook both the norm and pool to get ReLU's input/output
    relu_map = {
        'relu_0': ('layer_norm1', 'pool1'),
        'relu_1': ('layer_norm2', 'pool2'),
        'relu_2': ('layer_norm3', 'pool3'),
        'relu_3': ('layer_norm4', 'pool4'),
        'relu_4': ('layer_norm5', 'pool6'),
    }

    if layer_name in relu_map:
        return relu_map[layer_name], layer_name

    # Try original name
    if hasattr(model, layer_name):
        return getattr(model, layer_name), layer_name

    return None, None


def get_pytorch_dy(base_path, layer_name='layer_norm3'):
    """
    Extract dy (gradient output) from PyTorch at specified layer backward pass.
    Uses the same input data and weights as the training model.

    Args:
        base_path: Path to the ONNX model directory
        layer_name: Name of the layer to extract gradients from (e.g., 'conv3', 'layer_norm4', 'relu_2')
    """

    # Load configuration
    C, T, N, batch_size = load_config()

    print(f"Configuration: C={C}, T={T}, N={N}, batch_size={batch_size}")
    print(f"Target layer: {layer_name}")

    # Storage for debug data
    debug_data = {
        'input': None,       # X - input to target layer
        'output': None,      # forward output
        'grad_output': None, # dY - upstream gradient (gradient flowing INTO this layer from next layer)
        'grad_input': None,  # gradient to previous layer (gradient flowing OUT of this layer to prev layer)
    }

    def forward_hook(module, input, output):
        """Capture input and output of target layer forward pass"""
        debug_data['input'] = input[0].detach().clone()
        debug_data['output'] = output.detach().clone()
        print(f"[HOOK] {layer_name} forward: input shape={input[0].shape}, output shape={output.shape}")

    def backward_hook(module, grad_input, grad_output):
        """Capture gradients in target layer backward pass
        grad_output[0]: gradient flowing INTO this layer (from next layer) = dY for this layer's backward
        grad_input[0]: gradient flowing OUT of this layer (to prev layer) = output of this layer's backward
        """
        debug_data['grad_output'] = grad_output[0].detach().clone()
        if grad_input[0] is not None:
            debug_data['grad_input'] = grad_input[0].detach().clone()
        print(f"[HOOK] {layer_name} backward: grad_output shape={grad_output[0].shape}")
        if grad_input[0] is not None:
            print(f"[HOOK] {layer_name} backward: grad_input shape={grad_input[0].shape}")

    # Load input data
    input_file = os.path.join(base_path, "inputs.npz")
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        print("Please run testtraingenerate.py first to generate input data.")
        return

    input_npz = np.load(input_file)
    input_data = torch.from_numpy(input_npz['input'])
    labels = torch.from_numpy(input_npz['labels'])

    print(f"✅ Loaded input data: input shape={input_data.shape}, labels shape={labels.shape}")

    # Load ONNX weights
    onnx_infer_path = os.path.join(base_path, "network_infer.onnx")
    if not os.path.exists(onnx_infer_path):
        print(f"❌ ONNX file not found: {onnx_infer_path}")
        print("Please run testtraingenerate.py first to generate the ONNX model.")
        return

    print(f"📥 Loading weights from ONNX: {onnx_infer_path}")
    onnx_model = onnx.load(onnx_infer_path)

    # Convert ONNX weights to dict
    onnx_weights = {}
    for init in onnx_model.graph.initializer:
        onnx_weights[init.name] = numpy_helper.to_array(init)

    print(f"✅ Loaded {len(onnx_weights)} weight tensors from ONNX")

    # Create GroupNorm model
    model = EpiDeNetDeployGroupNorm(C=C, T=T, output_classes=N)

    # Load weights into model
    model.conv1.weight.data = torch.from_numpy(onnx_weights['conv1_weight'].copy())
    model.conv2.weight.data = torch.from_numpy(onnx_weights['conv2_weight'].copy())
    model.conv3.weight.data = torch.from_numpy(onnx_weights['conv3_weight'].copy())
    model.conv4.weight.data = torch.from_numpy(onnx_weights['conv4_weight'].copy())
    model.conv5.weight.data = torch.from_numpy(onnx_weights['conv5_weight'].copy())
    model.fcn.weight.data = torch.from_numpy(onnx_weights['fcn_weight'].copy())
    model.fcn.bias.data = torch.from_numpy(onnx_weights['fcn_bias'].copy())

    # Load LayerNorm weights (convert from [C,H,W] to [C])
    for i in range(1, 6):
        ln_name = f'layer_norm{i}'
        gn_module = getattr(model, ln_name)

        weight_key = None
        bias_key = None
        for key in onnx_weights.keys():
            if f'layer_norm{i}' in key and 'weight' in key:
                weight_key = key
            if f'layer_norm{i}' in key and 'bias' in key:
                bias_key = key

        if weight_key and bias_key:
            w = onnx_weights[weight_key]
            b = onnx_weights[bias_key]
            gn_module.weight.data = torch.from_numpy(w.reshape(w.shape[0], -1).mean(axis=1).copy())
            gn_module.bias.data = torch.from_numpy(b.reshape(b.shape[0], -1).mean(axis=1).copy())
            print(f"✅ Loaded {ln_name} weights")

    print("✅ All weights loaded into PyTorch model")

    # Get target module
    target_module, actual_name = get_module_by_name(model, layer_name)
    if target_module is None:
        print(f"❌ Layer '{layer_name}' not found in model")
        print(f"Available layers: {[name for name, _ in model.named_modules()]}")
        return

    # Handle ReLU layers specially (they're tuples of (norm_name, pool_name))
    is_relu = isinstance(target_module, tuple)
    hooks = []

    if is_relu:
        norm_name, pool_name = target_module
        norm_module = getattr(model, norm_name)
        pool_module = getattr(model, pool_name)

        print(f"✅ Found ReLU layer: {actual_name}")
        print(f"   Hooking {norm_name} (output = ReLU input) and {pool_name} (input = ReLU output)")

        # Hook to capture ReLU input (output of layer_norm)
        def relu_input_hook(module, input, output):
            debug_data['input'] = output.detach().clone()  # ReLU input = norm output
            print(f"[HOOK] {actual_name} input (from {norm_name}): shape={output.shape}")

        # Hook to capture ReLU output (input to pool)
        def relu_output_hook(module, input, output):
            debug_data['output'] = input[0].detach().clone()  # ReLU output = pool input
            print(f"[HOOK] {actual_name} output (to {pool_name}): shape={input[0].shape}")

        # Hook to capture ReLU backward: grad from pool, grad to norm
        def relu_backward_hook_pool(module, grad_input, grad_output):
            # grad_output[0] = gradient flowing into pool from next layer
            # This is the gradient that will go through ReLU backward
            debug_data['grad_output'] = grad_output[0].detach().clone()
            print(f"[HOOK] {actual_name} grad_output (from {pool_name}): shape={grad_output[0].shape}")

        def relu_backward_hook_norm(module, grad_input, grad_output):
            # grad_output[0] = gradient flowing into norm from ReLU backward
            # This is the output of ReLU backward
            debug_data['grad_input'] = grad_output[0].detach().clone()
            print(f"[HOOK] {actual_name} grad_input (to {norm_name}): shape={grad_output[0].shape}")

        hooks.append(norm_module.register_forward_hook(relu_input_hook))
        hooks.append(pool_module.register_forward_hook(relu_output_hook))
        hooks.append(pool_module.register_full_backward_hook(relu_backward_hook_pool))
        hooks.append(norm_module.register_full_backward_hook(relu_backward_hook_norm))

    else:
        print(f"✅ Found target layer: {actual_name} ({type(target_module).__name__})")

        # Register hooks
        hooks.append(target_module.register_forward_hook(forward_hook))
        hooks.append(target_module.register_full_backward_hook(backward_hook))

    # Run forward and backward pass
    model.train()
    criterion = nn.CrossEntropyLoss()

    model.zero_grad()
    output = model(input_data)
    loss = criterion(output, labels)

    print(f"\n📊 Forward pass complete: loss={loss.item():.6f}")

    loss.backward()

    print(f"✅ Backward pass complete")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Print results
    print("\n" + "="*80)
    print(f"=== PYTORCH GRADIENT OUTPUT AT {layer_name} ===")
    print("="*80)

    if debug_data['input'] is not None:
        X = debug_data['input']
        print(f"\n[INPUT X] Shape: {X.shape}")
        print(f"Input dtype: {X.dtype}")
        X_flat = X.numpy().flatten()
        print(f"X (first 20 values): {X_flat[:20]}")
        print(f"X statistics: min={X_flat.min():.8f}, max={X_flat.max():.8f}, mean={X_flat.mean():.8f}")

        # For GroupNorm layers, compute stats
        if 'layer_norm' in layer_name or 'norm' in layer_name.lower():
            N_batch = X.shape[0]
            for n in range(N_batch):
                x_group = X[n].flatten()
                mean = x_group.mean().item()
                var = x_group.var(unbiased=False).item()
                inv_std = 1.0 / np.sqrt(var + 0.001)
                print(f"[STATS] Batch {n}: mean={mean:.8f}, inv_std={inv_std:.8f}")

    if debug_data['output'] is not None:
        Y = debug_data['output']
        print(f"\n[OUTPUT Y] Shape: {Y.shape}")
        Y_flat = Y.numpy().flatten()
        print(f"Y (first 20 values): {Y_flat[:20]}")
        print(f"Y statistics: min={Y_flat.min():.8f}, max={Y_flat.max():.8f}, mean={Y_flat.mean():.8f}")

    if debug_data['grad_output'] is not None:
        dY = debug_data['grad_output']
        print(f"\n[GRADIENT dY - Input to backward] Shape: {dY.shape}")
        print(f"Gradient dtype: {dY.dtype}")
        dY_flat = dY.numpy().flatten()

        # Print all values
        print(f"\n{'='*80}")
        print(f"dY VALUES FOR {layer_name} (all {len(dY_flat)} elements):")
        print(f"{'='*80}")
        for i, val in enumerate(dY_flat):
            print(f"{val:.8f}", end="")
            if i < len(dY_flat) - 1:
                print(", ", end="")
            if (i + 1) % 10 == 0:
                print()
        print()

        print(f"\ndY statistics: min={dY_flat.min():.8f}, max={dY_flat.max():.8f}, mean={dY_flat.mean():.8f}")
        print(f"dY L2 norm: {np.linalg.norm(dY_flat):.8f}")

    if debug_data['grad_input'] is not None:
        dX = debug_data['grad_input']
        print(f"\n[GRADIENT dX - Output from backward] Shape: {dX.shape}")
        dX_flat = dX.numpy().flatten()
        print(f"dX (first 20 values): {dX_flat[:20]}")
        print(f"dX statistics: min={dX_flat.min():.8f}, max={dX_flat.max():.8f}, mean={dX_flat.mean():.8f}")

    # If target has learnable parameters, print their gradients (not applicable for ReLU)
    if not is_relu:
        if hasattr(target_module, 'weight') and target_module.weight is not None:
            if target_module.weight.grad is not None:
                dW = target_module.weight.grad.detach().numpy()
                print(f"\n[WEIGHT GRADIENT dW] Shape: {dW.shape}")
                print(f"dW (first 20 values): {dW.flatten()[:20]}")

        if hasattr(target_module, 'bias') and target_module.bias is not None:
            if target_module.bias.grad is not None:
                db = target_module.bias.grad.detach().numpy()
                print(f"\n[BIAS GRADIENT db] Shape: {db.shape}")
                print(f"db values: {db}")

    print("\n" + "="*80)
    print("=== END ===")
    print("="*80)

    # Save gradients to file for comparison
    output_file = os.path.join(base_path, f"pytorch_gradients_{layer_name}.npz")
    save_dict = {}

    if debug_data['input'] is not None:
        save_dict['X'] = debug_data['input'].numpy()
    if debug_data['output'] is not None:
        save_dict['Y'] = debug_data['output'].numpy()
    if debug_data['grad_output'] is not None:
        save_dict['dY'] = debug_data['grad_output'].numpy()
    if debug_data['grad_input'] is not None:
        save_dict['dX'] = debug_data['grad_input'].numpy()

    if not is_relu:
        if hasattr(target_module, 'weight') and target_module.weight is not None:
            if target_module.weight.grad is not None:
                save_dict['dW'] = target_module.weight.grad.detach().numpy()
        if hasattr(target_module, 'bias') and target_module.bias is not None:
            if target_module.bias.grad is not None:
                save_dict['db'] = target_module.bias.grad.detach().numpy()

    np.savez(output_file, **save_dict)
    print(f"\n✅ Gradients saved to {output_file}")


def list_model_layers():
    """List all available layers in the EpiDeNet model"""
    C, T, N, batch_size = load_config()
    model = EpiDeNetDeployGroupNorm(C=C, T=T, output_classes=N)

    print("\n" + "="*80)
    print("Available layers in EpiDeNet model:")
    print("="*80)

    layer_info = []
    for name, module in model.named_modules():
        if name == '':  # Skip root module
            continue
        module_type = type(module).__name__
        layer_info.append((name, module_type))

    # Group by type
    from collections import defaultdict
    by_type = defaultdict(list)
    for name, mtype in layer_info:
        by_type[mtype].append(name)

    # Print by category
    print("\n[Convolutional Layers]")
    for name in by_type.get('Conv2d', []):
        print(f"  - {name}")

    print("\n[GroupNorm Layers]")
    for name in by_type.get('GroupNorm', []):
        print(f"  - {name}")

    print("\n[ReLU Activations] (F.relu calls, not modules)")
    relu_layers = ['relu_0', 'relu_1', 'relu_2', 'relu_3', 'relu_4']
    for name in relu_layers:
        print(f"  - {name} (after layer_norm{relu_layers.index(name)+1})")

    print("\n[Pooling Layers]")
    for name in by_type.get('AvgPool2d', []):
        print(f"  - {name}")

    print("\n[Flatten Layer]")
    for name in by_type.get('Flatten', []):
        print(f"  - {name}")

    print("\n[Fully Connected Layer]")
    for name in by_type.get('Linear', []):
        print(f"  - {name}")

    print("\n" + "="*80)
    print("Usage examples:")
    print("  python get_pytorch_dy.py [base_path] conv3")
    print("  python get_pytorch_dy.py [base_path] layer_norm3")
    print("  python get_pytorch_dy.py [base_path] relu3")
    print("  python get_pytorch_dy.py [base_path] pool3")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Parse arguments
    base_path = None
    layer_name = 'layer_norm3'  # default

    # Check for --list flag
    if '--list' in sys.argv or '-l' in sys.argv:
        list_model_layers()
        sys.exit(0)

    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        base_path = sys.argv[1]
    if len(sys.argv) > 2:
        layer_name = sys.argv[2]

    # If base_path not provided, use default
    if base_path is None:
        C, T, N, batch_size = load_config()
        folder_name = f"EPIDENET_train_C{C}_T{T}_N{N}"
        base_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "onnx",
            folder_name
        )

    print(f"Base path: {base_path}")
    print(f"Layer name: {layer_name}")
    get_pytorch_dy(base_path, layer_name)
