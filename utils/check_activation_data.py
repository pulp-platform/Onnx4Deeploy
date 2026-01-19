import numpy as np
import torch
import sys
sys.path.insert(0, '/app/Onnx4Deeploy/Tests/Models/MI-BMInet/mi_bminet_model')
from mi_bminet import MIBMINetDeploy

# Load model and weights
model = MIBMINetDeploy(C=8, T=2000, N=4)
npz_dict = dict(np.load('/app/Onnx4Deeploy/Tests/Models/MI-BMInet/onnx/MI_BMINet_train_C8_T2000_F1_8_D2_N4_fc_sep_conv2/inputs_with_weights.npz'))

# Load weights
for k, v in model.state_dict().items():
    if k in npz_dict:
        model.state_dict()[k].copy_(torch.from_numpy(npz_dict[k]))

# Get input
x = torch.from_numpy(npz_dict['input'])
x.requires_grad_(True)

# Hook to capture activation2 input and output
captured = {}

def hook_activation2(_m, inp, out):
    captured['activation2_input'] = inp[0] if isinstance(inp, (tuple, list)) else inp
    captured['activation2_output'] = out[0] if isinstance(out, (tuple, list)) else out
    if captured['activation2_input'].requires_grad:
        captured['activation2_input'].retain_grad()
    if captured['activation2_output'].requires_grad:
        captured['activation2_output'].retain_grad()

def hook_pool2(_m, inp, out):
    captured['pool2_input'] = inp[0] if isinstance(inp, (tuple, list)) else inp
    captured['pool2_output'] = out[0] if isinstance(out, (tuple, list)) else out
    if captured['pool2_input'].requires_grad:
        captured['pool2_input'].retain_grad()

h1 = model.activation2.register_forward_hook(hook_activation2)
h2 = model.pool2.register_forward_hook(hook_pool2)

# Forward pass
labels = torch.from_numpy(npz_dict['labels'])
output = model(x)

# Compute loss
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(output, labels)

# Backward
loss.backward()

h1.remove()
h2.remove()

# Analysis
print("=== Forward Pass Analysis ===")
print(f"activation2_input shape: {captured['activation2_input'].shape}")
print(f"activation2_output shape: {captured['activation2_output'].shape}")
print(f"pool2_input shape: {captured['pool2_input'].shape}")

# Check if activation2_output and pool2_input are the same
act2_out = captured['activation2_output']
pool2_in = captured['pool2_input']
print(f"\nactivation2_output == pool2_input? {torch.equal(act2_out, pool2_in)}")
print(f"activation2_output is pool2_input? {act2_out is pool2_in}")

print("\n=== Gradient Analysis ===")
if captured['activation2_input'].grad is not None:
    grad_act2_in = captured['activation2_input'].grad.numpy()
    print(f"grad(activation2_input) shape: {grad_act2_in.shape}")
    print(f"  First 10: {grad_act2_in.flatten()[:10]}")
else:
    print("grad(activation2_input) is None")

if captured['activation2_output'].grad is not None:
    grad_act2_out = captured['activation2_output'].grad.numpy()
    print(f"grad(activation2_output) shape: {grad_act2_out.shape}")
    print(f"  First 10: {grad_act2_out.flatten()[:10]}")
else:
    print("grad(activation2_output) is None")

if captured['pool2_input'].grad is not None:
    grad_pool2_in = captured['pool2_input'].grad.numpy()
    print(f"grad(pool2_input) shape: {grad_pool2_in.shape}")
    print(f"  First 10: {grad_pool2_in.flatten()[:10]}")
else:
    print("grad(pool2_input) is None")

# Compare gradients
print("\n=== Gradient Comparison ===")
if captured['activation2_output'].grad is not None and captured['pool2_input'].grad is not None:
    print(f"grad(activation2_output) == grad(pool2_input)? {torch.equal(captured['activation2_output'].grad, captured['pool2_input'].grad)}")

# Check ReLU behavior
act2_in_np = captured['activation2_input'].detach().numpy()
act2_out_np = captured['activation2_output'].detach().numpy()
print(f"\n=== ReLU Behavior ===")
print(f"activation2_input min/max: [{act2_in_np.min():.6f}, {act2_in_np.max():.6f}]")
print(f"activation2_output min/max: [{act2_out_np.min():.6f}, {act2_out_np.max():.6f}]")
print(f"Number of negative inputs: {np.sum(act2_in_np < 0)} / {act2_in_np.size}")
print(f"Number of zeros in output: {np.sum(act2_out_np == 0)} / {act2_out_np.size}")

# Load the npz files and compare
pool2_data = np.load('pool2_gradients.npz')
activation2_data = np.load('activation2_gradients.npz')

print("\n=== Comparing with saved npz files ===")
print(f"pool2_gradients.npz grad_node_input matches grad(pool2_input)? {np.allclose(pool2_data['grad_node_input'], grad_pool2_in)}")
print(f"activation2_gradients.npz grad_node_input matches grad(activation2_input)? {np.allclose(activation2_data['grad_node_input'], grad_act2_in)}")
