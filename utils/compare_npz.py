import numpy as np

# Load both npz files
pool2_data = np.load('pool2_gradients.npz')
activation2_data = np.load('activation2_gradients.npz')

print("=== pool2_gradients.npz ===")
print(f"Keys: {list(pool2_data.keys())}")
for key in pool2_data.keys():
    if key != 'meta':
        arr = pool2_data[key]
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
        print(f"  range=[{arr.min():.6f}, {arr.max():.6f}]")
        print(f"  First 10: {arr.flatten()[:10]}")

print("\n=== activation2_gradients.npz ===")
print(f"Keys: {list(activation2_data.keys())}")
for key in activation2_data.keys():
    if key != 'meta':
        arr = activation2_data[key]
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
        print(f"  range=[{arr.min():.6f}, {arr.max():.6f}]")
        print(f"  First 10: {arr.flatten()[:10]}")

print("\n=== Comparing grad_node_input ===")
pool2_grad = pool2_data['grad_node_input']
activation2_grad = activation2_data['grad_node_input']

print(f"pool2 shape: {pool2_grad.shape}")
print(f"activation2 shape: {activation2_grad.shape}")

if pool2_grad.shape == activation2_grad.shape:
    are_equal = np.array_equal(pool2_grad, activation2_grad)
    are_close = np.allclose(pool2_grad, activation2_grad)

    print(f"\nArrays exactly equal: {are_equal}")
    print(f"Arrays close (default tolerance): {are_close}")

    if are_equal:
        print("\n*** WARNING: grad_node_input values are IDENTICAL! ***")
        print("This is unexpected since they're from different nodes (pool2 vs activation2)")
    else:
        diff = np.abs(pool2_grad - activation2_grad)
        print(f"\nMax difference: {diff.max():.6e}")
        print(f"Mean difference: {diff.mean():.6e}")
        print(f"Number of different values: {np.sum(pool2_grad != activation2_grad)}")
else:
    print("Shapes are different - cannot compare directly")

print("\n=== Comparing grad_x ===")
pool2_grad_x = pool2_data['grad_x']
activation2_grad_x = activation2_data['grad_x']

print(f"pool2 grad_x shape: {pool2_grad_x.shape}")
print(f"activation2 grad_x shape: {activation2_grad_x.shape}")

if pool2_grad_x.shape == activation2_grad_x.shape:
    are_equal = np.array_equal(pool2_grad_x, activation2_grad_x)
    print(f"grad_x arrays exactly equal: {are_equal}")
    if not are_equal:
        diff = np.abs(pool2_grad_x - activation2_grad_x)
        print(f"Max difference: {diff.max():.6e}")
        print(f"Mean difference: {diff.mean():.6e}")
