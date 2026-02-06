# Mamba Clean ONNX - No Dynamic Operations

## Optimizations Applied

### ❌ Removed All Dynamic Operations

**Problem**: Dynamic operations create Shape/Gather/Const nodes in ONNX.

**Solution**: Use only static operations that don't depend on runtime tensor shapes.

### Key Changes:

#### 1. **Removed `x.size()` / `x.shape` Operations**

**Before (Bad)**:
```python
x = x + self.pos_embedding.narrow(1, 0, x.size(1))  # Dynamic!
```

**After (Good)**:
```python
x = x + self.pos_embedding  # Static - assumes fixed max_seq_len
```

#### 2. **No Dynamic Slicing**

**Before (Bad)**:
```python
pos_emb = self.pos_embedding[:, :seq_len, :]  # Dynamic slicing
```

**After (Good)**:
```python
# Assume input always has shape (batch, max_seq_len, d_model)
# No slicing needed
x = x + self.pos_embedding
```

#### 3. **Fixed Padding (Not Dynamic)**

**Good Practice**:
```python
# Fixed padding value - known at export time
x_padded = F.pad(x_proj_t, (self.conv_padding, 0), value=0.0)
```

#### 4. **torch.chunk Instead of Dynamic Split**

**Good Practice**:
```python
# Chunk splits on a fixed dimension
x_proj, res = torch.chunk(x_and_res, 2, dim=-1)
```

### What Causes Dynamic Graphs:

| ❌ Avoid | ✅ Use Instead |
|----------|----------------|
| `x.size()` | Fixed constants from config |
| `x.shape[1]` | Use fixed `max_seq_len` |
| `.narrow(1, 0, length)` | Direct indexing with fixed size |
| `x[:, :length, :]` | Assume fixed input shape |
| `.reshape(batch, -1, d)` where batch is dynamic | Pre-define all shapes |
| `unsqueeze()` on parameters | Pre-initialize with correct shape |

## ONNX Export Assumptions

For clean ONNX export, we assume:

1. **Fixed Input Shape**: `(batch_size, max_seq_len, d_model)`
   - `batch_size = 1`
   - `max_seq_len = 512`
   - `d_model = 256`

2. **No Variable-Length Sequences**: All sequences are padded to `max_seq_len`

3. **Static Graph**: All operations have compile-time known shapes

## Expected ONNX Graph Structure

### Per Layer (~10-15 nodes):

```
LayerNorm
  ↓
Linear (in_proj) ← MatMul + Add
  ↓
Transpose
  ↓
Pad (fixed padding=3)
  ↓
Conv1d
  ↓
Transpose
  ↓
Silu
  ↓
Linear (x_proj) ← MatMul
  ↓
Chunk → [B, C]
  ↓
Linear (dt_proj) ← MatMul + Add
  ↓
Softplus
  ↓
Exp (A_log → A)
  ↓
SelectiveSSM ← Custom operator (single node!)
  ↓
Silu
  ↓
Mul
  ↓
Linear (out_proj) ← MatMul
  ↓
Add (residual)
```

### What You WON'T See:

- ❌ Shape operators
- ❌ Gather operators (extracting dimensions)
- ❌ Const/ConstantOfShape (except for normal constants)
- ❌ Reshape with dynamic dimensions
- ❌ Unsqueeze/Squeeze chains
- ❌ Slice with dynamic indices

## Verification

After export, check the ONNX graph:

```python
import onnx

model = onnx.load("network.onnx")

# Count node types
op_counts = {}
for node in model.graph.node:
    op = node.op_type
    op_counts[op] = op_counts.get(op, 0) + 1

# Check for dynamic operations
dynamic_ops = ['Shape', 'Gather', 'ConstantOfShape', 'Reshape', 'Slice']
has_dynamic = any(op in op_counts for op in dynamic_ops)

if has_dynamic:
    print("⚠️  Warning: Dynamic operations found")
    for op in dynamic_ops:
        if op in op_counts:
            print(f"   {op}: {op_counts[op]}")
else:
    print("✅ Clean graph - no dynamic operations")

# Check for custom operators
custom_ops = [n for n in model.graph.node if 'SelectiveSSM' in n.op_type]
print(f"✅ Found {len(custom_ops)} SelectiveSSM custom operators")
```

## Configuration

All shapes are defined at initialization:

```python
config = {
    "batch_size": 1,        # Fixed batch size
    "d_model": 256,         # Model dimension
    "n_layers": 4,          # Number of layers
    "d_state": 16,          # SSM state dimension
    "d_conv": 4,            # Conv kernel size
    "max_seq_len": 512,     # FIXED sequence length
    "num_classes": 10,      # Output classes
}
```

## Result

With these optimizations:
- ✅ **Clean ONNX graph** with ~10-15 nodes per layer
- ✅ **No fragmented operations** (Shape/Gather/Const chains)
- ✅ **Single SelectiveSSM** custom operator per layer
- ✅ **Easy to visualize** and debug in Netron
- ✅ **Efficient inference** with fewer graph nodes
