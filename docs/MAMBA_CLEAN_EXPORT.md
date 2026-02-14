# Mamba Clean ONNX Export Guide

## Overview

Mamba model now exports **clean ONNX graphs** using custom operators:
- ✅ Only high-level operators (LayerNorm, Linear, Conv1d, Silu, SelectiveSSM)
- ✅ ~10-15 nodes per layer (not 50-80 fragmented nodes)
- ✅ No excessive Const/Shape/Gather/Reshape nodes
- ✅ Single `ai.mamba::SelectiveSSM` custom operator per layer

## How It Works

### 1. Custom Operator Definition

Located in `onnx4deeploy/models/pytorch_models/mamba/mamba.py`:

```python
class SelectiveSSMFunction(Function):
    @staticmethod
    def forward(ctx, x, A, B, C, D):
        # PyTorch implementation for training/inference
        ...

    @staticmethod
    def symbolic(g, x, A, B, C, D):
        # ONNX export representation
        return g.op("ai.mamba::SelectiveSSM", x, A, B, C, D, outputs=1)
```

**Key Point**: PyTorch automatically uses the `symbolic()` method during ONNX export.

### 2. Export Configuration

In `mamba_exporter.py`:

```python
torch.onnx.export(
    model, input_tensor, output_file,
    opset_version=17,
    custom_opsets={"ai.mamba": 1},  # Register custom operator domain
    do_constant_folding=True,
)
```

### 3. No Graph Optimization

We skip aggressive graph optimizations to preserve custom operators:

```python
def run_inference_optimization(self, onnx_file: str, output_file: str):
    # Skip optimizations that might unfold custom operators
    print("⏭️  Skipping graph optimizations (preserving custom operators)")
    shutil.copy(onnx_file, output_file)
```

## Usage

### Command Line

```bash
python Onnx4Deeploy.py --model mamba
```

### Python API

```python
from onnx4deeploy.models import MambaExporter

exporter = MambaExporter()
onnx_path = exporter.export_inference()
```

## Expected ONNX Structure

### Per Layer (Clean):

```
Input
  ↓
LayerNorm
  ↓
Linear (in_proj)
  ↓
Conv1d
  ↓
Silu
  ↓
Linear (x_proj)
  ↓
ai.mamba::SelectiveSSM  ← Single custom operator!
  ↓
Silu
  ↓
Linear (out_proj)
  ↓
Add (residual)
  ↓
Output
```

**Total: ~10-15 high-level nodes per layer**

## Custom Operator Details

- **Name**: `SelectiveSSM`
- **Domain**: `ai.mamba`
- **Version**: 1
- **Full Name**: `ai.mamba::SelectiveSSM`

### Inputs:
1. `x` - Input tensor (B, L, D)
2. `A` - State matrix (D, N)
3. `B` - Input projection (B, L, N)
4. `C` - Output projection (B, L, N)
5. `D` - Skip connection weights (D,)

### Output:
- `y` - Output tensor (B, L, D)

## Configuration

Edit `load_config()` in `mamba_exporter.py`:

```python
config = {
    "d_model": 256,        # Model dimension
    "n_layers": 4,         # Number of layers
    "d_state": 16,         # SSM state dimension
    "d_conv": 4,           # Conv kernel size
    "max_seq_len": 512,    # Sequence length
    "num_classes": 10,     # Output classes
    "opset_version": 17,   # ONNX opset
}
```

## Benefits

1. **Clean Graph**: Easy to visualize and understand
2. **Efficient**: Fewer nodes = faster inference
3. **Maintainable**: High-level operators are easier to optimize
4. **Portable**: Custom operator can be implemented for any runtime

## Next Steps

To use this ONNX model in production:

1. **Implement the custom operator** in your inference runtime (C++/CUDA)
2. **Register it** with your ONNX Runtime
3. **Run inference** with the clean ONNX graph

See: [ONNX Runtime Custom Operators](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html)
