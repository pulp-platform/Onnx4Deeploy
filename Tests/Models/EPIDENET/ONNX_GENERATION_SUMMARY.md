# EpiDeNet ONNX Generation - Summary

## ✅ Completed Tasks

Successfully implemented ONNX export functionality for the EpiDeNet EOG classification model, following the MI-BMInet architecture and patterns.

## 📁 Files Created

### Model Definition
- **`epidenet_model/__init__.py`** - Package initialization
- **`epidenet_model/epidenet.py`** - Three model versions:
  - `EpiDeNet` - Training version with BatchNorm and optional Dropout
  - `EpiDeNetInference` - Inference version with explicit padding
  - `EpiDeNetDeploy` - Deployment-optimized with fused BatchNorm

### Configuration & Scripts
- **`config.yaml`** - Model and training configuration
- **`testinfergenerate.py`** - Generate inference ONNX models
- **`testtraingenerate.py`** - Generate training ONNX models
- **`README_ONNX.md`** - Comprehensive usage documentation
- **`ONNX_GENERATION_SUMMARY.md`** - This summary file

## 🔧 Key Improvements

### 1. **Replaced AdaptiveAvgPool2d with Fixed AvgPool2d**
   - **Issue**: AdaptiveAvgPool2d can cause issues with ONNX export
   - **Solution**: Calculated final temporal dimension after pooling (T / 128)
   - **Benefit**: Cleaner ONNX graph, better hardware compatibility

### 2. **Three Model Versions**
   ```
   EpiDeNet (Training)
   ├─ BatchNorm layers
   ├─ Optional Dropout
   └─ Padding='same'

   EpiDeNetInference
   ├─ BatchNorm layers
   ├─ No Dropout
   └─ Explicit padding

   EpiDeNetDeploy (Optimized)
   ├─ Fused BatchNorm into Conv
   ├─ No Dropout
   └─ Explicit padding
   ```

### 3. **Proper Utility Imports**
   - Resolved import conflicts by using absolute paths
   - Imported from CCT/utils directory
   - Used `sys.path.insert(0, ...)` for priority

## 📊 Generated ONNX Models

### Inference Model
```
Location: onnx/EPIDENET_EOG_C16_T1000_N11/
Files:
  - network.onnx (34KB) - Optimized inference model
  - inputs.npz - Test input data
  - outputs.npz - Test output data

Model I/O:
  - Input: (1, 1, 16, 1000) - EOG signals
  - Output: (1, 11) - Class logits
```

### Training Model
```
Location: onnx/EPIDENET_train_C16_T1000_N11/
Files:
  - network.onnx (45KB) - Final training model with SGD
  - network_infer.onnx - Inference-only model
  - network_train.onnx - Training model (pre-optimization)
  - network_pre_sgd.onnx - Before SGD optimizer
  - inputs.npz - Test inputs (data + labels)
  - outputs.npz - Test gradient outputs

Model I/O:
  - Inputs: input (1,1,16,1000), labels (1,)
  - Outputs: 7 parameter updates (gradients)
    - conv1_weight_updated
    - conv2_weight_updated
    - conv3_weight_updated
    - conv4_weight_updated
    - conv5_weight_updated
    - fcn_weight_updated
    - fcn_bias_updated
```

## 🧪 Testing Results

### Inference ONNX Generation
```bash
$ python testinfergenerate.py
✅ Loaded config: C=16, T=1000, N=11, opset_version=12
📦 Creating deployment model with random weights...
✅ ONNX model saved
✅ ONNX model simplified
✅ Output data saved
✅ Successfully unified GEMM input dimensions
```

### Training ONNX Generation
```bash
$ python testtraingenerate.py
✅ Inference ONNX model saved
📋 All Parameters: ['conv1_weight', 'conv1_bias', ...]
🔧 Training Parameters: ['fcn_weight', 'fcn_bias', ...]
❄️  Frozen Parameters: ['conv1_bias', 'conv2_bias', ...]
✅ Added SGD nodes
🎉 EpiDeNet training model generation complete!
```

## 📝 Model Architecture

```
Input (1, 1, 16, 1000)
    ↓
Conv1 (1→4, 1×4) + BN + ReLU + MaxPool(1×8)
    ↓
Conv2 (4→16, 1×16) + BN + ReLU + MaxPool(1×4)
    ↓
Conv3 (16→16, 1×8) + BN + ReLU + MaxPool(1×4)
    ↓
Conv4 (16→16, 16×1) + BN + ReLU + MaxPool(1×1)  [Spatial Conv]
    ↓
Conv5 (16→16, 1×1) + BN + ReLU + AvgPool(1×7)
    ↓
Flatten + FC (16→11)
    ↓
Output (1, 11)
```

## 🔄 Dimension Flow

```
Input:           (1, 1, 16, 1000)
After Conv1:     (1, 4, 16, 1000)
After Pool1:     (1, 4, 16, 125)   [T/8]
After Conv2:     (1, 16, 16, 125)
After Pool2:     (1, 16, 16, 31)   [T/32]
After Conv3:     (1, 16, 16, 31)
After Pool3:     (1, 16, 16, 7)    [T/128]
After Conv4:     (1, 16, 1, 7)     [Spatial reduction]
After Conv5:     (1, 16, 1, 7)
After Pool6:     (1, 16, 1, 1)
After Flatten:   (1, 16)
Output:          (1, 11)
```

## 🎯 Usage Examples

### Generate Inference ONNX
```bash
# Default output location
python testinfergenerate.py

# Custom output location
python testinfergenerate.py /path/to/output/
```

### Generate Training ONNX
```bash
# Default output location
python testtraingenerate.py

# Custom output location
python testtraingenerate.py /path/to/output/
```

### Customize Training Parameters
Edit `testtraingenerate.py` lines 223-240 to select which layers to train:
```python
requires_grad = [
    name for name in all_param_names
    if name in [
        # Train only classifier
        "fcn_weight",
        "fcn_bias",

        # Or add convolutional layers
        "conv1_weight",
        "conv2_weight",
        ...
    ]
]
```

## 📚 Configuration

Edit `config.yaml` to customize:
```yaml
epidenet:
  C: 16                    # Number of EOG channels
  T: 1000                  # Time samples (at 500Hz = 2 seconds)
  N: 11                    # Number of classes
  p_dropout: 0.0           # Dropout probability
  batch_size: 1            # Batch size
  opset_version: 12        # ONNX opset version

training:
  learning_rate: 0.001     # SGD learning rate
```

## ✨ Features Implemented

1. ✅ Three model versions (Training, Inference, Deploy)
2. ✅ BatchNorm fusion for deployment
3. ✅ Configurable training parameters
4. ✅ ONNX simplification and optimization
5. ✅ Test data generation
6. ✅ Gradient output support
7. ✅ SGD optimizer integration
8. ✅ Fixed pooling layers (no AdaptiveAvgPool2d)
9. ✅ Comprehensive documentation

## 🔗 Compatibility

- **PyTorch**: >=2.0.0
- **ONNX**: Opset 12
- **ONNX Runtime**: For inference and training
- **Hardware**: Optimized for edge deployment

## 📖 Additional Resources

- See `README_ONNX.md` for detailed usage instructions
- Model definition: `epidenet_model/epidenet.py`
- Configuration: `config.yaml`

---

**Generated**: 2026-01-12
**Based on**: MI-BMInet architecture patterns
