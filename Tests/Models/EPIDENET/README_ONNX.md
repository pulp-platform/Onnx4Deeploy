# EpiDeNet ONNX Export

This directory contains scripts to export EpiDeNet models to ONNX format for both inference and training.

## Model Architecture

**EpiDeNet** is a CNN-based model for EOG (Electrooculography) signal classification:

- **Input**: (batch, 1, C, T) - EOG signals
  - C: Number of channels (default: 16)
  - T: Number of time samples (default: 1000, representing 2 seconds at 500Hz)
- **Output**: (batch, N) - Class logits (default: 11 classes)

### Model Versions

1. **EpiDeNet** - Training version with BatchNorm and optional Dropout
2. **EpiDeNetInference** - Inference version with explicit padding
3. **EpiDeNetDeploy** - Deployment-optimized version with fused BatchNorm

## Configuration

Edit `config.yaml` to customize model parameters:

```yaml
epidenet:
  pretrained: False        # Path to pretrained model or False
  C: 16                    # Number of EOG channels
  T: 1000                  # Number of time samples
  N: 11                    # Number of classes
  p_dropout: 0.0           # Dropout probability
  batch_size: 1            # Batch size for ONNX export
  opset_version: 12        # ONNX opset version

training:
  learning_rate: 0.001     # Learning rate for SGD optimizer
```

## Usage

### Generate Inference ONNX

Generate a deployment-optimized ONNX model for inference:

```bash
# Use default config.yaml
python testinfergenerate.py

# Or specify custom output path
python testinfergenerate.py /path/to/output/directory
```

This generates:
- `network.onnx` - Inference model with fused BatchNorm
- `inputs.npz` - Test input data
- `outputs.npz` - Test output data

### Generate Training ONNX

Generate an ONNX model for on-chip training:

```bash
# Use default config.yaml
python testtraingenerate.py

# Or specify custom output path
python testtraingenerate.py /path/to/output/directory
```

This generates:
- `network_infer.onnx` - Inference model
- `network_train.onnx` - Training model (before optimization)
- `network_pre_sgd.onnx` - Training model (before SGD)
- `network.onnx` - Final training model with SGD optimizer
- `inputs.npz` - Test input and label data
- `outputs.npz` - Test gradient outputs

## Model Structure

```
Input (1, 1, 16, 1000)
    ↓
Conv1 (1→4, kernel 1×4) + BN + ReLU + MaxPool(1×8)
    ↓
Conv2 (4→16, kernel 1×16) + BN + ReLU + MaxPool(1×4)
    ↓
Conv3 (16→16, kernel 1×8) + BN + ReLU + MaxPool(1×4)
    ↓
Conv4 (16→16, kernel 16×1, spatial) + BN + ReLU + MaxPool(1×1)
    ↓
Conv5 (16→16, kernel 1×1) + BN + ReLU + AdaptiveAvgPool
    ↓
Flatten + FC (16→11)
    ↓
Output (1, 11)
```

## Example: Load Pretrained Model

To export a pretrained model:

```python
import torch
from epidenet_model.epidenet import epidenet_small, epidenet_deploy

# Load pretrained training model
train_model = epidenet_small(
    pretrained='path/to/checkpoint.pth',
    C=16,
    T=1000,
    N=11
)

# Create deployment model with fused BatchNorm
deploy_model = epidenet_deploy(
    pretrained_model=train_model,
    C=16,
    T=1000,
    N=11
)

# Export to ONNX
dummy_input = torch.randn(1, 1, 16, 1000)
torch.onnx.export(
    deploy_model,
    dummy_input,
    "epidenet_deploy.onnx",
    opset_version=12,
    input_names=["input"],
    output_names=["output"]
)
```

## Directory Structure

```
EPIDENET/
├── epidenet_model/
│   ├── __init__.py
│   └── epidenet.py          # Model definitions
├── utils/                    # Symlink to MI-BMInet utils
├── config.yaml               # Configuration file
├── testinfergenerate.py      # Generate inference ONNX
├── testtraingenerate.py      # Generate training ONNX
├── README_ONNX.md            # This file
└── onnx/                     # Generated ONNX models
    ├── EPIDENET_EOG_C16_T1000_N11/
    └── EPIDENET_train_C16_T1000_N11/
```

## Dependencies

```bash
pip install torch torchvision onnx onnxruntime onnxsim pyyaml
```

## Notes

1. **BatchNorm Fusion**: The deployment model fuses BatchNorm into Conv layers for faster inference
2. **No Dropout**: Dropout is removed in inference/deployment models
3. **Fixed Batch Size**: ONNX models use fixed batch size (default: 1) for cleaner graphs
4. **Trainable Parameters**: Edit `testtraingenerate.py` to customize which layers to train

## Testing

After generation, test the ONNX model:

```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession("onnx/EPIDENET_EOG_C16_T1000_N11/network.onnx")

# Load test data
data = np.load("onnx/EPIDENET_EOG_C16_T1000_N11/inputs.npz")
input_data = data['input']

# Run inference
output = session.run(None, {"input": input_data})[0]
print(f"Output shape: {output.shape}")
print(f"Predicted class: {np.argmax(output, axis=1)}")
```
