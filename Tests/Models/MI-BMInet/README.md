# MI-BMInet ONNX Graph Generation

This directory contains scripts for generating inference and training graphs for MI-BMInet model.

## Directory Structure

```
MI-BMInet/
├── config.yaml              # Configuration file for model parameters
├── testinfergenerate.py     # Script to generate inference ONNX graph
├── testtraingenerate.py     # Script to generate training ONNX graph
├── mi_bminet_model/         # MI-BMInet model definition
│   ├── __init__.py
│   └── mi_bminet.py        # Model architecture
├── onnx/                    # Output directory for generated ONNX files
└── utils/                   # Symbolic link to CCT/utils

```

## Configuration

Edit [config.yaml](config.yaml) to adjust model parameters:

```yaml
mi_bminet:
  pretrained: False
  img_size: 32          # Input image size
  num_classes: 10       # Number of output classes
  embedding_dim: 128    # Embedding dimension
  num_heads: 1          # Number of attention heads
  num_layers: 2         # Number of layers
  batch_size: 1         # Batch size for training
  opset_version: 12     # ONNX opset version

training:
  learning_rate: 0.01   # Learning rate for SGD optimizer
```

## Usage

### Generate Inference Graph

```bash
cd /app/Onnx4Deeploy/Tests/Models/MI-BMInet
python testinfergenerate.py
```

This will generate:
- `onnx/MI_BMInet_infer_{img_size}_{embedding_dim}_{num_heads}_{num_layers}/network.onnx`
- `onnx/MI_BMInet_infer_{img_size}_{embedding_dim}_{num_heads}_{num_layers}/inputs.npz`
- `onnx/MI_BMInet_infer_{img_size}_{embedding_dim}_{num_heads}_{num_layers}/outputs.npz`

### Generate Training Graph

```bash
cd /app/Onnx4Deeploy/Tests/Models/MI-BMInet
python testtraingenerate.py
```

This will generate:
- `onnx/MI_BMInet_train_{img_size}_{embedding_dim}_{num_heads}_{num_layers}/network.onnx` (training graph with SGD)
- `onnx/MI_BMInet_train_{img_size}_{embedding_dim}_{num_heads}_{num_layers}/network_infer.onnx` (inference version)
- `onnx/MI_BMInet_train_{img_size}_{embedding_dim}_{num_heads}_{num_layers}/network_train.onnx` (training version before optimization)
- Additional training artifacts

### Custom Output Path

You can specify a custom output path:

```bash
python testinfergenerate.py /path/to/output/directory
python testtraingenerate.py /path/to/output/directory
```

## Model Architecture

The MI-BMInet model ([mi_bminet_model/mi_bminet.py](mi_bminet_model/mi_bminet.py)) is a simple CNN-based architecture with:
- Convolutional feature extraction layers
- Batch normalization
- Adaptive pooling
- Fully connected classifier

**Note:** This is a placeholder implementation. Replace with your actual MI-BMInet architecture if you have a specific model definition.

## Training Configuration

In the training graph generation, you can configure which layers to train by modifying the `requires_grad` list in [testtraingenerate.py](testtraingenerate.py):

```python
requires_grad = [
    name
    for name in all_param_names
    if any(keyword in name for keyword in [
        "classifier",  # Train classifier layers
        # Add more keywords to train more layers
    ])
]
```

## Dependencies

- PyTorch
- ONNX
- ONNXRuntime (with training support)
- NumPy
- PyYAML

## Reference

This implementation follows the same structure as the CCT model in `/app/Onnx4Deeploy/Tests/Models/CCT/`.
