# Optimization Pipeline Guide

This guide explains how to use the flexible optimization pipeline system in Onnx4Deeploy to customize ONNX model optimizations.

## Overview

The optimization pipeline system provides a flexible way to control which optimization passes are applied to your ONNX models and in what order. This replaces the previous hardcoded optimization sequences.

## Key Concepts

### 1. Optimization Pass

An `OptimizationPass` is a single optimization operation (e.g., removing Identity nodes, fusing Reshape operations).

### 2. Optimization Pipeline

An `OptimizationPipeline` is a sequence of optimization passes that are applied in order.

### 3. Pass Configuration

A `PassConfig` controls whether a pass is enabled and provides parameters to the pass.

## Using Default Pipelines

The simplest way is to use the default pipelines:

```python
from onnx4deeploy.models.simple_mlp_exporter import SimpleMlpExporter

# Default inference pipeline is used automatically
exporter = SimpleMlpExporter()
output_file = exporter.export_inference()
```

## Customizing Pipelines

### Method 1: Override get_inference_pipeline() in Your Exporter

```python
from onnx4deeploy.core.base_exporter import BaseONNXExporter
from onnx4deeploy.core.optimization_passes import create_inference_pipeline

class MyModelExporter(BaseONNXExporter):
    def get_inference_pipeline(self):
        # Start with default pipeline
        pipeline = create_inference_pipeline()

        # Disable a specific pass
        pipeline.disable_pass('softmax_axis')

        # Or remove it entirely
        pipeline.remove_pass('biasgelu_opt')

        return pipeline
```

### Method 2: Build Custom Pipeline from Scratch

```python
from onnx4deeploy.core.optimization_pipeline import OptimizationPipeline
from onnx4deeploy.core.optimization_passes import (
    RenameNodesPass,
    RemoveIdentityPass,
    UnifyGemmPass
)

class MyModelExporter(BaseONNXExporter):
    def get_inference_pipeline(self):
        # Create empty pipeline
        pipeline = OptimizationPipeline(name="custom_inference")

        # Add only the passes you want
        pipeline.add_pass(RenameNodesPass())
        pipeline.add_pass(RemoveIdentityPass())
        pipeline.add_pass(UnifyGemmPass())

        return pipeline
```

### Method 3: Configure Pass Parameters

Some passes accept parameters:

```python
from onnx4deeploy.core.optimization_pipeline import PassConfig

class MyModelExporter(BaseONNXExporter):
    def get_inference_pipeline(self):
        pipeline = create_inference_pipeline()

        # Configure the randomize_initializers pass
        pipeline.configure_pass(
            'randomize_initializers',
            PassConfig(enabled=True, params={'seed': 123})
        )

        return pipeline
```

## Example: CCT Transformer Model

CCT uses a transformer-specific pipeline with LayerNorm fusion:

```python
class CCTExporter(BaseONNXExporter):
    def get_inference_pipeline(self):
        # Use transformer-specific pipeline
        pipeline = create_transformer_inference_pipeline(
            embedding_dim=self.config['embedding_dim'],
            num_heads=self.config['num_heads'],
            input_shape=self.get_input_shape()
        )
        return pipeline
```

This pipeline includes:
1. Randomize initializers (for testing)
2. Rename nodes for C compatibility
3. **ONNX Runtime transformer optimization** (includes LayerNorm fusion)
4. Rename nodes again
5. All standard inference optimizations

## Available Standard Passes

### Inference Passes

- `rename_nodes` - Rename nodes for C compatibility
- `remove_identity` - Remove Identity nodes
- `reshape_fusion` - Optimize and fuse Reshape operations
- `unify_gemm` - Unify GEMM input dimensions
- `biasgelu_opt` - Optimize BiasGelu operations
- `squeeze_unsqueeze` - Convert Squeeze/Unsqueeze to attributes
- `sum_to_add` - Convert Sum operations to Add
- `softmax_axis` - Optimize Softmax axis attribute
- `shape_inference` - Run shape inference with custom op support

### Transformer-Specific Passes

- `onnxruntime_transformer` - ONNX Runtime transformer optimization (includes LayerNorm fusion)
  - Requires parameters: `embedding_dim`, `num_heads`, `input_shape`

### Testing Passes

- `randomize_initializers` - Randomize zero initializers for testing
  - Optional parameter: `seed` (default: 42)

### Training Passes

- `training_optimization` - Comprehensive training optimization pipeline
  - Optional parameter: `split_layernormgrad` (default: False)

## Advanced Usage: Adding Custom Passes

You can create your own optimization passes:

```python
from onnx4deeploy.core.optimization_pipeline import OptimizationPass, PassConfig

class MyCustomPass(OptimizationPass):
    def __init__(self):
        super().__init__(
            name="my_custom_pass",
            description="My custom optimization"
        )

    def apply(self, onnx_file: str, output_file: str, config: PassConfig) -> bool:
        try:
            # Load model
            import onnx
            model = onnx.load(onnx_file)

            # Apply your custom optimization
            # ...

            # Save model
            onnx.save(model, output_file)
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

# Use it in your exporter
class MyModelExporter(BaseONNXExporter):
    def get_inference_pipeline(self):
        pipeline = create_inference_pipeline()
        pipeline.add_pass(MyCustomPass())
        return pipeline
```

## Pipeline Methods

### Adding/Removing Passes

```python
pipeline.add_pass(RemoveIdentityPass())  # Add a pass
pipeline.remove_pass('reshape_fusion')    # Remove by name
```

### Enabling/Disabling Passes

```python
pipeline.disable_pass('softmax_axis')  # Disable (skip but keep in pipeline)
pipeline.enable_pass('softmax_axis')   # Re-enable
```

### Configuring Passes

```python
pipeline.configure_pass('randomize_initializers',
                       PassConfig(enabled=True, params={'seed': 999}))
```

### Cloning Pipelines

```python
custom_pipeline = default_pipeline.clone()
custom_pipeline.disable_pass('biasgelu_opt')
```

## Complete Example

Here's a complete example showing different pipeline configurations:

```python
from onnx4deeploy.core.base_exporter import BaseONNXExporter
from onnx4deeploy.core.optimization_passes import (
    create_inference_pipeline,
    RenameNodesPass,
    RemoveIdentityPass,
)
from onnx4deeploy.core.optimization_pipeline import PassConfig

class MyComplexModel(BaseONNXExporter):
    def get_inference_pipeline(self):
        if self.config.get('use_minimal_optimization', False):
            # Minimal pipeline for debugging
            pipeline = OptimizationPipeline(name="minimal")
            pipeline.add_pass(RemoveIdentityPass())
            return pipeline

        elif self.config.get('use_aggressive_optimization', False):
            # Aggressive pipeline with all passes
            pipeline = create_inference_pipeline()
            # All passes enabled by default
            return pipeline

        else:
            # Standard pipeline with some customization
            pipeline = create_inference_pipeline()

            # Disable problematic passes for this model
            pipeline.disable_pass('softmax_axis')

            # Configure specific passes
            pipeline.configure_pass(
                'randomize_initializers',
                PassConfig(enabled=True, params={'seed': 42})
            )

            return pipeline
```

## Benefits

1. **Flexibility**: Easy to add, remove, or reorder optimization passes
2. **Reusability**: Share common pipelines across models
3. **Debugging**: Disable specific passes to identify issues
4. **Testability**: Easy to test individual passes
5. **Extensibility**: Simple to add custom optimization passes
6. **Control**: Fine-grained control over each optimization step

## Migration from Old Code

Old hardcoded approach:
```python
def run_inference_optimization(self, onnx_file, output_file):
    remove_identity_nodes(onnx_file, output_file)
    optimize_reshape_fusion(output_file, output_file)
    unify_gemm_input_dims(output_file, output_file)
    # ... many more hardcoded calls
```

New pipeline approach:
```python
def get_inference_pipeline(self):
    return create_inference_pipeline()  # All passes configured
```

## See Also

- `optimization_pipeline.py` - Pipeline implementation
- `optimization_passes.py` - Standard pass definitions
- `cct_exporter.py` - Example of transformer pipeline
- `simple_mlp_exporter.py` - Example of default pipeline
