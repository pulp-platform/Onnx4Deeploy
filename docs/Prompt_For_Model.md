# Prompt for Adding New Models to Onnx4Deeploy

## Standard Prompt Template

Please create a new neural network model **[ModelName]** under the directory  
`/app/Onnx4Deeploy/onnx4deeploy/models`, including both the PyTorch model  
implementation and the ONNX exporter.

The model code is provided below:

`[Paste your PyTorch model code here]`

---

## Requirements

### 1. ONNX Export Rules (Must Be Strictly Followed)

- ✅ **No dynamic shape operations**  
    Avoid runtime dimension queries such as `x.size()`, `x.shape[i]`, etc.

- ✅ **No dynamic padding**  
    All padding values must be constants.

- ✅ **All dimensions must be fixed**  
    Define all dimension constants (e.g., `self.B`, `self.T`, `self.D`) in `__init__`.

- ✅ **Remove dropout**  
    Deployment versions do not require dropout, or set it to `0.0`.

- ✅ **Avoid dynamic reshape**  
    Use fixed-dimension `reshape` or `nn.Flatten`.

- ✅ **inplace=False**  
    All activation functions (e.g., ReLU) must use `inplace=False` for ONNX compatibility.

- ✅ **Canonical PyTorch style**  
    Use standard, best-practice PyTorch coding patterns.


---

### 2. Files to Create

#### Directory Structure

`onnx4deeploy/models/ ├── pytorch_models/ │   └── [model_name_lowercase]/ │       ├── __init__.py │       └── [model_name_lowercase].py   # PyTorch model implementation └── [model_name_lowercase]_exporter.py  # ONNX Exporter`

#### Files to Modify

1. `onnx4deeploy/models/__init__.py` – add exporter imports

2. `Onnx4Deeploy.py` – add CLI entry support


---

## Coding Conventions

### PyTorch Model File

`pytorch_models/[model_name]/[model_name].py`

- Add a detailed docstring stating this is the **Deploy Version**

- Clearly describe architecture hierarchy and dimension changes

- Define all constant dimensions in `__init__`

- Implement `_initialize_weights()` (Kaiming / Xavier initialization)

- Add detailed dimension comments in `forward()`


---

### Exporter File

`[model_name]_exporter.py`

Must:

- Inherit from `BaseONNXExporter`

- Implement:

    - `load_config()`

    - `create_model()`

    - `get_input_shape()`

    - `get_trainable_params()`

    - `_get_config_string()`

    - `save_test_data()`


---

## Reference Files

- Simple model  
    `/app/Onnx4Deeploy/onnx4deeploy/models/simple_mlp_exporter.py`

- CNN model  
    `/app/Onnx4Deeploy/onnx4deeploy/models/lightweight_cnn_exporter.py`

- Transformer model  
    `/app/Onnx4Deeploy/onnx4deeploy/models/sleep_convit_exporter.py`

- ONNX rules  
    `/app/Onnx4Deeploy/docs/MAMBA_NO_DYNAMIC_OPS.md`


---

## Notes

- Keep dropout parameters but set them to `0.0` (API compatibility)

- Use `nn.Flatten(start_dim=1)` instead of `x.view(x.size(0), -1)`

- Avoid `x.size()` and `x.shape[i]`

- Use fixed padding values only

- All activation functions must use `inplace=False`


---

## Usage Examples

### Example 1: Add LightweightCNN

`Please create a new neural network model LightweightCNN under /app/Onnx4Deeploy/onnx4deeploy/models, including the PyTorch implementation and the ONNX exporter.  The model code is provided below:  [Paste LightweightCNN code here]  Requirements: - No dynamic shape operations - No dynamic padding - Canonical PyTorch style - Remove dropout - Add exporter and CLI entry support`

---

### Example 2: Add SleepConViT

`Please create a new neural network model SleepConViT under /app/Onnx4Deeploy/onnx4deeploy/models.  This is a Vision Transformer model for sleep stage classification.  The model code is provided below:  [Paste SleepConViT code here]  Requirements: - No dynamic shape operations - No dynamic padding - Canonical PyTorch style - Remove dropout (or set to 0) - Use inplace=False for all ReLU - Add exporter and CLI entry support`

---

## Key Principles Summary

### ❌ Patterns to Avoid

`# Dynamic shape operations x = x + self.pos_embedding[:, :x.size(1), :] batch_size = x.size(0) x = x.reshape(batch_size, -1)  # Dynamic padding padding = x.size(1) // 2 x = F.pad(x, (padding, padding))  # Dropout self.dropout = nn.Dropout(0.1) x = self.dropout(x)  # inplace=True self.relu = nn.ReLU(inplace=True)`

---

### ✅ Recommended Patterns

`# Fixed dimensions self.batch_size = batch_size self.seq_len = seq_len x = x.reshape(self.batch_size, self.seq_len, -1)  # Fixed padding self.padding = 12 x = F.pad(x, (self.padding, self.padding))  # Dropout removed (or set to 0.0) # self.dropout = nn.Dropout(0.0)  # inplace=False self.relu = nn.ReLU(inplace=False)  # Flatten self.flatten = nn.Flatten(start_dim=1) x = self.flatten(x)`

---

## File Templates

### `__init__.py`

`"""[ModelName] PyTorch model."""  from .[model_name] import [ModelName]  __all__ = ["[ModelName]"]`

---

### PyTorch Model Template

`"""[ModelName] - [Description] (Deploy Version)."""  import torch import torch.nn as nn import torch.nn.init as init  class [ModelName](nn.Module):     """     [ModelName] for [task] (Deploy Version).      Optimized for ONNX export:     - No dropout     - Fixed shapes     - Static padding     - Linear computation graph     """      def __init__(self, [parameters...]):         super().__init__()          # Fixed dimensions         self.batch_size = batch_size          # Layers         ...          self._initialize_weights()      def _initialize_weights(self):         for m in self.modules():             if isinstance(m, (nn.Conv2d, nn.Conv1d)):                 init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")                 if m.bias is not None:                     init.constant_(m.bias, 0)             elif isinstance(m, nn.Linear):                 init.xavier_normal_(m.weight)                 if m.bias is not None:                     init.constant_(m.bias, 0)      def forward(self, x):         """         Args:             x: Input tensor [...]          Returns:             Output tensor [...]         """         return x`

---

## Verification Checklist

-  PyTorch model file created

-  `__init__.py` created

-  Exporter file created

-  `models/__init__.py` updated

-  `Onnx4Deeploy.py` updated

-  No dynamic shape operations

-  No dynamic padding

-  All dimensions fixed

-  Dropout removed or set to `0.0`

-  Activations use `inplace=False`

-  Weight initialization implemented

-  Detailed docstrings added
