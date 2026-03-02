# Training Graph Generation Guide

This guide covers every option available when generating ONNX training graphs with
Onnx4Deeploy, from CLI flags to Python-level configuration and extension points.

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Stages](#pipeline-stages)
3. [Generated Files](#generated-files)
4. [CLI Reference](#cli-reference)
5. [Configuration Keys](#configuration-keys)
6. [Training Strategy](#training-strategy)
7. [Gradient Accumulation](#gradient-accumulation)
8. [Data Source](#data-source)
9. [Test Data Layout (inputs.npz / outputs.npz)](#test-data-layout)
10. [Extending: Custom Exporters](#extending-custom-exporters)
11. [Extending: Custom Data Sources](#extending-custom-data-sources)
12. [Complete Examples](#complete-examples)

---

## Overview

The training export pipeline converts a PyTorch model into three ONNX graphs
suitable for on-device training with Deeploy / ORT:

```
PyTorch model
    │
    ▼  torch.onnx.export (training mode)
network_infer.onnx          ← forward graph with all initializers
    │
    ▼  onnxruntime.training.artifacts.generate_artifacts
network_train.onnx          ← forward + loss + backward graph (ORT format)
    │
    ▼  Deeploy training optimizations
network.onnx                ← Deeploy-ready training graph (final)
    │
    ▼  ORT InferenceSession simulation
inputs.npz / outputs.npz   ← reference test data for on-device verification
```

---

## Pipeline Stages

### Stage 1 — PyTorch → ONNX (inference graph)

`torch.onnx.export` is called in `TrainingMode.TRAINING` so that ops like
LayerNorm and Dropout export their training-specific outputs (saved mean,
inv_std_var, etc.) required by ORT's gradient builders.

For opset ≥ 17, a custom symbolic override forces `LayerNormalization` to
declare three outputs, ensuring ORT can attach `LayerNormalizationGrad`.

### Stage 2 — Inference optimizations

The inference optimization pipeline runs on `network_infer.onnx` before passing
it to `generate_artifacts`.  The exact passes depend on the model (see
`get_inference_pipeline()` in the exporter class).  During training export the
flag `_for_training=True` is set so model-specific subclasses can skip ORT
transformer fusion that would create `com.microsoft` custom ops incompatible
with `generate_artifacts`.

### Stage 3 — ORT artifact generation

`onnxruntime.training.artifacts.generate_artifacts` produces:

| File | Contents |
|------|----------|
| `network_train.onnx` | Forward + SoftmaxCELoss + backward graph, ORT format |
| `eval_model.onnx` | Forward + loss only (no gradients) |
| `optimizer_model.onnx` | SGD parameter-update graph |
| `checkpoint/` | Initial parameter values |

Which parameters are trained vs frozen is controlled by
[`training_strategy`](#training-strategy).

### Stage 4 — Deeploy training optimizations

`run_training_optimization` applies Deeploy-specific graph transforms to
`network_train.onnx` and saves the result as `network_train_optim.onnx`, which
is then copied to `network.onnx` (the final model consumed by
`generateTrainingNetwork.py`).

### Stage 5 — Reference test data generation

`create_training_test_data()` runs the training graph through an ORT
`InferenceSession` to produce `inputs.npz` and `outputs.npz`.  These files are
used by the Deeploy test runner to verify on-device correctness.

---

## Generated Files

All files land in the output directory (default:
`./onnx/model/<ModelName>_train<config_string>/`).

| File | Description |
|------|-------------|
| `network.onnx` | Final Deeploy-ready training graph (use this with `generateTrainingNetwork.py`) |
| `network_infer.onnx` | Intermediate inference graph (source of initial weights) |
| `network_train.onnx` | ORT training graph, pre-Deeploy optimization |
| `network_train_optim.onnx` | Training graph after Deeploy optimization passes |
| `eval_model.onnx` | ORT eval graph (forward + loss, no backward) |
| `optimizer_model.onnx` | ORT SGD optimizer graph |
| `checkpoint/` | Initial parameter checkpoint directory |
| `inputs.npz` | Per-mini-batch input data for on-device test |
| `outputs.npz` | Reference outputs (updated weights + per-batch losses) |

---

## CLI Reference

Run from `/app/Onnx4Deeploy/`:

```bash
PYTHONPATH=/app/Deeploy:/app/Onnx4Deeploy \
python Onnx4Deeploy.py --model <NAME> --mode train [OPTIONS]
```

### Required

| Flag | Description |
|------|-------------|
| `--model NAME` | Model to export (e.g. `SimpleMLP`, `SleepConViT`, `CCT`) |
| `--mode train` | Enable training export pipeline |

### Training Data Options

| Flag | Default | Description |
|------|---------|-------------|
| `--n-batches N` | `4` | Total mini-batches to prepare in `inputs.npz`. Each mini-batch is one complete forward+backward pass on the device. |
| `--n-accum N` | `1` | Mini-batches per SGD update (gradient accumulation window). Must divide `--n-batches` evenly. |
| `--batch-size N` | `1` | Samples per mini-batch (the first dimension of the input tensor). |
| `--dataset {random,mnist}` | `random` | Data source for input/label generation. `random` uses Gaussian noise; `mnist` uses real MNIST images. |
| `--data-path PATH` | `/tmp/mnist` | Root directory for dataset files. Used with `--dataset mnist`; MNIST binaries are downloaded here if absent. |

### Output

| Flag | Default | Description |
|------|---------|-------------|
| `-o PATH` | `./onnx/model/<name>_train<cfg>/` | Override the output directory. |

### Derived quantities

| Quantity | Formula | Example (`--n-batches 6 --n-accum 3`) |
|----------|---------|---------------------------------------|
| `n_steps` | `n_batches // n_accum` | 2 SGD updates |
| effective batch | `batch_size × n_accum` | 3× per-sample batch size |

---

## Configuration Keys

These keys live in the dictionary returned by `load_config()`.  For
`SimpleMlpExporter` they can be overridden via `_config_overrides` (populated
automatically from CLI flags).

### Model architecture

| Key | Type | Default (SimpleMLP) | Description |
|-----|------|---------------------|-------------|
| `batch_size` | `int` | `1` | Samples per mini-batch (first input dimension) |
| `input_height` | `int` | `8` (`28` with MNIST) | Spatial height of input, or first spatial dim |
| `input_width` | `int` | `8` (`28` with MNIST) | Spatial width of input, or second spatial dim |
| `hidden_size` | `int` | `8` | Hidden layer size |
| `num_classes` | `int` | `10` | Number of output classes |
| `opset_version` | `int` | `17` | ONNX opset. Must be ≥ 13 for ORT training. |
| `dropout` | `float` | `0.0` | Dropout probability (0 disables dropout) |

### Training loop

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_batches` | `int` | `4` | Total mini-batches (see [Gradient Accumulation](#gradient-accumulation)) |
| `n_accum` | `int` | `1` | Accumulation steps per SGD update |
| `learning_rate` | `float` | `0.001` | SGD learning rate for reference simulation |

### Training strategy

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `training_strategy` | `str` | `"no_bias"` | Which parameters to train (see [Training Strategy](#training-strategy)) |
| `custom_trainable_params` | `List[str]` | `[]` | Explicit list of param names when `training_strategy="custom"` |

### Data source

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `dataset` | `str` | `"random"` | `"random"` or `"mnist"` |
| `data_path` | `str\|None` | `None` | Filesystem path for dataset files; `None` → `/tmp/mnist` |
| `data_split` | `str` | `"train"` | `"train"` or `"test"` (MNIST split to draw from) |

---

## Training Strategy

The `training_strategy` key controls which parameters are passed to
`generate_artifacts` as `requires_grad` (trainable) vs `frozen_params`.

| Strategy | Trainable parameters | Use case |
|----------|----------------------|----------|
| `"full"` | All model parameters (weights + biases) | Full fine-tuning |
| `"no_bias"` | All weights, **no** biases *(default)* | Standard embedded training; biases are small and often frozen to reduce memory |
| `"last_layer"` | Only the final classification layer (`fc2.weight`, `fc2.bias`) | Transfer learning / head-only fine-tuning |
| `"custom"` | Parameters listed in `custom_trainable_params` | Arbitrary frozen/trainable split |

### Example — custom strategy

```python
exporter._config_overrides = {
    "training_strategy": "custom",
    "custom_trainable_params": ["fc1.weight", "fc2.weight"],
}
```

> **Note for other models:** `get_trainable_params()` is a method on
> `BaseONNXExporter` with a default that returns all parameters.  Each model
> exporter overrides it with its own strategy logic.

---

## Gradient Accumulation

Gradient accumulation lets you simulate a larger effective batch size without
loading all samples into memory simultaneously.

### Terminology

| Term | Meaning |
|------|---------|
| `n_batches` | Total forward+backward passes; equals the number of mini-batch entries in `inputs.npz` |
| `n_accum` | Mini-batches per SGD update (accumulation window) |
| `n_steps` | SGD weight updates = `n_batches / n_accum` |

**Constraint:** `n_batches % n_accum == 0`.

### What happens per mini-batch

```
For each SGD update step (n_steps total):
  └─ For each accumulation step (n_accum mini-batches):
       ├─ Set lazy_reset_grad = True  on the FIRST mini-batch of the step
       │   (zeros the on-device accumulation buffer)
       ├─ Set lazy_reset_grad = False on subsequent mini-batches
       ├─ Run forward + backward
       └─ Accumulate gradient internally (InPlaceAccumulatorV2)
  └─ Apply SGD: weight -= lr × accumulated_gradient
```

### Reference simulation (Python side)

`create_training_test_data()` mirrors the above exactly:
- Calls the ORT session `n_batches` times with distinct (input, label) pairs.
- Manually accumulates the per-step gradient tensors (exposed as extra ORT
  outputs by patching the in-memory `network_train.onnx`).
- Applies SGD once every `n_accum` steps using the accumulated gradient.
- Records the loss scalar from every forward pass → `outputs.npz["loss"]`.

All losses within one accumulation window are computed with the **same
(pre-update) weights**, matching on-device behaviour.

---

## Data Source

The data source controls how `(input, label)` pairs are generated for
`inputs.npz`.  It is selected via the `dataset` config key or `--dataset` CLI
flag.

### `"random"` (default)

Generates Gaussian random inputs (`np.random.randn`) and uniform-random integer
labels.  Seed is fixed to `42` for reproducibility.

Behaviour is identical to the original implementation — no external files
required.

### `"mnist"`

Loads real MNIST images from the IDX binary format.

- Files are downloaded automatically to `data_path` (default `/tmp/mnist`) if
  not already present (source: `ossci-datasets.s3.amazonaws.com`).
- Images are normalised to `[0, 1]` and flattened to match the model's
  `input_shape`.  If `input_size ≠ 784`, linear resampling (`np.interp`) is
  applied — this is adequate for functional testing.
- Labels are the original MNIST class indices (0–9).

**When `--dataset mnist` is used with `SimpleMLP`, `load_config()` automatically
adjusts:**

| Key | Auto-set value | Can be overridden? |
|-----|----------------|--------------------|
| `input_height` | `28` | Yes — pass `--input-height N` |
| `input_width` | `28` | Yes — pass `--input-width N` |
| `num_classes` | `10` | Yes — pass `--num-classes N` |

This means the generated model will have input size 28×28 = 784 by default.
Ensure `--hidden-size` is set appropriately (e.g. `128` or `256`) for a
non-trivial network.

---

## Test Data Layout

### `inputs.npz`

Gradient accumulation buffer entries are **omitted** — the C harness
zero-initialises them after `InitTrainingNetwork()`.

```
arr_0000          float32(batch, input_size)  ← input data,       mini-batch 0
arr_0001          int64(batch,)               ← labels,           mini-batch 0
arr_0002          float32(H, I)               ← fc1_weight        (initial)
arr_0003          float32(C, H)               ← fc2_weight        (initial)
arr_0004          uint8(1,)                   ← lazy_reset_grad   (always True for mb 0)

mb1_arr_0000      float32(batch, input_size)  ← input data,       mini-batch 1
mb1_arr_0001      int64(batch,)               ← labels,           mini-batch 1
...
mbN_arr_0000                                  ← input data,       mini-batch N
mbN_arr_0001                                  ← labels,           mini-batch N
```

- **Base entries** (`arr_*`): positional over all non-grad-buffer graph inputs,
  in the order they appear in `network.onnx`.
- **Per-mini-batch DATA entries** (`mb{I}_arr_{J}`): only the first
  `num_data_inputs` (= 2) base entries are replicated per extra mini-batch,
  because weights and control signals are shared across all mini-batches of the
  same SGD step.

### `outputs.npz`

```
fc1_weight        float32(H, I)               ← weight after all SGD updates
fc2_weight        float32(C, H)               ← weight after all SGD updates
loss              float32(n_batches,)          ← per-mini-batch reference loss
```

The on-device runner compares each printed loss value against the reference
`loss` array to verify correctness.

---

## Extending: Custom Exporters

To add a new model that supports training export, subclass `BaseONNXExporter`
and implement the three required methods:

```python
from onnx4deeploy.core.base_exporter import BaseONNXExporter

class MyModelExporter(BaseONNXExporter):

    def load_config(self):
        config = {
            "batch_size": 1,
            "num_classes": 10,
            "opset_version": 17,
            "training_strategy": "no_bias",
            "n_accum": 1,
            "dataset": "random",
            "data_path": None,
            "data_split": "train",
            # ... model-specific keys ...
        }
        if hasattr(self, "_config_overrides") and self._config_overrides:
            config.update(self._config_overrides)
        return config

    def create_model(self):
        return MyPyTorchModel(...)

    def get_input_shape(self):
        return (self.config["batch_size"], ...)
```

### Optional overrides

| Method | Default behaviour | When to override |
|--------|-------------------|------------------|
| `get_trainable_params(all_names)` | Returns all param names | Custom frozen/trainable split |
| `get_data_source()` | Returns `RandomDataSource` | Use real dataset |
| `get_inference_pipeline()` | Standard inference passes | Model needs extra ONNX transforms |
| `get_training_pipeline()` | Standard training passes | Model needs custom Deeploy transforms |
| `create_training_test_data()` | Generic ORT simulation | Model has non-standard graph structure |
| `_get_config_string()` | Returns `""` | Include config dims in output folder name |

The base class `create_training_test_data()` works for models whose training
graph structure matches the standard ORT pattern (gradient outputs named
`<param>_grad`).  For models with custom loss or gradient structures (like
SimpleMLP with `InPlaceAccumulatorV2`), override the method and call
`get_data_source()` to keep data loading decoupled.

---

## Extending: Custom Data Sources

Implement the `DataSource` interface to plug in any dataset:

```python
from onnx4deeploy.data.base_datasource import DataSource
import numpy as np
from typing import List, Tuple

class MyDataSource(DataSource):
    def load_batches(
        self,
        n_batches: int,
        input_shape: Tuple[int, ...],
        num_classes: int,
        seed: int = 42,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Returns:
            inputs_list: n_batches float32 arrays, each shape == input_shape
            labels_list: n_batches int64 arrays,   each shape == (batch_size,)
        """
        ...
```

Then override `get_data_source()` in your exporter:

```python
def get_data_source(self):
    if self.config.get("dataset") == "my_dataset":
        return MyDataSource(...)
    return super().get_data_source()  # falls back to RandomDataSource
```

The DataSource is called once per `create_training_test_data()` invocation.
The returned lists are independent of the gradient-accumulation logic — the
exporter is responsible for feeding the correct mini-batch index into the
training loop.

### Built-in data sources

| Class | Module | Description |
|-------|--------|-------------|
| `RandomDataSource` | `onnx4deeploy.data.random_datasource` | Gaussian inputs, random labels (default) |
| `MNISTDataSource` | `onnx4deeploy.data.mnist_datasource` | Real MNIST images, auto-downloaded |

---

## Complete Examples

### Minimal test — 1 mini-batch, random data

```bash
cd /app/Onnx4Deeploy
PYTHONPATH=/app/Deeploy:/app/Onnx4Deeploy \
python Onnx4Deeploy.py --model SimpleMLP --mode train --n-batches 1
```

### Standard test — 4 mini-batches, 4 SGD steps

```bash
python Onnx4Deeploy.py --model SimpleMLP --mode train --n-batches 4
```

### Gradient accumulation — 6 mini-batches, 2 SGD steps of 3

```bash
python Onnx4Deeploy.py --model SimpleMLP --mode train \
    --n-batches 6 --n-accum 3
```

### MNIST data — 28×28 input, real images

```bash
python Onnx4Deeploy.py --model SimpleMLP --mode train \
    --dataset mnist --n-batches 8
```

Input size auto-adjusts to 784 (28×28).  For a more capable network:

```bash
python Onnx4Deeploy.py --model SimpleMLP --mode train \
    --dataset mnist --n-batches 8 \
    --input-height 28 --input-width 28 --hidden-size 128
```

### MNIST with local data directory (avoid repeated downloads)

```bash
python Onnx4Deeploy.py --model SimpleMLP --mode train \
    --dataset mnist --data-path /data/mnist --n-batches 4
```

### MNIST with gradient accumulation

```bash
python Onnx4Deeploy.py --model SimpleMLP --mode train \
    --dataset mnist --n-batches 6 --n-accum 3
```

### Custom output directory

```bash
python Onnx4Deeploy.py --model SimpleMLP --mode train \
    --n-batches 4 -o /tmp/my_mlp_train
```

### Python API

```python
import sys
sys.path.insert(0, "/app/Onnx4Deeploy")
import os
os.environ["PYTHONPATH"] = "/app/Deeploy:/app/Onnx4Deeploy"

from onnx4deeploy.models.simple_mlp_exporter import SimpleMlpExporter

exporter = SimpleMlpExporter(save_path="/tmp/my_mlp_train")
exporter._config_overrides = {
    "n_batches": 6,
    "n_accum": 3,
    "dataset": "mnist",
    "training_strategy": "full",
}
exporter.export_training()
```

---

## Integration with Deeploy Test Runner

After generating the training graph, run the on-device test:

```bash
cd /app/Deeploy/DeeployTest
PYTHONPATH=/app/Deeploy:/app/Onnx4Deeploy \
python deeployTrainingRunner_siracusa.py \
    -t /app/Onnx4Deeploy/onnx/model/simplemlp_train_28x28_128_10 \
    --n-accum 3
```

**Always pass `--n-accum` to the runner when accumulation was used.**  The
runner regenerates `testinputs.h` (which embeds `N_ACCUM_STEPS` as a
compile-time constant) and recompiles before simulation.

See `TRAINING_GUIDE.md` in `/app/AI_AGENT/` for the full two-step workflow.
