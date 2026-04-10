# Onnx4Deeploy

[![CI](https://github.com/pulp-platform/Onnx4Deeploy/workflows/CI/badge.svg)](https://github.com/pulp-platform/Onnx4Deeploy/actions)
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](.github/CONTRIBUTING.md)

**A comprehensive framework for ONNX model generation, optimization, and deployment for Deeploy.**

Onnx4Deeploy provides a unified interface for exporting PyTorch models to ONNX format with specialized optimizations for inference **and on-device training** on Deeploy hardware accelerators.

---

## ✨ Features

### 🎯 Core capabilities
- **Unified model export** — single API for both inference and training-mode ONNX graph generation
- **15 model exporters** — MLperf Tiny, ViT-family, Mamba and simple reference models (see [Supported models](#-supported-models))
- **38 operator test generators** — every Deeploy-supported ONNX op has its own reference test
- **Training graph optimization** — custom passes (`fuse_mse_loss`, `fuse_global_average_pool_grad`, GEMM conversion, gradient-node cleanup, shape simplification, …) specialized for on-device training
- **Type-safe API** — full type annotations and docstrings

### 🧪 Testing framework
- Pytest-based test suite (inference + training mode, per model)
- ONNX Runtime validation of every exported graph
- `inputs.npz` / `outputs.npz` layout checks to keep the training harness in sync

---

## 📦 Installation

### ⚠️ Hard requirements

This project has **strict** version pins in `requirements.txt` because the training-graph generation path is very sensitive to `torch`/`onnxruntime-training` interactions. Deviating from these versions is **not supported**.

| Dependency | Version | Notes |
|---|---|---|
| **Python** | **3.10 only** (`==3.10.*`) | Enforced via `pyproject.toml` |
| `torch` | `2.7.0` | Pinned exactly |
| `onnx` | `1.16.0` | Pinned exactly |
| `onnxruntime-training` | **`1.19.2`** | Pinned exactly — **not** the regular `onnxruntime` package |
| `onnx-graphsurgeon` | `0.5.8` | Pinned exactly |
| `onnxscript` | `0.5.7` | Pinned exactly |
| `onnxsim` | `0.4.36` | Pinned exactly |
| `numpy` | `1.26.4` | Pinned exactly |
| `pyyaml` | `6.0.2` | Pinned exactly |

### Platform note (onnxruntime-training)

`onnxruntime-training` is **only published as a prebuilt wheel for x86_64 Linux** (amd64). There is no official aarch64 / Apple Silicon / Windows wheel. On Apple Silicon Macs you need to run the installation and all generation scripts inside a `linux/amd64` Docker container (Docker Desktop supports this via QEMU / Rosetta). On other platforms you must build `onnxruntime-training` from source.

### Install from source

```bash
git clone https://github.com/pulp-platform/Onnx4Deeploy.git
cd Onnx4Deeploy

# (Recommended) create a clean Python 3.10 environment
conda create -n onnx4deeploy python=3.10 -y
conda activate onnx4deeploy

# Install pinned runtime dependencies
pip install -r requirements.txt

# Install the package itself
pip install -e .
```

### Verify installation

```bash
python -c "import onnx4deeploy; print(onnx4deeploy.__version__)"
python -c "from onnxruntime.training import artifacts; print('ORT training OK')"
```

Both commands must succeed. If the second one raises `ModuleNotFoundError: No module named 'onnxruntime.training'`, you have the regular `onnxruntime` package installed instead of `onnxruntime-training` — uninstall it and reinstall from `requirements.txt`.

---

## 🚀 Quick start

Onnx4Deeploy provides two main features: **operator-level** generation and **model-level** export.

### 🎯 Command-line tool

Use the unified CLI tool `Onnx4Deeploy.py`:

```bash
# Generate an operator test
python Onnx4Deeploy.py -operator Relu -o ./onnx

# Generate a model inference graph
python Onnx4Deeploy.py -model CCT -mode infer -o ./onnx

# Generate a model training graph
python Onnx4Deeploy.py -model CCT -mode train -o ./onnx

# List available options
python Onnx4Deeploy.py --list-models
python Onnx4Deeploy.py --list-operators
python Onnx4Deeploy.py --examples
```

**Available arguments:**
- `-operator NAME` — generate an operator test (e.g., `Relu`, `Add`, `Gemm`, `ConvGradXW`)
- `-model NAME` — generate a model ONNX (see [Supported models](#-supported-models))
- `-mode {infer,train}` — model export mode (default: `infer`)
- `-o PATH` — output directory path
- `--n-epochs`, `--n-steps`, `--n-batches`, `--n-accum`, `--batch-size`, `--dataset`, `--data-path`, `--data-size`, `--lr`, `--classes` — training-mode knobs
- `--list-models`, `--list-operators`, `--examples` — help / discovery

---

## 📚 Supported models

| Category | Model | Inference | Training | Notes |
|---|---|---|---|---|
| **MLperf Tiny** | ResNet-8 | ✅ | ✅ | Image classification (CIFAR-10) |
| **MLperf Tiny** | MobileNetV2-0.35 | ✅ | ✅ | Visual Wake Words |
| **MLperf Tiny** | DS-CNN-XS / DS-CNN-S | ✅ | ✅ | Keyword spotting |
| **MLperf Tiny** | Autoencoder-tiny / -MLperf | ✅ | ✅ | Anomaly detection |
| **BMI / EEG** | EpiDeNet | ✅ | ✅ | Epilepsy detection |
| **BMI / EEG** | MIBMInet | ✅ | ✅ | Motor-imagery BMI |
| **Sleep staging** | SleepConViT | ✅ | ✅ | Sleep stage classification |
| **Transformer** | CCT | ✅ | ✅ | Compact Convolutional Transformer |
| **Transformer** | TinyTransformer | ✅ | ✅ | Patch-based Transformer (MNIST) |
| **Transformer** | TinyViT (5M/11M/21M) | ✅ | ✅ | Compact ViT variants |
| **Transformer** | MobileViT (XXS/XS/S) | ✅ | ✅ | Mobile-friendly hybrid ViT |
| **SSM** | Mamba | ✅ | ❌ | Selective SSM; training export not yet supported |
| **Reference / demo** | SimpleMLP | ✅ | ✅ | Minimal MLP |
| **Reference / demo** | SimpleCNN | ✅ | ✅ | Minimal strided-conv CNN |
| **Reference / demo** | LightweightCNN | ✅ | ✅ | Compact image classifier |

`--list-models` is the authoritative source; this table is for orientation.

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](.github/CONTRIBUTING.md) for details.

### Development setup

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run the full test suite (skip slow MNIST training test)
pytest tests/ -m "not slow"

# Format code
black --line-length=100 .
isort --profile=black --line-length=100 .
```

---

## 📄 License

All licenses used in this repository are listed under the `LICENSES` folder. Unless specified otherwise in the respective file headers, all code checked into this repository is made available under a permissive license.

- Most software sources and tool scripts are licensed under the [MIT license](https://opensource.org/licenses/mit).
- Markdown, JSON, text files, pictures, PDFs, are licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0) license (CC BY 4.0).

To extract license information for all files, you can use the [reuse tool](https://reuse.software/) and by running `reuse spdx` in the root directory of this repository.

---

## 🙏 Acknowledgments

- Built with [ONNX](https://onnx.ai/)
- Tested with [ONNX Runtime Training](https://onnxruntime.ai/docs/get-started/training-on-device.html)
- Optimized for [Deeploy](https://github.com/pulp-platform/Deeploy)

---

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/pulp-platform/Onnx4Deeploy/issues)
- **Documentation**: [docs/](docs/)
