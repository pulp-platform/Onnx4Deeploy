# Onnx4Deeploy

[![CI](https://github.com/runwangdl/Onnx4Deeploy/workflows/CI/badge.svg)](https://github.com/runwangdl/Onnx4Deeploy/actions)
[![Tests](https://img.shields.io/badge/tests-91%20passed-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](.github/CONTRIBUTING.md)

**A comprehensive framework for ONNX model generation, optimization, and deployment for Deeploy.**

Onnx4Deeploy provides a unified interface for exporting PyTorch models to ONNX format with specialized optimizations for training and inference on Deeploy hardware accelerators.

---

## ✨ Features

### 🎯 Core Capabilities
- **Unified Model Export**: Single API for both inference and training mode ONNX export
- **27 Operator Tests**: Comprehensive test coverage for all supported ONNX operators
- **3 Pre-built Models**: CCT, EpiDeNet, and MI-BMInet ready to use
- **Training Graph Optimization**: Specialized optimizations for on-device training
- **Type-safe API**: Full type annotations and documentation

### 🔧 Optimization Suite
- GEMM conversion and fusion
- Gradient node optimization
- Graph cleaning and simplification
- Shape operation optimization
- Node naming and annotation utilities

### 🧪 Testing Framework
- 91 comprehensive tests (100% passing)
- Pytest-based test suite
- ONNX Runtime validation
- Baseline comparison testing

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.0+
- ONNX 1.14+
- ONNX Runtime 1.19+

### Install from source

```bash
git clone https://github.com/runwangdl/Onnx4Deeploy.git
cd Onnx4Deeploy
pip install -e .
```

### Verify installation

```bash
python -c "import onnx4deeploy; print(onnx4deeploy.__version__)"
# Output: 0.2.0
```

---

## 🚀 Quick Start

Onnx4Deeploy provides two main features: **Operator-level** generation and **Model-level** export.

### 🎯 Command Line Tool (Recommended)

Use the unified CLI tool `Onnx4Deeploy.py`:

```bash
# Generate operator tests
python Onnx4Deeploy.py -operator Relu -o ./onnx

# Generate model inference graph
python Onnx4Deeploy.py -model CCT -mode infer -o ./onnx

# Generate model training graph
python Onnx4Deeploy.py -model CCT -mode train -o ./onnx

# List available options
python Onnx4Deeploy.py --list-models
python Onnx4Deeploy.py --list-operators
python Onnx4Deeploy.py --examples
```

**Available Arguments:**
- `-operator NAME`: Generate operator test (e.g., Relu, Add, Gemm)
- `-model NAME`: Generate model ONNX (e.g., CCT, EpiDeNet, MIBMInet)
- `-mode {infer,train}`: Model export mode (default: infer)
- `-o PATH`: Output directory path
- `--list-models`: List all available models
- `--list-operators`: List all available operators
- `--examples`: Show usage examples

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](.github/CONTRIBUTING.md) for details.

### Quick Development Setup

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black .
isort .
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Built with [ONNX](https://onnx.ai/)
- Tested with [ONNX Runtime](https://onnxruntime.ai/)
- Optimized for [Deeploy](https://deeploy.ml/) hardware

---

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/runwangdl/Onnx4Deeploy/issues)
- **Documentation**: [docs/](docs/)
- **Progress**: [REFACTORING_STATUS.md](REFACTORING_STATUS.md)

---

**Status**: 🚧 v0.2.0 refactoring in progress | ✅ 91/91 tests passing | 🎯 Production-ready core
