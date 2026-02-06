# Contributing to Onnx4Deeploy

Thank you for your interest in contributing to Onnx4Deeploy! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Onnx4Deeploy.git
   cd Onnx4Deeploy
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/runwangdl/Onnx4Deeploy.git
   ```

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- **Description**: Clear description of the bug
- **Environment**: Python version, OS, package versions
- **Reproduction**: Minimal code to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Error messages**: Full error messages and stack traces

### Feature Requests

When requesting features, please include:

- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives**: Have you considered alternatives?
- **Additional context**: Any other relevant information

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed

## Adding New Operators

To add a new ONNX operator:

1. Create operator class in `onnx4deeploy/operators/`
2. Inherit from `BaseOperator`
3. Implement required methods:
   - `create_model()`
   - `generate_inputs()`
   - `expected_output()`
4. Add tests in `tests/operators/`
5. Update documentation

Example:
```python
from onnx4deeploy.operators.base_operator import BaseOperator

class NewOperator(BaseOperator):
    def create_model(self):
        # Implement operator logic
        pass
```

## Adding New Model Exporters

To add a new model exporter:

1. Create exporter in `onnx4deeploy/models/`
2. Inherit from `BaseExporter`
3. Implement model export logic
4. Add tests in `tests/models/`
5. Register in CLI tool

## Questions?

- Open a [GitHub Issue](https://github.com/runwangdl/Onnx4Deeploy/issues)
- Check existing documentation in `docs/`
- Review existing code for examples

## Thank You!

Your contributions make Onnx4Deeploy better for everyone. We appreciate your time and effort!
