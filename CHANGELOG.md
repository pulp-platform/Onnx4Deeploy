# Changelog

All notable changes to Onnx4Deeploy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **MobileViT** model support (XXS, XS, S variants) - hybrid CNN-Transformer for mobile/edge
- **TinyViT** model support - efficient hierarchical vision transformer
- **CCT LoRA fine-tuning** end-to-end Deeploy support — `--use-lora` produces a
  training graph that passes `deeployTrainingRunner_tiled_siracusa.py` (front-end
  + tiling + GVSoC) at fp32

### Changed
- Optimized MobileViT ONNX export: replaced dynamic `view(-1,...)` with static `reshape(batch_size,...)`
- Fixed dimension propagation in transformer blocks to eliminate Shape/Gather nodes

### Fixed
- `convert_sum_to_add` (the copy in `trainOptimization.py` actually used by the
  training pipeline) now stamps `value_info` on every `_intermediate_{j}` tensor
  produced when an N-input Sum is split into chained Adds. Without this,
  Deeploy's front-end shape assertion fires whenever a gradient feed-point has
  more than 3 contributors (e.g. LoRA adapters add 3 extra branches per Q/K/V).
- New pass `duplicate_constant_fed_transposes` (`graph_cleaner.py`, wired in
  `train_optimizer.py`) — duplicates any Constant-fed Transpose/Reshape with
  multiple consumers so each folded Constant has a single user, preventing
  Deeploy's `hoistConstant` `len(constant.outputs) <= 1` assertion that fires
  in LoRA when frozen weights are shared by forward MatMul and backward Gemm.

## [0.2.1] - 06.02.2026
### Changed
- Transferred repo to `pulp-platform` organization
- Use `pre-commit` for linting and code formatting in CI


### Added
- Pre-commit hooks configuration for code quality automation
- Comprehensive contributing guide (CONTRIBUTING.md)
- Security policy (SECURITY.md)
- Code of conduct (CODE_OF_CONDUCT.md)
- Use `reuse` for license management and compliance
- Added proper license headers to all source files
-
### Changed
-

### Fixed
-

## [0.2.0] - 06.02.2026

### Added
- Unified CLI tool (`Onnx4Deeploy.py`) for operator and model generation
- Support for 27+ ONNX operators with comprehensive testing
- Three pre-built model exporters: CCT, EpiDeNet, MI-BMInet
- Mamba model support with clean export
- Training mode ONNX export capabilities
- Optimization pipeline framework
- Type-safe API with full type annotations
- Modern Python packaging with pyproject.toml
- GitHub Actions CI/CD pipeline with matrix testing (Python 3.8-3.11)
- 91 comprehensive tests with 100% pass rate
- Documentation for operator testing and optimization pipeline

### Changed
- Major refactoring of codebase architecture
- Improved code organization with clear separation of concerns
- Enhanced test framework with pytest markers
- Updated dependencies to latest stable versions

### Fixed
- Various bug fixes in operator implementations
- Improved ONNX graph optimization passes
- Enhanced gradient node handling

## [0.1.0] - 25.10.2025

### Added
- Initial release
- Basic ONNX model generation framework
- Core operator implementations
- PyTorch to ONNX export utilities

---

## Version History Legend

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security vulnerability fixes

[Unreleased]: https://github.com/pulp-platform/Onnx4Deeploy/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/pulp-platform/Onnx4Deeploy/releases/tag/v0.2.1
[0.2.0]: https://github.com/pulp-platform/Onnx4Deeploy/releases/tag/v0.2.0
[0.1.0]: https://github.com/pulp-platform/Onnx4Deeploy/releases/tag/v0.1.0
