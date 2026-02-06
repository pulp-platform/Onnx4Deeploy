# Changelog

All notable changes to Onnx4Deeploy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Pre-commit hooks configuration for code quality automation
- Comprehensive contributing guide (CONTRIBUTING.md)
- Security policy (SECURITY.md)
- Code of conduct (CODE_OF_CONDUCT.md)

## [0.2.0] - 2026

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

## [0.1.0] - 2025

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

[Unreleased]: https://github.com/runwangdl/Onnx4Deeploy/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/runwangdl/Onnx4Deeploy/releases/tag/v0.2.0
[0.1.0]: https://github.com/runwangdl/Onnx4Deeploy/releases/tag/v0.1.0
