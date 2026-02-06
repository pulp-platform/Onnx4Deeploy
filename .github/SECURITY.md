# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Currently supported versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

We take the security of Onnx4Deeploy seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via:

1. **Email**: Send details to the project maintainers (check repository for contact information)
2. **GitHub Security Advisory**: Use the "Security" tab in the GitHub repository to privately report vulnerabilities

### What to Include

Please include the following information in your report:

- Type of vulnerability
- Full paths of affected source files
- Location of the affected code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the vulnerability, including how an attacker might exploit it

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: Within 7 days
  - High: Within 30 days
  - Medium: Within 90 days
  - Low: Next scheduled release

### Process

1. **Receipt**: We will acknowledge receipt of your vulnerability report
2. **Assessment**: We will confirm the vulnerability and determine its severity
3. **Fix**: We will work on a fix and prepare a security advisory
4. **Release**: We will release a patched version and publish the security advisory
5. **Credit**: We will credit you in the advisory (unless you prefer to remain anonymous)

## Security Best Practices

When using Onnx4Deeploy:

### General Guidelines

- Always use the latest stable version
- Keep dependencies up to date
- Use virtual environments to isolate dependencies
- Review ONNX models from untrusted sources before loading

### Model Loading

```python
# Safe: Load models you created or from trusted sources
model = onnx.load("path/to/trusted/model.onnx")

# Caution: Validate models from untrusted sources
# ONNX models can contain arbitrary Python code in custom operators
# Always inspect and validate before loading
```

### Dependency Security

We use:
- `pre-commit` hooks with security checks
- Automated dependency scanning in CI/CD
- Regular dependency updates

### Known Security Considerations

1. **ONNX Model Loading**: ONNX models can contain custom operators with arbitrary code
   - Only load models from trusted sources
   - Consider using `onnx.checker.check_model()` before loading

2. **PyTorch Models**: When exporting PyTorch models, ensure model source is trusted
   - Custom PyTorch operators can execute arbitrary code during export

3. **Dependencies**: Keep all dependencies updated
   - PyTorch, ONNX, and ONNX Runtime may have their own security advisories

## Security Updates

Security updates will be released as:
- Patch versions (0.2.x) for the current major version
- Security advisories published on GitHub
- Announcements in release notes and CHANGELOG

## Acknowledgments

We appreciate the security research community's efforts in responsibly disclosing vulnerabilities. Contributors who report valid security issues will be acknowledged in:
- Security advisory
- CHANGELOG
- Project documentation (if desired)

Thank you for helping keep Onnx4Deeploy and its users safe!
