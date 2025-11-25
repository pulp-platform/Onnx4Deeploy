# Onnx4Deeploy
A lightweight tool for deploying ONNX models on deeploy.

## Overview
### Test Directory Structure
The test directory in this project contains comprehensive test cases, primarily including:

- CCT model training graph optimization pipeline -> Onnx4Deeploy/Tests/Models/CCT/testtraingenerate.py
- CCT End-to-end inference -> Onnx4Deeploy/Tests/Models/CCT/testinfergenerate.py
- Various operator unit tests -> Onnx4Deeploy/Tests/Operators
- Useful tools for Deeploy
    - Local Linting Fix: Onnx4Deeploy/utils/localLinting.sh
    - E2E layerwise latency visualization: Onnx4Deeploy/utils/Visualization
