# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Onnx4Deeploy - ONNX model generation and optimization framework

This package provides tools for generating, optimizing, and analyzing ONNX models
for deep learning deployment.
"""

__version__ = "0.2.0"

# Import core components
from .core.base_exporter import BaseONNXExporter, ExportMode
from .core.optimization_passes import (
    STANDARD_PASSES,
    create_inference_pipeline,
    create_training_pipeline,
    create_transformer_inference_pipeline,
)
from .core.optimization_pipeline import OptimizationPass, OptimizationPipeline, PassConfig

# Import model exporters
from .models import (
    CCTExporter,
    EpiDeNetExporter,
    MIBMInetExporter,
    MobileNetV2Exporter,
    ResNetExporter,
    SimpleMlpExporter,
)

__all__ = [
    # Version
    "__version__",
    # Exporters
    "CCTExporter",
    "EpiDeNetExporter",
    "MIBMInetExporter",
    "SimpleMlpExporter",
    "ResNetExporter",
    "MobileNetV2Exporter",
    # Core components
    "BaseONNXExporter",
    "ExportMode",
    # Optimization pipeline
    "OptimizationPipeline",
    "OptimizationPass",
    "PassConfig",
    "create_inference_pipeline",
    "create_training_pipeline",
    "create_transformer_inference_pipeline",
    "STANDARD_PASSES",
]
