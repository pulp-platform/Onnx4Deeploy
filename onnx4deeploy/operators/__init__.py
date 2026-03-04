# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Onnx4Deeploy Operators Module.

This module contains ONNX operator test generators and testing framework.
"""

from .add import AddOperatorTest
from .averagepool import AveragePoolOperatorTest
from .averagepool_grad import AveragePoolGradOperatorTest
from .base_operator import BaseOperatorTest, SimpleElementwiseOperator
from .concat import ConcatOperatorTest
from .conv2d import Conv2DOperatorTest
from .conv_grad_b import ConvGradBOperatorTest
from .conv_grad_w import ConvGradWOperatorTest
from .conv_grad_x import ConvGradXOperatorTest
from .gemm import GemmOperatorTest
from .groupnorm import GroupNormOperatorTest
from .groupnorm_grad_b import GroupNormGradBOperatorTest
from .groupnorm_grad_w import GroupNormGradWOperatorTest
from .groupnorm_grad_w_h2 import GroupNormGradWH2OperatorTest
from .groupnorm_grad_x import GroupNormGradXOperatorTest
from .groupnorm_h2 import GroupNormH2OperatorTest
from .inplaceaccumulatorv2 import InPlaceAccumulatorV2OperatorTest
from .layernorm import LayerNormOperatorTest
from .layernorm_grad import LayerNormGradOperatorTest
from .matmul import MatMulOperatorTest
from .maxpool import MaxPoolOperatorTest
from .reducesum import ReduceSumOperatorTest
from .relu import ReluOperatorTest
from .relu_exporter import ReluExporter
from .relu_grad import ReluGradOperatorTest
from .sgd import SGDOperatorTest
from .softmax_cross_entropy import SoftmaxCrossEntropyOperatorTest
from .softmax_cross_entropy_dual_output import SoftmaxCrossEntropyDualOutputOperatorTest
from .softmax_cross_entropy_grad import SoftmaxCrossEntropyGradOperatorTest
from .softmax_grad import SoftmaxGradOperatorTest
from .split import SplitOperatorTest
from .sub import SubOperatorTest
from .transpose import TransposeOperatorTest

__all__ = [
    "BaseOperatorTest",
    "SimpleElementwiseOperator",
    "ReluExporter",
    "AddOperatorTest",
    "ReluOperatorTest",
    "GemmOperatorTest",
    "MatMulOperatorTest",
    "ReduceSumOperatorTest",
    "ConcatOperatorTest",
    "SplitOperatorTest",
    "SubOperatorTest",
    "TransposeOperatorTest",
    "MaxPoolOperatorTest",
    "AveragePoolOperatorTest",
    "AveragePoolGradOperatorTest",
    "ReluGradOperatorTest",
    "SGDOperatorTest",
    "InPlaceAccumulatorV2OperatorTest",
    "SoftmaxCrossEntropyOperatorTest",
    "SoftmaxCrossEntropyDualOutputOperatorTest",
    "SoftmaxCrossEntropyGradOperatorTest",
    "SoftmaxGradOperatorTest",
    "LayerNormOperatorTest",
    "LayerNormGradOperatorTest",
    "GroupNormOperatorTest",
    "GroupNormH2OperatorTest",
    "GroupNormGradBOperatorTest",
    "GroupNormGradWOperatorTest",
    "GroupNormGradXOperatorTest",
    "GroupNormGradWH2OperatorTest",
    "Conv2DOperatorTest",
    "ConvGradXOperatorTest",
    "ConvGradWOperatorTest",
    "ConvGradBOperatorTest",
]
