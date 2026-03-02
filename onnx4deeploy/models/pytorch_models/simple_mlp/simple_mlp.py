# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Simple Multi-Layer Perceptron (MLP) for demonstration - Deploy Version."""

import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    Simple two-layer MLP for image classification (Deploy Version).

    Architecture:
    - Input:   (batch_size, 1, height, width)
    - Flatten: (batch_size, input_size)
    - FC1 (Gemm): input_size  -> hidden_size
    - FC2 (Gemm): hidden_size -> num_classes
    - Output:  logits (batch_size, num_classes)

    Args:
        input_size: Size of flattened input (e.g., 16 for 4x4 images)
        hidden_size: Number of neurons in the hidden layer
        num_classes: Number of output classes
        dropout: Ignored (kept for API compatibility)
    """

    def __init__(
        self,
        input_size: int = 16,
        hidden_size: int = 4,
        num_classes: int = 2,
        dropout: float = 0.0,  # Ignored, kept for compatibility
    ):
        super(SimpleMLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.flatten = nn.Flatten(start_dim=1)

        # Gemm 1: input_size -> hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)

        # Gemm 2: hidden_size -> num_classes
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class SimpleFlatMLP(nn.Module):
    """
    Simple two-layer MLP that accepts pre-flattened (2-D) input.

    Identical to SimpleMLP but without the Flatten layer, so no ``Flatten``
    node appears in the exported ONNX graph.

    Architecture:
    - Input:  (batch_size, input_size)  — already flat
    - FC1 (Gemm): input_size  -> hidden_size
    - FC2 (Gemm): hidden_size -> num_classes
    - Output: logits (batch_size, num_classes)

    Args:
        input_size: Size of flattened input (e.g., 16 for 4x4 images)
        hidden_size: Number of neurons in the hidden layer
        num_classes: Number of output classes
        dropout: Ignored (kept for API compatibility)
    """

    def __init__(
        self,
        input_size: int = 16,
        hidden_size: int = 4,
        num_classes: int = 2,
        dropout: float = 0.0,  # Ignored, kept for compatibility
    ):
        super(SimpleFlatMLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Gemm 1: input_size -> hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)

        # Gemm 2: hidden_size -> num_classes
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
