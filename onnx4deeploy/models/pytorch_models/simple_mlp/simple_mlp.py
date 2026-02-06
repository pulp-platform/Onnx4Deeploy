# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Simple Multi-Layer Perceptron (MLP) for demonstration - Deploy Version."""

import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for image classification (Deploy Version).

    This is a clean deployment version optimized for ONNX export:
    - No dropout layers (removed for inference)
    - Fixed input shape (no dynamic branching)
    - Uses nn.Flatten for clarity
    - Clean, linear computation graph

    Architecture:
    - Input: Image (batch_size, 1, height, width)
    - Flatten: (batch_size, input_size)
    - FC1: (input_size) -> (hidden_size) + ReLU
    - FC2: (hidden_size) -> (hidden_size) + ReLU
    - FC3: (hidden_size) -> (num_classes)
    - Output: Logits (batch_size, num_classes)

    Args:
        input_size: Size of flattened input (e.g., 784 for 28x28 images)
        hidden_size: Number of neurons in hidden layers
        num_classes: Number of output classes
        dropout: Ignored (kept for API compatibility, always uses 0.0)
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 128,
        num_classes: int = 10,
        dropout: float = 0.0,  # Ignored, kept for compatibility
    ):
        super(SimpleMLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Flatten layer (explicit for ONNX)
        self.flatten = nn.Flatten(start_dim=1)

        # Layer 1: Input -> Hidden
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()

        # Layer 2: Hidden -> Hidden
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()

        # Layer 3: Hidden -> Output
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass (clean, no branching).

        Args:
            x: Input tensor of shape (batch_size, 1, height, width)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Flatten: (B, 1, H, W) -> (B, H*W)
        x = self.flatten(x)

        # Layer 1: Linear + ReLU
        x = self.fc1(x)
        x = self.relu1(x)

        # Layer 2: Linear + ReLU
        x = self.fc2(x)
        x = self.relu2(x)

        # Layer 3: Linear (output logits)
        x = self.fc3(x)

        return x
