"""Lightweight CNN for image classification - Deploy Version."""

import torch.nn as nn
import torch.nn.init as init


class LightweightCNN(nn.Module):
    """
    Lightweight CNN for image classification (Deploy Version).

    This is a clean deployment version optimized for ONNX export:
    - No dropout layers (removed for inference)
    - Fixed input shape (no dynamic operations)
    - Static padding values (no dynamic padding)
    - Uses nn.Flatten for ONNX compatibility
    - Clean, linear computation graph

    Architecture:
    - Input: Image (batch_size, input_channels, 28, 28)
    - Conv1: (input_channels, 28, 28) -> (20, 24, 24) + ReLU + MaxPool -> (20, 12, 12)
    - Conv2: (20, 12, 12) -> (10, 12, 12) + ReLU + MaxPool -> (10, 6, 6)
    - Conv3: (10, 6, 6) -> (12, 4, 4) + ReLU
    - Conv4: (12, 4, 4) -> (10, 4, 4) + ReLU
    - Flatten: (10, 4, 4) -> (160)
    - FC: (160) -> (num_classes)
    - Output: Logits (batch_size, num_classes)

    Args:
        batch_size: Fixed batch size for deployment (default: 1)
        input_channels: Number of input channels (default: 1 for grayscale)
        num_classes: Number of output classes (default: 10)
        dropout: Ignored (kept for API compatibility, always uses 0.0)
    """

    def __init__(
        self,
        batch_size: int = 1,
        input_channels: int = 1,
        num_classes: int = 10,
        dropout: float = 0.0,  # Ignored, kept for compatibility
    ):
        super(LightweightCNN, self).__init__()

        # Shape constants (fixed for ONNX export)
        self.batch_size = batch_size
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.fc_channels = 160  # Fixed: 10 * 4 * 4 = 160

        # Convolutional Block 1: Input -> Conv1 -> ReLU -> Pool
        # Input: (B, input_channels, 28, 28) -> Output: (B, 20, 12, 12)
        self.conv1 = nn.Conv2d(input_channels, 20, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 2: Conv2 -> ReLU -> Pool
        # Input: (B, 20, 12, 12) -> Output: (B, 10, 6, 6)
        self.conv2 = nn.Conv2d(20, 10, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 3: Conv3 -> ReLU
        # Input: (B, 10, 6, 6) -> Output: (B, 12, 4, 4)
        self.conv3 = nn.Conv2d(10, 12, kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.ReLU()

        # Convolutional Block 4: Conv4 -> ReLU
        # Input: (B, 12, 4, 4) -> Output: (B, 10, 4, 4)
        self.conv4 = nn.Conv2d(12, 10, kernel_size=1, stride=1, padding=0)
        self.relu4 = nn.ReLU()

        # Flatten layer (explicit for ONNX)
        # Input: (B, 10, 4, 4) -> Output: (B, 160)
        self.flatten = nn.Flatten(start_dim=1)

        # Fully connected layer
        # Input: (B, 160) -> Output: (B, num_classes)
        self.fc = nn.Linear(self.fc_channels, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights for the model.
        Applies Kaiming Initialization for Conv2D layers and Xavier Initialization for Linear layers.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Kaiming Initialization for convolutional layers
                init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Xavier Initialization for fully connected layers
                init.xavier_normal_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward pass (clean, no branching, no dynamic operations).

        Args:
            x: Input tensor of shape (batch_size, input_channels, 28, 28)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Convolutional Block 1
        # (B, input_channels, 28, 28) -> (B, 20, 24, 24)
        x = self.conv1(x)
        x = self.relu1(x)
        # (B, 20, 24, 24) -> (B, 20, 12, 12)
        x = self.pool1(x)

        # Convolutional Block 2
        # (B, 20, 12, 12) -> (B, 10, 12, 12)
        x = self.conv2(x)
        x = self.relu2(x)
        # (B, 10, 12, 12) -> (B, 10, 6, 6)
        x = self.pool2(x)

        # Convolutional Block 3
        # (B, 10, 6, 6) -> (B, 12, 4, 4)
        x = self.conv3(x)
        x = self.relu3(x)

        # Convolutional Block 4
        # (B, 12, 4, 4) -> (B, 10, 4, 4)
        x = self.conv4(x)
        x = self.relu4(x)

        # Flatten
        # (B, 10, 4, 4) -> (B, 160)
        x = self.flatten(x)

        # Fully connected layer
        # (B, 160) -> (B, num_classes)
        x = self.fc(x)

        return x
