# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpiDeNet(nn.Module):
    """
    EpiDeNet model for EOG signal classification (Training version)

    Input: (batch, 1, C, T) - EOG signals where C is number of channels, T is time samples
    Output: (batch, N) - class logits
    """

    def __init__(self, C=16, T=1000, output_classes=11, p_dropout=0.0):
        super().__init__()
        self.C = C
        self.T = T
        self.output_classes = output_classes

        # All convolutions with padding=0
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=4, kernel_size=(1, 4), stride=(1, 1), padding=0
        )
        self.bn1 = nn.BatchNorm2d(4)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8))

        self.conv2 = nn.Conv2d(
            in_channels=4, out_channels=16, kernel_size=(1, 16), stride=(1, 1), padding=0
        )
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))

        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=(1, 8), stride=(1, 1), padding=0
        )
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))

        # Spatial convolution
        self.conv4 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=(C, 1), stride=(1, 1), padding=0
        )
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))

        self.conv5 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=(1, 1), stride=(1, 1), padding=0
        )
        self.bn5 = nn.BatchNorm2d(16)

        # Calculate final temporal dimension after all convs and pooling
        # Conv1: T - 4 + 1, Pool1: / 8
        # Conv2: - 16 + 1, Pool2: / 4
        # Conv3: - 8 + 1, Pool3: / 4
        # Conv5: - 1 + 1 (no change)
        # Formula: ((((T - 4 + 1) // 8 - 16 + 1) // 4 - 8 + 1) // 4)
        final_temporal_dim = (((T - 3) // 8 - 15) // 4 - 7) // 4
        self.pool6 = nn.AvgPool2d((1, final_temporal_dim))

        self.flatten = nn.Flatten()
        self.fcn = nn.Linear(16, output_classes)

        # Optional dropout
        self.dropout = nn.Dropout(p=p_dropout) if p_dropout > 0 else None

    def forward(self, x):
        # Input: (batch, 1, C, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool6(x)

        x = self.flatten(x)

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.fcn(x)
        return x


class EpiDeNetInference(nn.Module):
    """
    EpiDeNet model for ONNX export (Inference version)

    - No dropout
    - Explicit padding (no 'same' padding)
    - Optimized for ONNX export

    Input: (batch, 1, C, T) - EOG signals
    Output: (batch, N) - class logits
    """

    def __init__(self, C=16, T=1000, output_classes=11):
        super().__init__()
        self.C = C
        self.T = T
        self.output_classes = output_classes

        # Block 1 - All padding=0
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=4, kernel_size=(1, 4), stride=(1, 1), padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(4)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8))

        # Block 2
        self.conv2 = nn.Conv2d(
            in_channels=4,
            out_channels=16,
            kernel_size=(1, 16),
            stride=(1, 1),
            padding=0,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))

        # Block 3
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=(1, 8),
            stride=(1, 1),
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))

        # Block 4 - Spatial convolution
        self.conv4 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=(C, 1),
            stride=(1, 1),
            padding=0,
            bias=False,
        )
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))

        # Block 5
        self.conv5 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            bias=False,
        )
        self.bn5 = nn.BatchNorm2d(16)

        # Calculate final temporal dimension after all convs and pooling
        # Conv1: T - 4 + 1, Pool1: / 8
        # Conv2: - 16 + 1, Pool2: / 4
        # Conv3: - 8 + 1, Pool3: / 4
        # Conv5: - 1 + 1 (no change)
        final_temporal_dim = (((T - 3) // 8 - 15) // 4 - 7) // 4
        self.pool6 = nn.AvgPool2d((1, final_temporal_dim))

        # Classifier
        self.flatten = nn.Flatten()
        self.fcn = nn.Linear(16, output_classes)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)

        # Block 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.pool6(x)

        # Classifier
        x = self.flatten(x)
        x = self.fcn(x)

        return x

    def load_from_training_model(self, training_model):
        """Load weights from training model"""
        # Block 1
        self.conv1.weight.data = training_model.conv1.weight.data.clone()
        self.bn1.weight.data = training_model.bn1.weight.data.clone()
        self.bn1.bias.data = training_model.bn1.bias.data.clone()
        self.bn1.running_mean.data = training_model.bn1.running_mean.data.clone()
        self.bn1.running_var.data = training_model.bn1.running_var.data.clone()

        # Block 2
        self.conv2.weight.data = training_model.conv2.weight.data.clone()
        self.bn2.weight.data = training_model.bn2.weight.data.clone()
        self.bn2.bias.data = training_model.bn2.bias.data.clone()
        self.bn2.running_mean.data = training_model.bn2.running_mean.data.clone()
        self.bn2.running_var.data = training_model.bn2.running_var.data.clone()

        # Block 3
        self.conv3.weight.data = training_model.conv3.weight.data.clone()
        self.bn3.weight.data = training_model.bn3.weight.data.clone()
        self.bn3.bias.data = training_model.bn3.bias.data.clone()
        self.bn3.running_mean.data = training_model.bn3.running_mean.data.clone()
        self.bn3.running_var.data = training_model.bn3.running_var.data.clone()

        # Block 4
        self.conv4.weight.data = training_model.conv4.weight.data.clone()
        self.bn4.weight.data = training_model.bn4.weight.data.clone()
        self.bn4.bias.data = training_model.bn4.bias.data.clone()
        self.bn4.running_mean.data = training_model.bn4.running_mean.data.clone()
        self.bn4.running_var.data = training_model.bn4.running_var.data.clone()

        # Block 5
        self.conv5.weight.data = training_model.conv5.weight.data.clone()
        self.bn5.weight.data = training_model.bn5.weight.data.clone()
        self.bn5.bias.data = training_model.bn5.bias.data.clone()
        self.bn5.running_mean.data = training_model.bn5.running_mean.data.clone()
        self.bn5.running_var.data = training_model.bn5.running_var.data.clone()

        # Classifier
        self.fcn.weight.data = training_model.fcn.weight.data.clone()
        self.fcn.bias.data = training_model.fcn.bias.data.clone()


def epidenet_small(
    pretrained=False,
    C=16,
    T=1000,
    N=11,
    p_dropout=0.0,
    **kwargs,
):
    """
    Create EpiDeNet model for EOG classification

    Args:
        pretrained: Path to pretrained weights or False
        C: Number of EOG channels (default: 16)
        T: Number of time samples (default: 1000)
        N: Number of classes (default: 11)
        p_dropout: Dropout probability

    Returns:
        EpiDeNet model instance
    """
    model = EpiDeNet(C=C, T=T, output_classes=N, p_dropout=p_dropout)

    if pretrained and pretrained is not False:
        checkpoint = torch.load(pretrained)
        if isinstance(checkpoint, dict) and "net" in checkpoint:
            model.load_state_dict(checkpoint["net"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Loaded pretrained weights from {pretrained}")

    return model


class EpiDeNetDeployGroupNorm(nn.Module):
    """
    EpiDeNet with GroupNorm (num_groups=1) for gradient-compatible training reference.

    This version uses GroupNorm instead of LayerNorm so that:
    - weight shape is [C] instead of [C, H, W]
    - gradient dGamma shape is [C], matching Deeploy's GroupNormGradW output

    Use this model to generate reference gradients for Deeploy testing.
    """

    def __init__(self, C=16, T=1000, output_classes=11):
        super().__init__()
        self.C = C
        self.T = T
        self.output_classes = output_classes

        # Calculate dimensions at each stage
        T1 = T - 3
        T1_pool = T1 // 8
        T2 = T1_pool - 15
        T2_pool = T2 // 4
        T3 = T2_pool - 7
        T3_pool = T3 // 4

        self.T1 = T1
        self.T1_pool = T1_pool
        self.T2 = T2
        self.T2_pool = T2_pool
        self.T3 = T3
        self.T3_pool = T3_pool

        # Block 1
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=4, kernel_size=(1, 4), stride=(1, 1), padding=0, bias=False
        )
        self.layer_norm1 = nn.GroupNorm(num_groups=1, num_channels=4, eps=0.001)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))

        # Block 2
        self.conv2 = nn.Conv2d(
            in_channels=4,
            out_channels=16,
            kernel_size=(1, 16),
            stride=(1, 1),
            padding=0,
            bias=False,
        )
        self.layer_norm2 = nn.GroupNorm(num_groups=1, num_channels=16, eps=0.001)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))

        # Block 3
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=(1, 8),
            stride=(1, 1),
            padding=0,
            bias=False,
        )
        self.layer_norm3 = nn.GroupNorm(num_groups=1, num_channels=16, eps=0.001)
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))

        # Block 4 - Spatial convolution
        self.conv4 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=(C, 1),
            stride=(1, 1),
            padding=0,
            bias=False,
        )
        self.layer_norm4 = nn.GroupNorm(num_groups=1, num_channels=16, eps=0.001)
        self.pool4 = nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1))

        # Block 5
        self.conv5 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            bias=False,
        )
        self.layer_norm5 = nn.GroupNorm(num_groups=1, num_channels=16, eps=0.001)

        # Initialize biases to match EpiDeNetDeploy (non-zero to avoid randomization)
        # and layer_norm5 weight differently to prevent ONNX weight sharing
        nn.init.constant_(self.layer_norm1.bias, 0.0001)
        nn.init.constant_(self.layer_norm2.bias, 0.0002)
        nn.init.constant_(self.layer_norm3.bias, 0.0003)
        nn.init.constant_(self.layer_norm4.bias, 0.0004)
        nn.init.constant_(self.layer_norm5.weight, 1.001)
        nn.init.constant_(self.layer_norm5.bias, 0.0005)

        # Final pooling
        final_temporal_dim = T3_pool
        self.pool6 = nn.AvgPool2d((1, final_temporal_dim))

        # Classifier
        self.fcn = nn.Linear(16, output_classes, bias=True)
        self.fc_input_size = 16

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.layer_norm1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.layer_norm2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.layer_norm3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Block 4
        x = self.conv4(x)
        x = self.layer_norm4(x)
        x = F.relu(x)
        x = self.pool4(x)

        # Block 5
        x = self.conv5(x)
        x = self.layer_norm5(x)
        x = F.relu(x)
        x = self.pool6(x)

        # Classifier
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.fc_input_size)
        x = self.fcn(x)

        return x

    def load_weights_from_layernorm_model(self, layernorm_model):
        """
        Load weights from EpiDeNetDeploy (LayerNorm version).
        LayerNorm weights [C, H, W] are converted to GroupNorm weights [C] via mean.
        """
        # Copy conv weights
        self.conv1.weight.data = layernorm_model.conv1.weight.data.clone()
        self.conv2.weight.data = layernorm_model.conv2.weight.data.clone()
        self.conv3.weight.data = layernorm_model.conv3.weight.data.clone()
        self.conv4.weight.data = layernorm_model.conv4.weight.data.clone()
        self.conv5.weight.data = layernorm_model.conv5.weight.data.clone()
        self.fcn.weight.data = layernorm_model.fcn.weight.data.clone()
        self.fcn.bias.data = layernorm_model.fcn.bias.data.clone()

        # Convert LayerNorm weights [C, H, W] -> GroupNorm weights [C]
        for i in range(1, 6):
            ln = getattr(layernorm_model, f"layer_norm{i}")
            gn = getattr(self, f"layer_norm{i}")
            ln_w = ln.weight.data
            ln_b = ln.bias.data
            gn.weight.data = ln_w.reshape(ln_w.shape[0], -1).mean(dim=1)
            gn.bias.data = ln_b.reshape(ln_b.shape[0], -1).mean(dim=1)

        print("✅ Loaded weights from LayerNorm model to GroupNorm model")
