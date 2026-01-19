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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1,4), stride=(1,1), padding=0)
        self.bn1 = nn.BatchNorm2d(4)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,8), stride=(1,8))

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(1,16), stride=(1,1), padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=(1,4), stride=(1,4))

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,8), stride=(1,1), padding=0)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=(1,4), stride=(1,4))

        # Spatial convolution
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(C,1), stride=(1,1), padding=0)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(kernel_size=(1,1), stride=(1,1))

        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,1), stride=(1,1), padding=0)
        self.bn5 = nn.BatchNorm2d(16)

        # Calculate final temporal dimension after all convs and pooling
        # Conv1: T - 4 + 1, Pool1: / 8
        # Conv2: - 16 + 1, Pool2: / 4
        # Conv3: - 8 + 1, Pool3: / 4
        # Conv5: - 1 + 1 (no change)
        # Formula: ((((T - 4 + 1) // 8 - 16 + 1) // 4 - 8 + 1) // 4)
        final_temporal_dim = ((((T - 3) // 8 - 15) // 4 - 7) // 4)
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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1,4), stride=(1,1), padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,8), stride=(1,8))

        # Block 2
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(1,16), stride=(1,1), padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=(1,4), stride=(1,4))

        # Block 3
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,8), stride=(1,1), padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=(1,4), stride=(1,4))

        # Block 4 - Spatial convolution
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(C,1), stride=(1,1), padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(kernel_size=(1,1), stride=(1,1))

        # Block 5
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,1), stride=(1,1), padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(16)

        # Calculate final temporal dimension after all convs and pooling
        # Conv1: T - 4 + 1, Pool1: / 8
        # Conv2: - 16 + 1, Pool2: / 4
        # Conv3: - 8 + 1, Pool3: / 4
        # Conv5: - 1 + 1 (no change)
        final_temporal_dim = ((((T - 3) // 8 - 15) // 4 - 7) // 4)
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


class EpiDeNetDeploy(nn.Module):
    """
    Deployment-optimized EpiDeNet for on-chip inference

    Features:
    - BatchNorm fused into Conv layers (all Conv have bias=True)
    - No Dropout layers
    - No padding layers (minimal operations)
    - MaxPool2d replaced with AvgPool2d for smoother downsampling
    - Clean ONNX export for edge deployment

    Input: (batch, 1, C, T) - EOG signals
    Output: (batch, N) - class logits
    """

    def __init__(self, C=16, T=1000, output_classes=11):
        super().__init__()
        self.C = C
        self.T = T
        self.output_classes = output_classes

        # All Conv layers have bias=True (BatchNorm is fused), padding=0
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1,4), stride=(1,1), padding=0, bias=True)
        self.pool1 = nn.AvgPool2d(kernel_size=(1,8), stride=(1,8))

        # Block 2
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(1,16), stride=(1,1), padding=0, bias=True)
        self.pool2 = nn.AvgPool2d(kernel_size=(1,4), stride=(1,4))

        # Block 3
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,8), stride=(1,1), padding=0, bias=True)
        self.pool3 = nn.AvgPool2d(kernel_size=(1,4), stride=(1,4))

        # Block 4 - Spatial convolution
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(C,1), stride=(1,1), padding=0, bias=True)
        self.pool4 = nn.AvgPool2d(kernel_size=(1,1), stride=(1,1))

        # Block 5
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,1), stride=(1,1), padding=0, bias=True)

        # Calculate final temporal dimension after all convs and pooling
        # Conv1: T - 4 + 1, Pool1: / 8
        # Conv2: - 16 + 1, Pool2: / 4
        # Conv3: - 8 + 1, Pool3: / 4
        # Conv5: - 1 + 1 (no change)
        final_temporal_dim = ((((T - 3) // 8 - 15) // 4 - 7) // 4)
        self.pool6 = nn.AvgPool2d((1, final_temporal_dim))

        # Classifier
        self.fcn = nn.Linear(16, output_classes, bias=True)

        # Store feature size for reshaping
        self.fc_input_size = 16

    def forward(self, x):
        """
        Forward pass for deployment

        Args:
            x: (batch, 1, C, T) - Input EOG signals

        Returns:
            (batch, N) - Class logits
        """
        # Block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Block 4
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)

        # Block 5
        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool6(x)

        # Classifier - use explicit reshape instead of flatten for cleaner ONNX
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.fc_input_size)
        x = self.fcn(x)

        return x

    def fuse_bn_to_conv(self, conv, bn):
        """
        Fuse BatchNorm parameters into Conv layer

        Math:
            Conv: y = W*x + b_conv
            BN:   z = gamma*(y - mu)/sigma + beta
            Fused: z = (gamma*W/sigma)*x + (gamma*(b_conv - mu)/sigma + beta)
        """
        # Get Conv parameters
        w_conv = conv.weight.data.clone()
        if conv.bias is not None:
            b_conv = conv.bias.data.clone()
        else:
            b_conv = torch.zeros(conv.out_channels, device=w_conv.device, dtype=w_conv.dtype)

        # Get BN parameters
        gamma = bn.weight.data.clone()
        beta = bn.bias.data.clone()
        running_mean = bn.running_mean.data.clone()
        running_var = bn.running_var.data.clone()
        eps = bn.eps

        # Compute fused parameters
        std = torch.sqrt(running_var + eps)
        w_fused = w_conv * (gamma / std).view(-1, 1, 1, 1)
        b_fused = (b_conv - running_mean) * gamma / std + beta

        return w_fused, b_fused

    def load_and_fuse_from_training_model(self, training_model):
        """
        Load weights from training model and fuse BatchNorm

        Args:
            training_model: EpiDeNet or EpiDeNetInference instance with BatchNorm
        """
        training_model.eval()  # Ensure BN uses running stats

        # Fuse Block 1
        w1, b1 = self.fuse_bn_to_conv(training_model.conv1, training_model.bn1)
        self.conv1.weight.data = w1
        self.conv1.bias.data = b1

        # Fuse Block 2
        w2, b2 = self.fuse_bn_to_conv(training_model.conv2, training_model.bn2)
        self.conv2.weight.data = w2
        self.conv2.bias.data = b2

        # Fuse Block 3
        w3, b3 = self.fuse_bn_to_conv(training_model.conv3, training_model.bn3)
        self.conv3.weight.data = w3
        self.conv3.bias.data = b3

        # Fuse Block 4
        w4, b4 = self.fuse_bn_to_conv(training_model.conv4, training_model.bn4)
        self.conv4.weight.data = w4
        self.conv4.bias.data = b4

        # Fuse Block 5
        w5, b5 = self.fuse_bn_to_conv(training_model.conv5, training_model.bn5)
        self.conv5.weight.data = w5
        self.conv5.bias.data = b5

        # Classifier (no fusion needed)
        self.fcn.weight.data = training_model.fcn.weight.data.clone()
        self.fcn.bias.data = training_model.fcn.bias.data.clone()

        print("✅ Successfully loaded and fused weights from training model")


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
        if isinstance(checkpoint, dict) and 'net' in checkpoint:
            model.load_state_dict(checkpoint['net'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Loaded pretrained weights from {pretrained}")

    return model


def epidenet_deploy(
    pretrained_model=None,
    C=16,
    T=1000,
    N=11,
    **kwargs,
):
    """
    Create deployment-optimized EpiDeNet model

    Args:
        pretrained_model: Optional EpiDeNet instance to load weights from
        C: Number of EOG channels
        T: Number of time samples
        N: Number of classes

    Returns:
        EpiDeNetDeploy instance ready for ONNX export

    Example:
        # Load pretrained training model
        train_model = epidenet_small(pretrained='model.pth', C=16, T=1000, N=11)

        # Create deployment model and fuse
        deploy_model = epidenet_deploy(pretrained_model=train_model, C=16, T=1000, N=11)

        # Export to ONNX
        dummy_input = torch.randn(1, 1, 16, 1000)
        torch.onnx.export(deploy_model, dummy_input, "epidenet_deploy.onnx")
    """
    model = EpiDeNetDeploy(C=C, T=T, output_classes=N)

    if pretrained_model is not None:
        model.load_and_fuse_from_training_model(pretrained_model)

    return model
