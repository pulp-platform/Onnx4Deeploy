import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDropout2d(nn.Dropout2d):
    """
    Dropout layer, where the last dimension is treated as channels
    """

    def __init__(self, p=0.5, inplace=False):
        super(TimeDropout2d, self).__init__(p=p, inplace=inplace)

    def forward(self, input):
        if self.training:
            input = input.permute(0, 3, 1, 2)
            input = F.dropout2d(input, self.p, True, self.inplace)
            input = input.permute(0, 2, 3, 1)
        return input


class MIBMINet(nn.Module):
    def __init__(
        self,
        F1=8,
        D=2,
        F2=None,
        C=8,
        T=2000,
        N=4,
        Nf=64,
        Nf2=16,
        p_dropout=0.5,
        activation="relu",
        seed=-1,
        dropout_type="dropout",
        pretrained=None,
        fold_id=0,
        # Unused args for compatibility
        img_size=None,
        num_classes=None,
        embedding_dim=None,
        num_heads=None,
        num_layers=None,
        in_channels=None,
        **kwargs,
    ):
        """
        F1:           Number of spectral filters
        D:            Number of spacial filters (per spectral filter), F2 = F1 * D
        F2:           Number or None. If None, then F2 = F1 * D
        C:            Number of EEG channels
        T:            Number of time samples
        N:            Number of classes
        Nf:           Size of temporal filter
        p_dropout:    Dropout Probability
        activation:   string, either 'elu' or 'relu'
        dropout_type: string, either 'dropout', 'SpatialDropout2d' or 'TimeDropout2D'
        """
        super(MIBMINet, self).__init__()

        # prepare network constants
        if F2 is None:
            F2 = F1 * D

        # check the activation input
        activation = activation.lower()
        assert activation in ["elu", "relu"]
        print("activation relu?: ", activation)

        # Prepare Dropout Type
        if dropout_type.lower() == "dropout":
            dropout = nn.Dropout
        elif dropout_type.lower() == "spatialdropout2d":
            dropout = nn.Dropout2d
        elif dropout_type.lower() == "timedropout2d":
            dropout = TimeDropout2d
        else:
            raise ValueError(
                "dropout_type must be one of SpatialDropout2d, Dropout or " "TimeDropout2d"
            )

        # store local values
        self.F1, self.D, self.F2, self.C, self.T, self.N = (F1, D, F2, C, T, N)
        self.p_dropout, self.activation = (p_dropout, activation)

        # Number of input neurons to the final fully connected layer
        n_features = (T // 8) // 8

        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (C, 1), bias=False)
        self.batch_norm1 = nn.BatchNorm2d(F1, momentum=0.01, eps=0.001)
        self.before_conv2_pad = nn.ZeroPad2d((Nf // 2 - 1, Nf // 2, 0, 0))
        self.conv2 = nn.Conv2d(F1, D * F1, (1, Nf), groups=F1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(D * F1, momentum=0.01, eps=0.001)
        self.activation1 = nn.ELU(inplace=False) if activation == "elu" else nn.ReLU(inplace=False)
        self.pool1 = nn.AvgPool2d((1, 8))
        self.dropout1 = dropout(p=p_dropout)

        # Block 2
        self.sep_conv_pad = nn.ZeroPad2d((Nf2 // 2 - 1, Nf2 // 2, 0, 0))
        self.sep_conv1 = nn.Conv2d(D * F1, D * F1, (1, Nf2), groups=D * F1, bias=False)
        self.sep_conv2 = nn.Conv2d(D * F1, F2, (1, 1), bias=False)
        self.batch_norm3 = nn.BatchNorm2d(F2, momentum=0.01, eps=0.001)
        self.activation2 = nn.ELU(inplace=False) if activation == "elu" else nn.ReLU(inplace=False)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = dropout(p=p_dropout)

        # Fully connected layer (classifier)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(F2 * n_features, N, bias=True)

        # initialize weights
        self._initialize_params(seed=seed)

        print("fold_id: ")
        print(fold_id)

        if pretrained is not None and pretrained is not False:
            if isinstance(pretrained, list):
                ckpt = torch.load(pretrained[fold_id])
            else:
                ckpt = torch.load(pretrained)
            # network parameters
            self.load_state_dict(ckpt["net"])

    def forward(self, x):
        # input dimensions: (s, 1, C, T) [batchsize, 1, numb of Channels, T]

        # Block 1
        x = self.conv1(x)  # output dim: (s, F1, 1, T)
        x = self.batch_norm1(x)
        x = self.before_conv2_pad(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation1(x)
        x = self.pool1(x)  # output dim: (s, D * F1, 1, T // 8)
        x = self.dropout1(x)

        # Block2
        x = self.sep_conv_pad(x)
        x = self.sep_conv1(x)  # output dim: (s, D * F1, 1, T // 8)
        x = self.sep_conv2(x)  # output dim: (s, F2, 1, T // 8)
        x = self.batch_norm3(x)
        x = self.activation2(x)
        x = self.pool2(x)  # output dim: (s, F2, 1, T // 64)
        x = self.dropout2(x)

        # Classification
        x = self.flatten(x)  # output dim: (s, F2 * (T // 64))
        x = self.fc(x)  # output dim: (s, N)

        return x

    def forward_with_tensor_stats(self, x):
        return self.forward(x)

    def _initialize_params(
        self, seed, weight_init=nn.init.xavier_uniform_, bias_init=nn.init.zeros_
    ):
        """
        Initializes all the parameters of the model

        Parameters:
         - weight_init: nn.init inplace function
         - bias_init:   nn.init inplace function
        """

        if seed >= 0:
            torch.manual_seed(seed)
            # using GPU
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        def init_weight(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                weight_init(m.weight)
            if isinstance(m, nn.Linear):
                bias_init(m.bias)

        self.apply(init_weight)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class MIBMINetInference(nn.Module):
    """Simplified MI-BMINet for clean ONNX export (no dropout, explicit padding)"""

    def __init__(
        self,
        F1=8,
        D=2,
        F2=None,
        C=8,
        T=2000,
        N=4,
        Nf=64,
        Nf2=16,
        activation="relu",
        **kwargs,
    ):
        super(MIBMINetInference, self).__init__()

        if F2 is None:
            F2 = F1 * D

        activation = activation.lower()
        assert activation in ["elu", "relu"]

        self.F1, self.D, self.F2, self.C, self.T, self.N = (F1, D, F2, C, T, N)
        self.Nf = Nf
        self.Nf2 = Nf2

        # Calculate output dimensions without padding
        # After conv2: T - Nf + 1
        # After pool1: (T - Nf + 1) // 8
        # After sep_conv1: ((T - Nf + 1) // 8) - Nf2 + 1
        # After pool2: (((T - Nf + 1) // 8) - Nf2 + 1) // 8
        T_after_conv2 = T - Nf + 1
        T_after_pool1 = T_after_conv2 // 8
        T_after_sepconv1 = T_after_pool1 - Nf2 + 1
        n_features = T_after_sepconv1 // 8

        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (C, 1), bias=False)
        self.batch_norm1 = nn.BatchNorm2d(F1, momentum=0.01, eps=0.001)
        self.conv2 = nn.Conv2d(F1, D * F1, (1, Nf), groups=F1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(D * F1, momentum=0.01, eps=0.001)
        # Use inplace=False for cleaner ONNX export (avoids Identity nodes)
        self.activation1 = nn.ELU(inplace=False) if activation == "elu" else nn.ReLU(inplace=False)
        self.pool1 = nn.AvgPool2d((1, 8))

        # Block 2
        self.sep_conv1 = nn.Conv2d(D * F1, D * F1, (1, Nf2), groups=D * F1, bias=False)
        self.sep_conv2 = nn.Conv2d(D * F1, F2, (1, 1), bias=False)
        self.batch_norm3 = nn.BatchNorm2d(F2, momentum=0.01, eps=0.001)
        # Use inplace=False for cleaner ONNX export (avoids Identity nodes)
        self.activation2 = nn.ELU(inplace=False) if activation == "elu" else nn.ReLU(inplace=False)
        self.pool2 = nn.AvgPool2d((1, 8))

        # Fully connected layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(F2 * n_features, N, bias=True)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.batch_norm1(x)
        # No padding - let dimensions shrink naturally
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation1(x)
        x = self.pool1(x)

        # Block 2
        # No padding - let dimensions shrink naturally
        x = self.sep_conv1(x)
        x = self.sep_conv2(x)
        x = self.batch_norm3(x)
        x = self.activation2(x)
        x = self.pool2(x)

        # Classification
        x = self.flatten(x)
        x = self.fc(x)

        return x

    def load_from_training_model(self, training_model):
        """Load weights from training model"""
        # Copy weights from training model
        self.conv1.weight.data = training_model.conv1.weight.data.clone()
        self.batch_norm1.weight.data = training_model.batch_norm1.weight.data.clone()
        self.batch_norm1.bias.data = training_model.batch_norm1.bias.data.clone()
        self.batch_norm1.running_mean.data = training_model.batch_norm1.running_mean.data.clone()
        self.batch_norm1.running_var.data = training_model.batch_norm1.running_var.data.clone()

        self.conv2.weight.data = training_model.conv2.weight.data.clone()
        self.batch_norm2.weight.data = training_model.batch_norm2.weight.data.clone()
        self.batch_norm2.bias.data = training_model.batch_norm2.bias.data.clone()
        self.batch_norm2.running_mean.data = training_model.batch_norm2.running_mean.data.clone()
        self.batch_norm2.running_var.data = training_model.batch_norm2.running_var.data.clone()

        self.sep_conv1.weight.data = training_model.sep_conv1.weight.data.clone()
        self.sep_conv2.weight.data = training_model.sep_conv2.weight.data.clone()
        self.batch_norm3.weight.data = training_model.batch_norm3.weight.data.clone()
        self.batch_norm3.bias.data = training_model.batch_norm3.bias.data.clone()
        self.batch_norm3.running_mean.data = training_model.batch_norm3.running_mean.data.clone()
        self.batch_norm3.running_var.data = training_model.batch_norm3.running_var.data.clone()

        self.fc.weight.data = training_model.fc.weight.data.clone()
        self.fc.bias.data = training_model.fc.bias.data.clone()


def mi_bminet_small(
    pretrained=False,
    img_size=None,
    num_classes=None,
    embedding_dim=None,
    num_heads=None,
    num_layers=None,
    in_channels=None,
    # BCI-specific parameters
    F1=8,
    D=2,
    F2=None,
    C=8,
    T=2000,
    N=4,
    Nf=64,
    Nf2=16,
    p_dropout=0.5,
    activation="relu",
    seed=-1,
    dropout_type="dropout",
    fold_id=0,
    **kwargs,
):
    """
    Create MI-BMINet model for BCI tasks
    Default parameters are for BCI Drone 4-class task
    """
    model = MIBMINet(
        F1=F1,
        D=D,
        F2=F2,
        C=C,
        T=T,
        N=N,
        Nf=Nf,
        Nf2=Nf2,
        p_dropout=p_dropout,
        activation=activation,
        seed=seed,
        dropout_type=dropout_type,
        pretrained=pretrained,
        fold_id=fold_id,
        **kwargs,
    )
    return model


def mi_bminet_base(
    pretrained=False,
    img_size=None,
    num_classes=None,
    embedding_dim=None,
    num_heads=None,
    num_layers=None,
    in_channels=None,
    # BCI-specific parameters
    F1=8,
    D=2,
    F2=None,
    C=8,
    T=2000,
    N=4,
    Nf=64,
    Nf2=16,
    p_dropout=0.5,
    activation="relu",
    seed=-1,
    dropout_type="dropout",
    fold_id=0,
    **kwargs,
):
    """
    Create MI-BMINet model for BCI tasks
    Default parameters are for BCI Drone 4-class task
    """
    model = MIBMINet(
        F1=F1,
        D=D,
        F2=F2,
        C=C,
        T=T,
        N=N,
        Nf=Nf,
        Nf2=Nf2,
        p_dropout=p_dropout,
        activation=activation,
        seed=seed,
        dropout_type=dropout_type,
        pretrained=pretrained,
        fold_id=fold_id,
        **kwargs,
    )
    return model


class MIBMINetDeploy(nn.Module):
    """
    Deployment-optimized MI-BMINet for on-chip inference

    Features:
    - LayerNorm for normalization (ONNX training compatible)
    - No Dropout layers (not needed in inference)
    - No padding layers (minimal operations)
    - Clean ONNX export for edge deployment

    Input: (batch, 1, C, T) - EEG signals
    Output: (batch, N) - class logits
    """

    def __init__(
        self,
        F1=8,
        D=2,
        F2=None,
        C=8,
        T=2000,
        N=4,
        Nf=64,
        Nf2=16,
        activation="relu",
        **kwargs,
    ):
        super(MIBMINetDeploy, self).__init__()

        if F2 is None:
            F2 = F1 * D

        activation = activation.lower()
        assert activation in [
            "elu",
            "relu",
        ], f"activation must be 'elu' or 'relu', got {activation}"

        self.F1, self.D, self.F2, self.C, self.T, self.N = (F1, D, F2, C, T, N)
        self.Nf = Nf
        self.Nf2 = Nf2

        # Calculate feature size after all pooling
        # After conv1: (batch, F1, 1, T)
        # After conv2: (batch, D*F1, 1, T - Nf + 1)
        # After pool1: (batch, D*F1, 1, (T - Nf + 1) // 8)
        # After sep_conv1: (batch, D*F1, 1, ((T - Nf + 1) // 8) - Nf2 + 1)
        # After sep_conv2: (batch, F2, 1, ((T - Nf + 1) // 8) - Nf2 + 1)
        # After pool2: (batch, F2, 1, (((T - Nf + 1) // 8) - Nf2 + 1) // 8)
        T_after_conv2 = T - Nf + 1
        T_after_pool1 = T_after_conv2 // 8
        T_after_sepconv1 = T_after_pool1 - Nf2 + 1
        n_features = T_after_sepconv1 // 8

        # Block 1: Spectral + Spatial filtering
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(C, 1), bias=False)
        self.layer_norm1 = nn.LayerNorm([F1, 1, T], eps=0.001)
        self.conv2 = nn.Conv2d(F1, D * F1, kernel_size=(1, Nf), groups=F1, bias=False)
        self.layer_norm2 = nn.LayerNorm([D * F1, 1, T_after_conv2], eps=0.001)
        self.activation1 = nn.ELU(inplace=False) if activation == "elu" else nn.ReLU(inplace=False)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 8))

        # Block 2: Depthwise separable convolution
        self.sep_conv1 = nn.Conv2d(D * F1, D * F1, kernel_size=(1, Nf2), groups=D * F1, bias=False)
        self.sep_conv2 = nn.Conv2d(D * F1, F2, kernel_size=(1, 1), bias=False)
        self.layer_norm3 = nn.LayerNorm([F2, 1, T_after_sepconv1], eps=0.001)
        self.activation2 = nn.ELU(inplace=False) if activation == "elu" else nn.ReLU(inplace=False)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))

        # Classifier
        self.fc = nn.Linear(F2 * n_features, N, bias=True)
        self.fc_input_size = F2 * n_features

    def forward(self, x):
        """
        Forward pass for deployment

        Args:
            x: (batch, 1, C, T) - Input EEG signals

        Returns:
            (batch, N) - Class logits
        """
        # Block 1
        x = self.conv1(x)  # (batch, F1, 1, T)
        x = self.layer_norm1(x)
        x = self.conv2(x)  # (batch, D*F1, 1, T-Nf+1)
        x = self.layer_norm2(x)
        x = self.activation1(x)
        x = self.pool1(x)  # (batch, D*F1, 1, (T-Nf+1)//8)

        # Block 2
        x = self.sep_conv1(x)  # (batch, D*F1, 1, (T-Nf+1)//8-Nf2+1)
        x = self.sep_conv2(x)  # (batch, F2, 1, (T-Nf+1)//8-Nf2+1)
        x = self.layer_norm3(x)
        x = self.activation2(x)
        x = self.pool2(x)  # (batch, F2, 1, ((T-Nf+1)//8-Nf2+1)//8)

        # Classifier - use explicit reshape instead of flatten for cleaner ONNX
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.fc_input_size)  # (batch, F2*n_features)
        x = self.fc(x)  # (batch, N)

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
            training_model: MIBMINet or MIBMINetInference instance with BatchNorm
        """
        training_model.eval()  # Ensure BN uses running stats

        # Fuse Block 1
        w1, b1 = self.fuse_bn_to_conv(training_model.conv1, training_model.batch_norm1)
        self.conv1.weight.data = w1
        self.conv1.bias.data = b1

        w2, b2 = self.fuse_bn_to_conv(training_model.conv2, training_model.batch_norm2)
        self.conv2.weight.data = w2
        self.conv2.bias.data = b2

        # Fuse Block 2
        # sep_conv1 has no BatchNorm, copy directly
        self.sep_conv1.weight.data = training_model.sep_conv1.weight.data.clone()
        if hasattr(training_model.sep_conv1, "bias") and training_model.sep_conv1.bias is not None:
            self.sep_conv1.bias.data = training_model.sep_conv1.bias.data.clone()
        else:
            # Initialize bias to zero if original doesn't have it
            nn.init.zeros_(self.sep_conv1.bias)

        # sep_conv2 + batch_norm3
        w3, b3 = self.fuse_bn_to_conv(training_model.sep_conv2, training_model.batch_norm3)
        self.sep_conv2.weight.data = w3
        self.sep_conv2.bias.data = b3

        # Classifier (no fusion needed)
        self.fc.weight.data = training_model.fc.weight.data.clone()
        self.fc.bias.data = training_model.fc.bias.data.clone()

        print("Successfully loaded and fused weights from training model")


def mi_bminet_deploy(
    pretrained_model=None,
    F1=8,
    D=2,
    F2=None,
    C=8,
    T=2000,
    N=4,
    Nf=64,
    Nf2=16,
    activation="relu",
    **kwargs,
):
    """
    Create deployment-optimized MI-BMINet model

    Args:
        pretrained_model: Optional MIBMINet instance to load weights from
        Other args: Same as MIBMINet

    Returns:
        MIBMINetDeploy instance ready for ONNX export

    Example:
        # Load pretrained training model
        train_model = mi_bminet_small(pretrained='model.pth')

        # Create deployment model and fuse
        deploy_model = mi_bminet_deploy(pretrained_model=train_model, C=8, T=2000, N=4)

        # Export to ONNX
        dummy_input = torch.randn(1, 1, 8, 2000)
        torch.onnx.export(deploy_model, dummy_input, "model_deploy.onnx")
    """
    model = MIBMINetDeploy(
        F1=F1,
        D=D,
        F2=F2,
        C=C,
        T=T,
        N=N,
        Nf=Nf,
        Nf2=Nf2,
        activation=activation,
        **kwargs,
    )

    if pretrained_model is not None:
        model.load_and_fuse_from_training_model(pretrained_model)

    return model


class MIBMINetDeployGroupNorm(nn.Module):
    """
    MI-BMINet with GroupNorm (num_groups=1) for gradient-compatible training reference.

    This version uses GroupNorm instead of LayerNorm so that:
    - weight shape is [C] instead of [C, H, W]
    - gradient dGamma shape is [C], matching Deeploy's GroupNormGradW output

    Use this model to generate reference gradients for Deeploy testing.
    """

    def __init__(
        self,
        F1=8,
        D=2,
        F2=None,
        C=8,
        T=2000,
        N=4,
        Nf=64,
        Nf2=16,
        activation="relu",
        **kwargs,
    ):
        super(MIBMINetDeployGroupNorm, self).__init__()

        if F2 is None:
            F2 = F1 * D

        activation = activation.lower()
        assert activation in [
            "elu",
            "relu",
        ], f"activation must be 'elu' or 'relu', got {activation}"

        self.F1, self.D, self.F2, self.C, self.T, self.N = (F1, D, F2, C, T, N)
        self.Nf = Nf
        self.Nf2 = Nf2

        T_after_conv2 = T - Nf + 1
        T_after_pool1 = T_after_conv2 // 8
        T_after_sepconv1 = T_after_pool1 - Nf2 + 1
        n_features = T_after_sepconv1 // 8

        # Block 1: Spectral + Spatial filtering
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(C, 1), bias=False)
        # GroupNorm with num_groups=1: weight shape is [F1], not [F1, 1, T]
        self.layer_norm1 = nn.GroupNorm(num_groups=1, num_channels=F1, eps=0.001)
        self.conv2 = nn.Conv2d(F1, D * F1, kernel_size=(1, Nf), groups=F1, bias=False)
        self.layer_norm2 = nn.GroupNorm(num_groups=1, num_channels=D * F1, eps=0.001)
        self.activation1 = nn.ELU(inplace=False) if activation == "elu" else nn.ReLU(inplace=False)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 8))

        # Block 2: Depthwise separable convolution
        self.sep_conv1 = nn.Conv2d(D * F1, D * F1, kernel_size=(1, Nf2), groups=D * F1, bias=False)
        self.sep_conv2 = nn.Conv2d(D * F1, F2, kernel_size=(1, 1), bias=False)
        self.layer_norm3 = nn.GroupNorm(num_groups=1, num_channels=F2, eps=0.001)
        self.activation2 = nn.ELU(inplace=False) if activation == "elu" else nn.ReLU(inplace=False)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))

        # Classifier
        self.fc = nn.Linear(F2 * n_features, N, bias=True)
        self.fc_input_size = F2 * n_features

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.layer_norm1(x)
        x = self.conv2(x)
        x = self.layer_norm2(x)
        x = self.activation1(x)
        x = self.pool1(x)

        # Block 2
        x = self.sep_conv1(x)
        x = self.sep_conv2(x)
        x = self.layer_norm3(x)
        x = self.activation2(x)
        x = self.pool2(x)

        # Classifier
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.fc_input_size)
        x = self.fc(x)

        return x

    def load_weights_from_layernorm_model(self, layernorm_model):
        """
        Load weights from MIBMINetDeploy (LayerNorm version).
        LayerNorm weights [C, H, W] are converted to GroupNorm weights [C] via mean.
        """
        # Copy conv weights
        self.conv1.weight.data = layernorm_model.conv1.weight.data.clone()
        self.conv2.weight.data = layernorm_model.conv2.weight.data.clone()
        self.sep_conv1.weight.data = layernorm_model.sep_conv1.weight.data.clone()
        self.sep_conv2.weight.data = layernorm_model.sep_conv2.weight.data.clone()
        self.fc.weight.data = layernorm_model.fc.weight.data.clone()
        self.fc.bias.data = layernorm_model.fc.bias.data.clone()

        # Convert LayerNorm weights [C, H, W] -> GroupNorm weights [C]
        # Use mean over H, W dimensions (same as convert_layernorm_to_groupnorm)
        ln1_w = layernorm_model.layer_norm1.weight.data  # [F1, 1, T]
        ln1_b = layernorm_model.layer_norm1.bias.data
        self.layer_norm1.weight.data = ln1_w.reshape(ln1_w.shape[0], -1).mean(dim=1)
        self.layer_norm1.bias.data = ln1_b.reshape(ln1_b.shape[0], -1).mean(dim=1)

        ln2_w = layernorm_model.layer_norm2.weight.data  # [D*F1, 1, T_after_conv2]
        ln2_b = layernorm_model.layer_norm2.bias.data
        self.layer_norm2.weight.data = ln2_w.reshape(ln2_w.shape[0], -1).mean(dim=1)
        self.layer_norm2.bias.data = ln2_b.reshape(ln2_b.shape[0], -1).mean(dim=1)

        ln3_w = layernorm_model.layer_norm3.weight.data  # [F2, 1, T_after_sepconv1]
        ln3_b = layernorm_model.layer_norm3.bias.data
        self.layer_norm3.weight.data = ln3_w.reshape(ln3_w.shape[0], -1).mean(dim=1)
        self.layer_norm3.bias.data = ln3_b.reshape(ln3_b.shape[0], -1).mean(dim=1)

        print("✅ Loaded weights from LayerNorm model to GroupNorm model")
