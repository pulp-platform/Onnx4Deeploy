import torch
import torch.nn as nn


class MIBMINet(nn.Module):
    """
    MI-BMINet: A simple neural network model for demonstration.
    This is a placeholder implementation. Replace with your actual MI-BMInet architecture.
    """
    def __init__(
        self,
        img_size=32,
        num_classes=10,
        embedding_dim=128,
        num_heads=1,
        num_layers=2,
        in_channels=3,
    ):
        super(MIBMINet, self).__init__()

        self.img_size = img_size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Simple CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim // 2, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def mi_bminet_small(pretrained=False, img_size=32, num_classes=10, embedding_dim=128, **kwargs):
    """
    Create a small MI-BMInet model
    """
    model = MIBMINet(
        img_size=img_size,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        **kwargs
    )
    return model


def mi_bminet_base(pretrained=False, img_size=32, num_classes=10, embedding_dim=256, **kwargs):
    """
    Create a base MI-BMInet model
    """
    model = MIBMINet(
        img_size=img_size,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        **kwargs
    )
    return model
