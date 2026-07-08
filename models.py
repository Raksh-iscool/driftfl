"""
Model Architectures for DriftFL Experiments.

Two models are used:
1. ResNet-18 for CIFAR-100 (image classification)
2. 3-layer MLP for synthetic regression

Both models expose a method to extract penultimate-layer features,
which is what DriftFL monitors for drift.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class CIFARResNet18(nn.Module):
    """
    ResNet-18 adapted for CIFAR-100.

    Standard ResNet-18 is designed for 224x224 ImageNet images.
    CIFAR-100 images are 32x32, so we modify the first conv layer
    to use a 3x3 kernel instead of 7x7, and remove the max pool.

    The penultimate layer produces 512-dimensional feature vectors,
    which is what DriftFL monitors.
    """

    def __init__(self, num_classes=100):
        super().__init__()
        # Load standard ResNet-18 structure
        base = resnet18(weights=None)

        # Replace first conv: 7x7 stride 2 -> 3x3 stride 1 (for 32x32 input)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        # Skip maxpool (CIFAR images are already small)

        self.layer1 = base.layer1  # 64 channels
        self.layer2 = base.layer2  # 128 channels
        self.layer3 = base.layer3  # 256 channels
        self.layer4 = base.layer4  # 512 channels

        self.avgpool = base.avgpool  # Global average pool -> 512-dim
        self.fc = nn.Linear(512, num_classes)

        # Feature dimension for DriftFL
        self.feature_dim = 512

    def extract_features(self, x):
        """
        Forward pass up to the penultimate layer.

        Returns 512-dimensional feature vectors BEFORE the final
        classification layer. These are what DriftFL monitors.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, 3, 32, 32)

        Returns
        -------
        torch.Tensor of shape (batch, 512)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (batch, 512)
        return x

    def forward(self, x):
        features = self.extract_features(x)
        return self.fc(features)


class RegressionMLP(nn.Module):
    """
    3-layer MLP for the synthetic regression task.

    Architecture: 20 -> 256 -> 128 -> 64 -> 1
    The 64-dimensional layer before the final output is the
    penultimate layer whose features DriftFL monitors.
    """

    def __init__(self, input_dim=20, hidden_dims=(256, 128, 64)):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc_out = nn.Linear(hidden_dims[2], 1)
        self.feature_dim = hidden_dims[2]  # 64

    def extract_features(self, x):
        """Extract features from the penultimate layer (64-dim)."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))  # (batch, 64)
        return x

    def forward(self, x):
        features = self.extract_features(x)
        return self.fc_out(features)


def get_model(task='cifar100', **kwargs):
    """
    Factory function to create models.

    Parameters
    ----------
    task : str
        'cifar100' for ResNet-18, 'synthetic' for MLP.

    Returns
    -------
    nn.Module
        The model with .feature_dim and .extract_features() available.
    """
    if task == 'cifar100':
        return CIFARResNet18(num_classes=kwargs.get('num_classes', 100))
    elif task == 'synthetic':
        return RegressionMLP(
            input_dim=kwargs.get('input_dim', 20),
            hidden_dims=kwargs.get('hidden_dims', (256, 128, 64))
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def count_parameters(model):
    """Count total trainable parameters and model size in bytes."""
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_params, total_bytes


if __name__ == "__main__":
    print("=== Model Architecture Tests ===")

    # Test ResNet-18 for CIFAR
    model_cifar = get_model('cifar100')
    x = torch.randn(4, 3, 32, 32)
    features = model_cifar.extract_features(x)
    output = model_cifar(x)
    params, size = count_parameters(model_cifar)
    print(f"CIFAR ResNet-18:")
    print(f"  Feature dim: {model_cifar.feature_dim}")
    print(f"  Feature shape: {features.shape}")  # (4, 512)
    print(f"  Output shape: {output.shape}")      # (4, 100)
    print(f"  Parameters: {params:,}")
    print(f"  Model size: {size / 1e6:.2f} MB")

    # Test MLP for synthetic
    model_synth = get_model('synthetic', input_dim=20)
    x = torch.randn(4, 20)
    features = model_synth.extract_features(x)
    output = model_synth(x)
    params, size = count_parameters(model_synth)
    print(f"\nSynthetic MLP:")
    print(f"  Feature dim: {model_synth.feature_dim}")
    print(f"  Feature shape: {features.shape}")  # (4, 64)
    print(f"  Output shape: {output.shape}")      # (4, 1)
    print(f"  Parameters: {params:,}")
    print(f"  Model size: {size / 1e6:.2f} MB")
