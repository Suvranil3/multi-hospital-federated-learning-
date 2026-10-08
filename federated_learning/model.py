"""
Small Convolutional Neural Network (SmallCNN) for CIFAR-10.
Designed for Phase 2 Centralized ML Baseline and reusable across
subsequent Federated Learning simulation clients.

Architecture:
- 3 Convolutional blocks with 2D Batch Normalization and Max-Pooling
- Fully connected classification head with Dropout regularization
- Parameter count: ~189k parameters (lightweight, memory-efficient)
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """
    Compact Convolutional Neural Network for CIFAR-10 image classification.

    Input shape: (batch_size, 3, 32, 32)
    Output shape: (batch_size, num_classes)
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3) -> None:
        super(SmallCNN, self).__init__()
        self.num_classes = num_classes

        # Feature extractor: 3 Conv-BN-ReLU-MaxPool blocks
        self.features = nn.Sequential(
            # Block 1: 3x32x32 -> 32x16x16
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: 32x16x16 -> 64x8x8
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3: 64x8x8 -> 64x4x4
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classification head: 64*4*4 (1024) -> 128 -> num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters in the model.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
    }


def get_model(num_classes: int = 10, dropout_rate: float = 0.3) -> SmallCNN:
    """
    Factory function to instantiate SmallCNN with default or custom parameters.
    """
    return SmallCNN(num_classes=num_classes, dropout_rate=dropout_rate)


def sanity_check(batch_size: int = 4) -> Tuple[bool, Tuple[int, ...]]:
    """
    Perform a self-contained forward-pass sanity check with a dummy batch.
    Verifies input tensor shape and output tensor logits shape.
    """
    model = get_model(num_classes=10)
    model.eval()
    dummy_input = torch.randn(batch_size, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)

    expected_shape = (batch_size, 10)
    actual_shape = tuple(output.shape)
    is_valid = actual_shape == expected_shape
    return is_valid, actual_shape


if __name__ == "__main__":
    print("=" * 60)
    print("SmallCNN Architecture & Sanity Verification")
    print("=" * 60)
    model = get_model()
    params = count_parameters(model)
    print(f"Total Parameters:         {params['total']:,}")
    print(f"Trainable Parameters:     {params['trainable']:,}")
    print(f"Non-trainable Parameters: {params['non_trainable']:,}")

    # Forward sanity check
    passed, shape = sanity_check(batch_size=8)
    print(f"Sanity Check Dummy Batch: input=(8, 3, 32, 32) -> output={shape}")
    print(f"Sanity Check Status:      {'PASSED' if passed else 'FAILED'}")
    assert passed, f"Expected output shape (8, 10), got {shape}"
    print("=" * 60)
