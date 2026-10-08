"""
Small Convolutional Neural Network (SmallCNN) for Medical Chest X-Ray Classification.
Migrated for the Primary Medical Project Scenario: Multi-Hospital Federated Learning
with PneumoniaMNIST (Normal vs. Pneumonia binary classification).

Architecture:
- Input: 1 x 28 x 28 (Single Grayscale Channel)
- Conv Block 1: Conv2d(1 -> 32, 3x3), BatchNorm2d(32), ReLU, MaxPool2d(2x2) -> 32 x 14 x 14
- Conv Block 2: Conv2d(32 -> 64, 3x3), BatchNorm2d(64), ReLU, MaxPool2d(2x2) -> 64 x 7 x 7
- Flatten: 64 * 7 * 7 = 3,136 features
- Dense Layer: Linear(3136 -> 128), ReLU, Dropout(0.2)
- Output Head: Linear(128 -> 2) -> 2-class logits (Normal: 0, Pneumonia: 1)

Parameter Count:
- Total parameters: 420,802
- Trainable parameters: 420,802
- Memory footprint: ~1.61 MB (lightweight, highly suitable for 10-hospital FL simulation)
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """
    Modular Convolutional Neural Network for binary chest X-ray classification.
    Designed for centralized baseline and subsequent 10-client Federated Learning.

    Input shape: (batch_size, 1, 28, 28)
    Output shape: (batch_size, num_classes=2)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        dropout_rate: float = 0.2,
    ) -> None:
        super(SmallCNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Feature extractor: 2 Conv-BN-ReLU-MaxPool blocks
        self.features = nn.Sequential(
            # Block 1: 1x28x28 -> 32x14x14
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: 32x14x14 -> 64x7x7
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classification head: 64*7*7 (3136) -> 128 -> num_classes (2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
        Returns:
            Logits of shape (batch_size, num_classes=2)
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


def get_model(
    in_channels: int = 1,
    num_classes: int = 2,
    dropout_rate: float = 0.2,
) -> SmallCNN:
    """
    Factory function to instantiate SmallCNN with default or custom parameters.
    """
    return SmallCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )


def sanity_check(batch_size: int = 4) -> Tuple[bool, Tuple[int, ...]]:
    """
    Perform a self-contained forward-pass sanity check with a dummy batch.
    Verifies input tensor shape and output tensor logits shape.
    """
    model = get_model(in_channels=1, num_classes=2)
    model.eval()
    dummy_input = torch.randn(batch_size, 1, 28, 28)
    with torch.no_grad():
        output = model(dummy_input)

    expected_shape = (batch_size, 2)
    actual_shape = tuple(output.shape)
    is_valid = actual_shape == expected_shape
    return is_valid, actual_shape


if __name__ == "__main__":
    print("=" * 65)
    print("SmallCNN Architecture & Sanity Verification (PneumoniaMNIST)")
    print("=" * 65)
    model = get_model()
    params = count_parameters(model)
    print(f"Input Shape:              (batch_size, 1, 28, 28)")
    print(f"Output Shape:             (batch_size, 2)")
    print(f"Total Parameters:         {params['total']:,}")
    print(f"Trainable Parameters:     {params['trainable']:,}")
    print(f"Non-trainable Parameters: {params['non_trainable']:,}")

    # Forward sanity check
    passed, shape = sanity_check(batch_size=8)
    print(f"Sanity Check Dummy Batch: input=(8, 1, 28, 28) -> output={shape}")
    print(f"Sanity Check Status:      {'PASSED' if passed else 'FAILED'}")
    assert passed, f"Expected output shape (8, 2), got {shape}"
    print("=" * 65)
