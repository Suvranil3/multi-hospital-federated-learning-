"""
Unit Tests for SmallCNN Model (Phase MEDICAL-02 Centralized Baseline).
Primary Task: 1x28x28 Grayscale Chest X-Rays -> 2-Class Logits (Normal vs Pneumonia).
Verifies:
1. Model construction and parameter configuration
2. Forward pass output dimensions [batch_size, 2] and value validity
3. Loss calculation with CrossEntropyLoss
4. Single backward pass and parameter update (one small training step)
"""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import SEED, set_seed, IMAGE_SHAPE, NUM_CLASSES
from federated_learning.model import SmallCNN, get_model, count_parameters


@pytest.fixture(autouse=True)
def setup_seed():
    """Enforce deterministic seed for each test."""
    set_seed(SEED)


def test_cnn_construction():
    """Verify SmallCNN initializes correctly with expected parameter counts."""
    model = get_model(in_channels=1, num_classes=NUM_CLASSES)
    assert isinstance(model, SmallCNN)
    assert model.in_channels == 1
    assert model.num_classes == NUM_CLASSES

    params = count_parameters(model)
    assert params["total"] == 420_802, f"Expected 420,802 parameters, got {params['total']}"
    assert params["trainable"] == params["total"]
    assert params["non_trainable"] == 0


@pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
def test_cnn_output_shape(batch_size):
    """Verify output shape is strictly [batch_size, 2] across varying batch sizes."""
    model = get_model(in_channels=1, num_classes=2)
    model.eval()
    dummy_input = torch.randn(batch_size, *IMAGE_SHAPE)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (batch_size, NUM_CLASSES), (
        f"Expected output shape ({batch_size}, {NUM_CLASSES}), got {output.shape}"
    )


def test_cnn_forward_pass_numerical_validity():
    """Verify forward pass does not produce NaNs or Inf values."""
    model = get_model(in_channels=1, num_classes=2)
    dummy_input = torch.randn(8, *IMAGE_SHAPE)
    output = model(dummy_input)

    assert not torch.isnan(output).any(), "Forward pass returned NaN"
    assert not torch.isinf(output).any(), "Forward pass returned Inf"


def test_cnn_loss_calculation():
    """Verify CrossEntropyLoss computation succeeds and outputs scalar float tensor."""
    model = get_model(in_channels=1, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    dummy_input = torch.randn(8, *IMAGE_SHAPE)
    dummy_targets = torch.randint(0, NUM_CLASSES, (8,), dtype=torch.int64)

    outputs = model(dummy_input)
    loss = criterion(outputs, dummy_targets)

    assert loss.dim() == 0, f"Loss must be scalar, got dim {loss.dim()}"
    assert not torch.isnan(loss), "Loss computed as NaN"
    assert loss.item() > 0.0, "Loss must be positive"


def test_cnn_one_training_step():
    """Verify backpropagation computes gradients and optimizer updates model weights."""
    model = get_model(in_channels=1, num_classes=2)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    dummy_input = torch.randn(4, *IMAGE_SHAPE)
    dummy_targets = torch.randint(0, NUM_CLASSES, (4,), dtype=torch.int64)

    # Capture initial weights of first convolutional layer
    initial_weights = model.features[0].weight.clone()

    # Step 1: Forward pass
    optimizer.zero_grad()
    outputs = model(dummy_input)
    loss = criterion(outputs, dummy_targets)

    # Step 2: Backward pass
    loss.backward()

    # Verify gradients exist and are non-zero
    assert model.features[0].weight.grad is not None
    assert torch.any(model.features[0].weight.grad != 0.0)

    # Step 3: Optimizer step
    optimizer.step()

    # Verify weights have changed
    updated_weights = model.features[0].weight
    assert not torch.equal(initial_weights, updated_weights), "Weights did not update after step"
