"""
PyTest Unit Tests: Medical Dataset Loading & Integrity (Phase MEDICAL-01).
Primary Dataset: PneumoniaMNIST (MedMNIST v3).
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np
from torchvision import transforms
from medmnist import PneumoniaMNIST

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import (
    DATA_RAW_DIR,
    SEED,
    IMAGE_SHAPE,
    NUM_CLASSES,
    TRAIN_SAMPLES,
    VAL_SAMPLES,
    TEST_SAMPLES,
    set_seed,
)


@pytest.fixture(autouse=True)
def setup_seed():
    """Ensure deterministic execution for each test."""
    set_seed(SEED)


def test_pneumoniamnist_official_splits_exist():
    """Verify PneumoniaMNIST can be instantiated for train, val, and test splits."""
    train_data = PneumoniaMNIST(split="train", root=str(DATA_RAW_DIR), download=True)
    val_data = PneumoniaMNIST(split="val", root=str(DATA_RAW_DIR), download=True)
    test_data = PneumoniaMNIST(split="test", root=str(DATA_RAW_DIR), download=True)

    assert len(train_data) == TRAIN_SAMPLES, f"Expected {TRAIN_SAMPLES} train, got {len(train_data)}"
    assert len(val_data) == VAL_SAMPLES, f"Expected {VAL_SAMPLES} val, got {len(val_data)}"
    assert len(test_data) == TEST_SAMPLES, f"Expected {TEST_SAMPLES} test, got {len(test_data)}"


def test_pneumoniamnist_batch_shape_and_types():
    """Verify PyTorch DataLoader yields correctly shaped 1x28x28 tensors and binary labels."""
    transform = transforms.ToTensor()
    dataset = PneumoniaMNIST(
        split="train", root=str(DATA_RAW_DIR), download=False, transform=transform
    )
    batch_size = 16
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    images, labels = next(iter(loader))

    # Verify tensor shapes: (batch_size, 1, 28, 28)
    expected_image_shape = (batch_size, *IMAGE_SHAPE)
    assert images.shape == expected_image_shape, (
        f"Expected images shape {expected_image_shape}, got {images.shape}"
    )

    # MedMNIST labels come as (batch_size, 1) -> squeezed to (batch_size,)
    squeezed_labels = labels.squeeze(-1)
    assert squeezed_labels.shape == (batch_size,), (
        f"Expected labels shape ({batch_size},), got {squeezed_labels.shape}"
    )

    # Verify data types
    assert images.dtype == torch.float32, f"Expected float32 images, got {images.dtype}"

    # Verify pixel value range [0.0, 1.0]
    assert images.min() >= 0.0, f"Image min value {images.min()} < 0.0"
    assert images.max() <= 1.0, f"Image max value {images.max()} > 1.0"

    # Verify binary label boundaries (0 = Normal, 1 = Pneumonia)
    assert squeezed_labels.min() >= 0
    assert squeezed_labels.max() < NUM_CLASSES
    unique_vals = torch.unique(squeezed_labels).tolist()
    for val in unique_vals:
        assert val in [0, 1], f"Unexpected label {val}"


def test_pneumoniamnist_deterministic_shuffling():
    """Verify that setting seed produces deterministic DataLoader sample orders."""
    transform = transforms.ToTensor()
    dataset = PneumoniaMNIST(
        split="train", root=str(DATA_RAW_DIR), download=False, transform=transform
    )

    # Run 1
    set_seed(42)
    loader1 = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    batch1_labels = next(iter(loader1))[1].squeeze(-1).tolist()

    # Run 2
    set_seed(42)
    loader2 = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    batch2_labels = next(iter(loader2))[1].squeeze(-1).tolist()

    assert batch1_labels == batch2_labels, (
        f"Deterministic shuffling mismatch: {batch1_labels} != {batch2_labels}"
    )
