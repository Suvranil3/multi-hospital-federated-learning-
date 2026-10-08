"""
PyTest Unit Test: CIFAR-10 Data Loading & Integrity (Phase 1).
"""

import sys
from pathlib import Path
import pytest
import torch
from torchvision import datasets, transforms

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import (
    DATA_RAW_DIR,
    SEED,
    CIFAR10_IMAGE_SHAPE,
    CIFAR10_NUM_CLASSES,
    set_seed,
)


@pytest.fixture(autouse=True)
def setup_seed():
    """Ensure deterministic execution for each test."""
    set_seed(SEED)


def test_cifar10_raw_splits_exist():
    """Verify CIFAR-10 can be instantiated for both train and test splits."""
    train_data = datasets.CIFAR10(root=str(DATA_RAW_DIR), train=True, download=True)
    test_data = datasets.CIFAR10(root=str(DATA_RAW_DIR), train=False, download=True)

    assert len(train_data) == 50000, f"Expected 50000 train samples, got {len(train_data)}"
    assert len(test_data) == 10000, f"Expected 10000 test samples, got {len(test_data)}"


def test_cifar10_batch_shape_and_types():
    """Verify PyTorch DataLoader yields correctly shaped tensors and labels."""
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(
        root=str(DATA_RAW_DIR), train=True, download=False, transform=transform
    )
    batch_size = 16
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    images, labels = next(iter(loader))

    # Verify tensor shapes
    expected_image_shape = (batch_size, *CIFAR10_IMAGE_SHAPE)
    assert images.shape == expected_image_shape, (
        f"Expected images shape {expected_image_shape}, got {images.shape}"
    )
    assert labels.shape == (batch_size,), (
        f"Expected labels shape ({batch_size},), got {labels.shape}"
    )

    # Verify data types
    assert images.dtype == torch.float32, f"Expected float32 images, got {images.dtype}"
    assert labels.dtype == torch.int64, f"Expected int64 labels, got {labels.dtype}"

    # Verify range of values: ToTensor normalizes to [0.0, 1.0]
    assert images.min() >= 0.0, f"Image min value {images.min()} < 0.0"
    assert images.max() <= 1.0, f"Image max value {images.max()} > 1.0"

    # Verify label boundaries
    assert labels.min() >= 0
    assert labels.max() < CIFAR10_NUM_CLASSES


def test_cifar10_deterministic_shuffling():
    """Verify that setting seed produces deterministic DataLoader sample orders."""
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(
        root=str(DATA_RAW_DIR), train=True, download=False, transform=transform
    )

    # Run 1
    set_seed(42)
    loader1 = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    batch1_labels = next(iter(loader1))[1].tolist()

    # Run 2
    set_seed(42)
    loader2 = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    batch2_labels = next(iter(loader2))[1].tolist()

    assert batch1_labels == batch2_labels, (
        f"Deterministic shuffling mismatch: {batch1_labels} != {batch2_labels}"
    )
