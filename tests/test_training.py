"""
Unit Tests for Centralized Data Preparation & Training Logic (Phase 2).
Verifies:
1. Stratified split produces 40,000 train samples (4,000/class) and 10,000 validation samples (1,000/class).
2. Untouched test split maintains exactly 10,000 samples.
3. DataLoader batch yielding, normalization range, and target types.
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import DATA_RAW_DIR, SEED, set_seed
from experiments.train_centralized import prepare_data_loaders


@pytest.fixture(autouse=True)
def setup_seed():
    """Deterministic seed for tests."""
    set_seed(SEED)


def test_prepare_data_loaders_splits():
    """Verify train, val, and test splits match exact requirements."""
    train_loader, val_loader, test_loader, split_info = prepare_data_loaders(
        data_dir=DATA_RAW_DIR,
        val_ratio=0.2,
        batch_size=32,
        seed=SEED,
        num_workers=0,  # Single-process for unit testing speed
    )

    assert split_info["train_samples"] == 40000, f"Expected 40,000 train samples, got {split_info['train_samples']}"
    assert split_info["val_samples"] == 10000, f"Expected 10,000 val samples, got {split_info['val_samples']}"
    assert split_info["test_samples"] == 10000, f"Expected 10,000 test samples, got {split_info['test_samples']}"
    assert split_info["train_samples_per_class"] == 4000
    assert split_info["val_samples_per_class"] == 1000

    # Total training dataset size was 50,000
    assert split_info["train_samples"] + split_info["val_samples"] == 50000


def test_prepare_data_loaders_batch_shapes():
    """Verify batch yield shapes and tensor dtypes from loaders."""
    train_loader, val_loader, test_loader, _ = prepare_data_loaders(
        data_dir=DATA_RAW_DIR,
        val_ratio=0.2,
        batch_size=16,
        seed=SEED,
        num_workers=0,
    )

    train_imgs, train_labels = next(iter(train_loader))
    assert train_imgs.shape == (16, 3, 32, 32)
    assert train_labels.shape == (16,)
    assert train_imgs.dtype == torch.float32
    assert train_labels.dtype == torch.int64

    val_imgs, val_labels = next(iter(val_loader))
    assert val_imgs.shape == (16, 3, 32, 32)
    assert val_labels.shape == (16,)
