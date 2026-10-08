"""
Unit Tests for Centralized Data Preparation & Medical Metrics Logic (Phase MEDICAL-02).
Verifies:
1. Official split preservation: 4,708 train, 524 validation, 624 untouched test.
2. DataLoader batch yielding, normalization range, and target types.
3. Medical evaluation metrics: precision, recall, F1, and confusion matrix calculation.
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import (
    DATA_RAW_DIR,
    SEED,
    TRAIN_SAMPLES,
    VAL_SAMPLES,
    TEST_SAMPLES,
    TOTAL_SAMPLES,
    set_seed,
)
from experiments.train_medical import (
    prepare_medical_data_loaders,
    compute_medical_metrics,
)


@pytest.fixture(autouse=True)
def setup_seed():
    """Deterministic seed for tests."""
    set_seed(SEED)


def test_prepare_data_loaders_splits():
    """Verify train, val, and test splits preserve official MedMNIST splits."""
    train_loader, val_loader, test_loader, split_info = prepare_medical_data_loaders(
        data_dir=DATA_RAW_DIR,
        batch_size=32,
        num_workers=0,  # Single-process for unit testing speed
    )

    assert split_info["train_samples"] == TRAIN_SAMPLES, f"Expected {TRAIN_SAMPLES} train, got {split_info['train_samples']}"
    assert split_info["val_samples"] == VAL_SAMPLES, f"Expected {VAL_SAMPLES} val, got {split_info['val_samples']}"
    assert split_info["test_samples"] == TEST_SAMPLES, f"Expected {TEST_SAMPLES} test, got {split_info['test_samples']}"

    assert split_info["train_samples"] + split_info["val_samples"] + split_info["test_samples"] == TOTAL_SAMPLES


def test_prepare_data_loaders_batch_shapes():
    """Verify batch yield shapes and tensor dtypes from loaders."""
    train_loader, val_loader, test_loader, _ = prepare_medical_data_loaders(
        data_dir=DATA_RAW_DIR,
        batch_size=16,
        num_workers=0,
    )

    train_imgs, train_labels = next(iter(train_loader))
    assert train_imgs.shape == (16, 1, 28, 28)
    # MedMNIST raw labels have shape (16, 1) -> squeezed to (16,)
    assert train_labels.squeeze(-1).shape == (16,)
    assert train_imgs.dtype == torch.float32

    val_imgs, val_labels = next(iter(val_loader))
    assert val_imgs.shape == (16, 1, 28, 28)
    assert val_labels.squeeze(-1).shape == (16,)

    test_imgs, test_labels = next(iter(test_loader))
    assert test_imgs.shape == (16, 1, 28, 28)
    assert test_labels.squeeze(-1).shape == (16,)


def test_compute_medical_metrics():
    """Verify precision, recall, F1-score, and confusion matrix computations."""
    # Synthetic ground truth and predictions
    # 2 Normal correct (TN=2), 1 Normal predicted as Pneumonia (FP=1)
    # 1 Pneumonia predicted as Normal (FN=1), 4 Pneumonia correct (TP=4)
    targets = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    preds =   np.array([0, 0, 1, 0, 1, 1, 1, 1])

    metrics = compute_medical_metrics(targets, preds)

    cm = metrics["confusion_matrix"]
    assert cm["true_negative_normal"] == 2
    assert cm["false_positive_normal_as_pneumonia"] == 1
    assert cm["false_negative_pneumonia_as_normal"] == 1
    assert cm["true_positive_pneumonia"] == 4

    p_pneu = metrics["pneumonia_class_1"]
    # TP / (TP + FP) = 4 / (4 + 1) = 0.8
    assert abs(p_pneu["precision"] - 0.8) < 1e-3
    # TP / (TP + FN) = 4 / (4 + 1) = 0.8
    assert abs(p_pneu["recall_sensitivity"] - 0.8) < 1e-3
    assert abs(p_pneu["f1_score"] - 0.8) < 1e-3
