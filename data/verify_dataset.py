"""
CIFAR-10 Dataset Verification Pipeline (Phase 1).

Verifies:
1. Dataset download into data/raw/
2. Sample counts (50,000 train, 10,000 test)
3. Image dimensions (3x32x32) and RGB channels
4. 10 classes and exact class balance (5,000/class train, 1,000/class test)
5. Valid label range [0..9]
6. Channel mean and standard deviation calculation
7. Generation of verification figures and JSON log report
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from experiments.config import (
    DATA_RAW_DIR,
    RESULTS_FIGURES_DIR,
    RESULTS_LOGS_DIR,
    SEED,
    CIFAR10_NUM_CLASSES,
    CIFAR10_TRAIN_SAMPLES,
    CIFAR10_TEST_SAMPLES,
    CIFAR10_TRAIN_SAMPLES_PER_CLASS,
    CIFAR10_TEST_SAMPLES_PER_CLASS,
    CIFAR10_IMAGE_SHAPE,
    CIFAR10_CLASSES,
    set_seed,
)


def verify_and_analyze_cifar10():
    print("=" * 70)
    print("PHASE 1: CIFAR-10 DATASET VERIFICATION & STATISTICAL ANALYSIS")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Target Raw Directory: {DATA_RAW_DIR}")
    print(f"Deterministic Seed: {SEED}")
    print("=" * 70)

    # 1. Enforce Deterministic Seeding
    set_seed(SEED)

    # 2. Download and Load Raw Datasets
    print("\n[Step 1/6] Loading CIFAR-10 datasets...")
    train_dataset_raw = datasets.CIFAR10(
        root=str(DATA_RAW_DIR), train=True, download=True
    )
    test_dataset_raw = datasets.CIFAR10(
        root=str(DATA_RAW_DIR), train=False, download=True
    )

    train_len = len(train_dataset_raw)
    test_len = len(test_dataset_raw)

    print(f"  Training samples loaded: {train_len}")
    print(f"  Test samples loaded:     {test_len}")

    # 3. Assert Split Sizes
    assert train_len == CIFAR10_TRAIN_SAMPLES, (
        f"Expected {CIFAR10_TRAIN_SAMPLES} training samples, found {train_len}"
    )
    assert test_len == CIFAR10_TEST_SAMPLES, (
        f"Expected {CIFAR10_TEST_SAMPLES} test samples, found {test_len}"
    )
    print("  ✓ Split sizes verified.")

    # 4. Check Image Dimensions & Data Types
    print("\n[Step 2/6] Verifying image dimensions and classes...")
    sample_img, sample_label = train_dataset_raw[0]
    raw_size = sample_img.size  # (W, H) in PIL
    raw_channels = len(sample_img.getbands())

    print(f"  Raw PIL image size: {raw_size} (Width, Height)")
    print(f"  Channels: {raw_channels} ({sample_img.getbands()})")
    assert raw_size == (32, 32), f"Expected (32, 32), got {raw_size}"
    assert raw_channels == 3, f"Expected 3 channels (RGB), got {raw_channels}"

    # Also verify with ToTensor
    to_tensor = transforms.ToTensor()
    tensor_sample = to_tensor(sample_img)
    tensor_shape = tuple(tensor_sample.shape)  # (C, H, W)
    print(f"  Tensor shape (C, H, W): {tensor_shape}")
    assert tensor_shape == CIFAR10_IMAGE_SHAPE, (
        f"Expected shape {CIFAR10_IMAGE_SHAPE}, got {tensor_shape}"
    )
    print("  ✓ Image dimensions and channels verified.")

    # 5. Verify Label Range & Class Distribution
    print("\n[Step 3/6] Inspecting class distributions...")
    train_targets = np.array(train_dataset_raw.targets)
    test_targets = np.array(test_dataset_raw.targets)

    unique_train_labels, train_counts = np.unique(train_targets, return_counts=True)
    unique_test_labels, test_counts = np.unique(test_targets, return_counts=True)

    print(f"  Discovered classes in train: {len(unique_train_labels)}")
    print(f"  Discovered classes in test:  {len(unique_test_labels)}")
    assert len(unique_train_labels) == CIFAR10_NUM_CLASSES
    assert len(unique_test_labels) == CIFAR10_NUM_CLASSES
    assert set(unique_train_labels) == set(range(CIFAR10_NUM_CLASSES))
    assert set(unique_test_labels) == set(range(CIFAR10_NUM_CLASSES))

    class_counts_train = {
        CIFAR10_CLASSES[i]: int(train_counts[i]) for i in range(CIFAR10_NUM_CLASSES)
    }
    class_counts_test = {
        CIFAR10_CLASSES[i]: int(test_counts[i]) for i in range(CIFAR10_NUM_CLASSES)
    }

    for cls_name, count in class_counts_train.items():
        assert count == CIFAR10_TRAIN_SAMPLES_PER_CLASS, (
            f"Class {cls_name} train count {count} != {CIFAR10_TRAIN_SAMPLES_PER_CLASS}"
        )
    for cls_name, count in class_counts_test.items():
        assert count == CIFAR10_TEST_SAMPLES_PER_CLASS, (
            f"Class {cls_name} test count {count} != {CIFAR10_TEST_SAMPLES_PER_CLASS}"
        )
    print("  ✓ Exact class balance confirmed (5,000 train/class, 1,000 test/class).")

    # 6. Calculate Channel Means and Standard Deviations
    print("\n[Step 4/6] Computing empirical RGB channel mean and std...")
    # Load all training images as normalized float32 tensors [0, 1]
    train_tensor_dataset = datasets.CIFAR10(
        root=str(DATA_RAW_DIR), train=True, download=False, transform=transforms.ToTensor()
    )
    loader = torch.utils.data.DataLoader(
        train_tensor_dataset, batch_size=5000, shuffle=False, num_workers=2
    )

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_squared_sum = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0

    for data, _ in loader:
        # data shape: (N, 3, 32, 32)
        channel_sum += torch.sum(data.to(torch.float64), dim=[0, 2, 3])
        channel_squared_sum += torch.sum((data.to(torch.float64)) ** 2, dim=[0, 2, 3])
        total_pixels += data.size(0) * data.size(2) * data.size(3)

    mean = channel_sum / total_pixels
    std = torch.sqrt((channel_squared_sum / total_pixels) - (mean ** 2))

    channel_mean = tuple(float(m) for m in mean)
    channel_std = tuple(float(s) for s in std)

    print(f"  Calculated Channel Means (R, G, B): {[round(x, 4) for x in channel_mean]}")
    print(f"  Calculated Channel Stds  (R, G, B): {[round(x, 4) for x in channel_std]}")
    print("  ✓ Channel statistics computed successfully.")

    # 7. Generate Visualizations
    print("\n[Step 5/6] Generating dataset visualizations...")

    # Figure 1: Sample Grid (10 classes x 5 samples)
    grid_fig_path = RESULTS_FIGURES_DIR / "cifar10_sample_grid.png"
    fig, axes = plt.subplots(10, 5, figsize=(10, 18))
    fig.suptitle("CIFAR-10: Class Representatives (5 Samples per Class)", fontsize=16, y=0.995)

    # Find 5 sample indices per class deterministically
    np_rng = np.random.default_state() if hasattr(np.random, 'default_state') else None
    rng = np.random.RandomState(SEED)

    for class_idx in range(CIFAR10_NUM_CLASSES):
        indices = np.where(train_targets == class_idx)[0]
        selected_indices = rng.choice(indices, size=5, replace=False)
        for col_idx, img_idx in enumerate(selected_indices):
            img, label = train_dataset_raw[img_idx]
            ax = axes[class_idx, col_idx]
            ax.imshow(img)
            ax.axis("off")
            if col_idx == 0:
                ax.set_title(
                    f"{CIFAR10_CLASSES[class_idx]} ({class_idx})",
                    loc="right",
                    fontsize=11,
                    fontweight="bold",
                    pad=10,
                )

    plt.tight_layout()
    plt.savefig(grid_fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved sample grid figure: {grid_fig_path}")

    # Figure 2: Class Distribution Bar Chart
    dist_fig_path = RESULTS_FIGURES_DIR / "cifar10_class_distribution.png"
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(CIFAR10_NUM_CLASSES)
    width = 0.38

    rects1 = ax.bar(
        x - width / 2,
        [class_counts_train[c] for c in CIFAR10_CLASSES],
        width,
        label=f"Train Set (N={train_len})",
        color="#2b5c8f",
        edgecolor="#16314f",
    )
    rects2 = ax.bar(
        x + width / 2,
        [class_counts_test[c] for c in CIFAR10_CLASSES],
        width,
        label=f"Test Set (N={test_len})",
        color="#e07a5f",
        edgecolor="#964028",
    )

    ax.set_ylabel("Number of Samples", fontsize=12, fontweight="bold")
    ax.set_title("CIFAR-10 Class Distribution (Train vs Test Splits)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=25, ha="right", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Value labels on top of bars
    ax.bar_label(rects1, padding=3, fontsize=9)
    ax.bar_label(rects2, padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(dist_fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved class distribution figure: {dist_fig_path}")

    # 8. Produce Machine-Readable JSON Report
    print("\n[Step 6/6] Generating verification report...")
    report_path = RESULTS_LOGS_DIR / "cifar10_verification_report.json"
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "experiment_id": "EXP-000-DATA",
        "status": "PASSED",
        "random_seed": SEED,
        "dataset_name": "CIFAR-10",
        "total_samples": train_len + test_len,
        "train_samples": train_len,
        "test_samples": test_len,
        "num_classes": CIFAR10_NUM_CLASSES,
        "classes": list(CIFAR10_CLASSES),
        "raw_image_resolution": [32, 32],
        "channels": 3,
        "tensor_shape": list(CIFAR10_IMAGE_SHAPE),
        "train_class_distribution": class_counts_train,
        "test_class_distribution": class_counts_test,
        "is_perfectly_balanced": True,
        "channel_mean_rgb": [float(round(m, 5)) for m in channel_mean],
        "channel_std_rgb": [float(round(s, 5)) for s in channel_std],
        "artifacts_generated": {
            "sample_grid_figure": str(grid_fig_path.relative_to(PROJECT_ROOT)),
            "class_distribution_figure": str(dist_fig_path.relative_to(PROJECT_ROOT)),
            "report_file": str(report_path.relative_to(PROJECT_ROOT)),
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"  ✓ Saved verification JSON report: {report_path}")
    print("\n" + "=" * 70)
    print("PHASE 1 VERIFICATION COMPLETE: ALL CHECKS PASSED.")
    print("=" * 70)
    return report_data


if __name__ == "__main__":
    verify_and_analyze_cifar10()
