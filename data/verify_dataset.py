"""
Medical Dataset Verification Pipeline (Phase MEDICAL-01).
Primary Dataset: PneumoniaMNIST (MedMNIST v3).

Verifies:
1. Automatic download and integrity of PneumoniaMNIST in data/raw/
2. Split sizes: Train = 4,708, Validation = 524, Test = 624 (Total = 5,856)
3. Image dimensions: 28x28 single-channel grayscale (1x28x28)
4. Binary classification labels: 0 = 'normal', 1 = 'pneumonia'
5. Exact class distributions across train, val, and test splits
6. Pixel statistics: min, max, empirical grayscale mean and standard deviation
7. Generates visual figures:
   - results/figures/pneumonia_sample_grid.png
   - results/figures/pneumonia_class_distribution.png
8. Emits structured report:
   - results/logs/pneumonia_mnist_verification_report.json
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
from torchvision import transforms
from medmnist import PneumoniaMNIST

from experiments.config import (
    DATA_RAW_DIR,
    RESULTS_FIGURES_DIR,
    RESULTS_LOGS_DIR,
    SEED,
    DATASET_NAME,
    NUM_CLASSES,
    CLASSES,
    CLASS_LABELS,
    IMAGE_SHAPE,
    TRAIN_SAMPLES,
    VAL_SAMPLES,
    TEST_SAMPLES,
    TOTAL_SAMPLES,
    TRAIN_CLASS_COUNTS,
    VAL_CLASS_COUNTS,
    TEST_CLASS_COUNTS,
    set_seed,
)


def verify_and_analyze_pneumoniamnist():
    print("=" * 75)
    print("PHASE MEDICAL-01: PNEUMONIAMNIST DATASET VERIFICATION & STATISTICAL ANALYSIS")
    print(f"Timestamp:            {datetime.now().isoformat()}")
    print(f"Target Raw Directory: {DATA_RAW_DIR}")
    print(f"Deterministic Seed:   {SEED}")
    print("=" * 75)

    # 1. Enforce Deterministic Seeding
    set_seed(SEED)

    # 2. Download and Load Official MedMNIST Splits
    print("\n[Step 1/6] Loading official PneumoniaMNIST dataset splits...")
    train_dataset = PneumoniaMNIST(split="train", root=str(DATA_RAW_DIR), download=True)
    val_dataset = PneumoniaMNIST(split="val", root=str(DATA_RAW_DIR), download=True)
    test_dataset = PneumoniaMNIST(split="test", root=str(DATA_RAW_DIR), download=True)

    train_len = len(train_dataset)
    val_len = len(val_dataset)
    test_len = len(test_dataset)
    total_len = train_len + val_len + test_len

    print(f"  Training samples loaded:   {train_len}")
    print(f"  Validation samples loaded: {val_len}")
    print(f"  Test samples loaded:       {test_len}")
    print(f"  Total samples:             {total_len}")

    # 3. Assert Split Sizes
    assert train_len == TRAIN_SAMPLES, f"Expected {TRAIN_SAMPLES} train, found {train_len}"
    assert val_len == VAL_SAMPLES, f"Expected {VAL_SAMPLES} val, found {val_len}"
    assert test_len == TEST_SAMPLES, f"Expected {TEST_SAMPLES} test, found {test_len}"
    assert total_len == TOTAL_SAMPLES, f"Expected {TOTAL_SAMPLES} total, found {total_len}"
    print("  ✓ Official split sizes verified.")

    # 4. Check Image Dimensions & Channels
    print("\n[Step 2/6] Verifying image dimensions, grayscale channels, and data types...")
    sample_pil, sample_label = train_dataset[0]
    raw_size = sample_pil.size  # (W, H)
    raw_mode = sample_pil.mode
    raw_bands = sample_pil.getbands()

    print(f"  PIL image resolution: {raw_size} (Width x Height)")
    print(f"  Image mode:           {raw_mode} (Bands: {raw_bands})")
    assert raw_size == (28, 28), f"Expected (28, 28), got {raw_size}"
    assert len(raw_bands) == 1, f"Expected 1 grayscale channel, got {len(raw_bands)}"

    # Check tensor shape with ToTensor
    to_tensor = transforms.ToTensor()
    tensor_sample = to_tensor(sample_pil)
    tensor_shape = tuple(tensor_sample.shape)
    print(f"  Tensor shape (C, H, W): {tensor_shape}")
    assert tensor_shape == IMAGE_SHAPE, f"Expected shape {IMAGE_SHAPE}, got {tensor_shape}"
    print("  ✓ Grayscale image shape 1x28x28 verified.")

    # 5. Verify Label Range & Class Distributions
    print("\n[Step 3/6] Inspecting label values and class distributions...")
    train_labels = train_dataset.labels.flatten()
    val_labels = val_dataset.labels.flatten()
    test_labels = test_dataset.labels.flatten()

    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    unique_val, counts_val = np.unique(val_labels, return_counts=True)
    unique_test, counts_test = np.unique(test_labels, return_counts=True)

    print(f"  Discovered classes in train: {unique_train.tolist()}")
    print(f"  Discovered classes in val:   {unique_val.tolist()}")
    print(f"  Discovered classes in test:  {unique_test.tolist()}")

    assert set(unique_train.tolist()) == {0, 1}, "Train labels must be exactly {0, 1}"
    assert set(unique_val.tolist()) == {0, 1}, "Val labels must be exactly {0, 1}"
    assert set(unique_test.tolist()) == {0, 1}, "Test labels must be exactly {0, 1}"

    train_dist = {CLASS_LABELS[int(k)]: int(v) for k, v in zip(unique_train, counts_train)}
    val_dist = {CLASS_LABELS[int(k)]: int(v) for k, v in zip(unique_val, counts_val)}
    test_dist = {CLASS_LABELS[int(k)]: int(v) for k, v in zip(unique_test, counts_test)}

    print(f"  Train split counts: {train_dist}")
    print(f"  Val split counts:   {val_dist}")
    print(f"  Test split counts:  {test_dist}")

    assert train_dist["normal"] == TRAIN_CLASS_COUNTS[0] and train_dist["pneumonia"] == TRAIN_CLASS_COUNTS[1]
    assert val_dist["normal"] == VAL_CLASS_COUNTS[0] and val_dist["pneumonia"] == VAL_CLASS_COUNTS[1]
    assert test_dist["normal"] == TEST_CLASS_COUNTS[0] and test_dist["pneumonia"] == TEST_CLASS_COUNTS[1]
    print("  ✓ Exact class counts confirmed across all official splits.")

    # 6. Calculate Empirical Pixel Statistics
    print("\n[Step 4/6] Computing empirical pixel statistics (min, max, mean, std)...")
    # Raw uint8 images array from underlying npz dataset
    raw_images = train_dataset.imgs  # (4708, 28, 28) uint8
    min_pixel_val = int(raw_images.min())
    max_pixel_val = int(raw_images.max())

    norm_images = raw_images.astype(np.float64) / 255.0
    grayscale_mean = float(norm_images.mean())
    grayscale_std = float(norm_images.std())

    print(f"  Raw Pixel Minimum:    {min_pixel_val}")
    print(f"  Raw Pixel Maximum:    {max_pixel_val}")
    print(f"  Empirical Mean [0,1]: {grayscale_mean:.5f}")
    print(f"  Empirical Std  [0,1]: {grayscale_std:.5f}")
    print("  ✓ Pixel statistics computed successfully.")

    # 7. Generate Visual Artifacts
    print("\n[Step 5/6] Generating medical dataset visualizations...")

    # Figure 1: Sample Grid of Chest X-Rays (Normal vs Pneumonia)
    sample_grid_path = RESULTS_FIGURES_DIR / "pneumonia_sample_grid.png"
    fig, axes = plt.subplots(2, 6, figsize=(12, 4.8))
    fig.suptitle("PneumoniaMNIST: Representative Pediatric Chest X-Rays (28×28 Grayscale)", fontsize=13, fontweight="bold", y=0.98)

    rng = np.random.RandomState(SEED)
    for row_idx, cls_idx in enumerate([0, 1]):
        cls_name = CLASS_LABELS[cls_idx]
        indices = np.where(train_labels == cls_idx)[0]
        selected_indices = rng.choice(indices, size=6, replace=False)

        for col_idx, img_idx in enumerate(selected_indices):
            img_arr = raw_images[img_idx]
            ax = axes[row_idx, col_idx]
            ax.imshow(img_arr, cmap="bone")
            ax.axis("off")
            if col_idx == 0:
                ax.set_title(
                    f"Label {cls_idx}: {cls_name.capitalize()}",
                    loc="right",
                    fontsize=10,
                    fontweight="bold",
                    color="#1E3A8A" if cls_idx == 0 else "#DC2626",
                    pad=6,
                )

    plt.tight_layout()
    plt.savefig(sample_grid_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved chest X-ray sample grid: {sample_grid_path}")

    # Figure 2: Class Distribution Bar Chart (Train vs Val vs Test)
    dist_fig_path = RESULTS_FIGURES_DIR / "pneumonia_class_distribution.png"
    fig, ax = plt.subplots(figsize=(9, 5.5))

    classes_display = ["Normal (0)", "Pneumonia (1)"]
    x = np.arange(len(classes_display))
    width = 0.25

    rects1 = ax.bar(
        x - width,
        [train_dist["normal"], train_dist["pneumonia"]],
        width,
        label=f"Train Split (N={train_len})",
        color="#2563EB",
        edgecolor="#1E3A8A",
    )
    rects2 = ax.bar(
        x,
        [val_dist["normal"], val_dist["pneumonia"]],
        width,
        label=f"Validation Split (N={val_len})",
        color="#10B981",
        edgecolor="#065F46",
    )
    rects3 = ax.bar(
        x + width,
        [test_dist["normal"], test_dist["pneumonia"]],
        width,
        label=f"Test Split (N={test_len})",
        color="#F59E0B",
        edgecolor="#92400E",
    )

    ax.set_ylabel("Number of Chest X-Ray Images", fontsize=11, fontweight="bold")
    ax.set_title("PneumoniaMNIST Class Distribution Across Official Splits", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes_display, fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    ax.bar_label(rects1, padding=3, fontsize=9, fontweight="bold")
    ax.bar_label(rects2, padding=3, fontsize=9, fontweight="bold")
    ax.bar_label(rects3, padding=3, fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(dist_fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved class distribution figure: {dist_fig_path}")

    # 8. Produce Machine-Readable JSON Report
    print("\n[Step 6/6] Writing structured verification report...")
    report_path = RESULTS_LOGS_DIR / "pneumonia_mnist_verification_report.json"
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "experiment_id": "EXP-MED-000-DATA",
        "status": "PASSED",
        "random_seed": SEED,
        "dataset_name": DATASET_NAME,
        "task": "binary-class chest X-ray classification",
        "official_source": "MedMNIST v3 (Zenodo / Yang et al.)",
        "total_samples": total_len,
        "train_samples": train_len,
        "validation_samples": val_len,
        "test_samples": test_len,
        "num_classes": NUM_CLASSES,
        "classes": list(CLASSES),
        "label_mapping": CLASS_LABELS,
        "image_resolution": [28, 28],
        "channels": 1,
        "tensor_shape": list(IMAGE_SHAPE),
        "train_class_distribution": train_dist,
        "validation_class_distribution": val_dist,
        "test_class_distribution": test_dist,
        "pixel_statistics": {
            "min_pixel_value": min_pixel_val,
            "max_pixel_value": max_pixel_val,
            "grayscale_mean": round(grayscale_mean, 5),
            "grayscale_std": round(grayscale_std, 5),
        },
        "artifacts_generated": {
            "sample_grid_figure": str(sample_grid_path.relative_to(PROJECT_ROOT)),
            "class_distribution_figure": str(dist_fig_path.relative_to(PROJECT_ROOT)),
            "report_file": str(report_path.relative_to(PROJECT_ROOT)),
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"  ✓ Saved verification JSON report: {report_path}")
    print("\n" + "=" * 75)
    print("PHASE MEDICAL-01 VERIFICATION COMPLETE: ALL CHECKS PASSED.")
    print("=" * 75)
    return report_data


if __name__ == "__main__":
    verify_and_analyze_pneumoniamnist()
