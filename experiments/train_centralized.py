"""
Centralized Small CNN Baseline Training on CIFAR-10 (Phase 2).

Establish a centralized benchmark before introducing Federated Learning.
Pipeline:
1. Loads raw CIFAR-10 data.
2. Deterministically partitions training data into train (80%, 40,000) and validation (20%, 10,000).
3. Leaves official CIFAR-10 test set (10,000) untouched until final post-training evaluation.
4. Trains SmallCNN with data augmentation, Adam optimizer, and CrossEntropyLoss.
5. Records epoch-by-epoch train/val loss & accuracy, training time, and system metadata.
6. Generates loss/accuracy learning curves and test confusion matrix figures.
7. Emits structured JSON experiment record to results/logs/centralized_cnn_results.json.
"""

import sys
from pathlib import Path
import os
import time
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Tuple, List

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from experiments.config import (
    DATA_RAW_DIR,
    RESULTS_FIGURES_DIR,
    RESULTS_LOGS_DIR,
    RESULTS_MODELS_DIR,
    SEED,
    CIFAR10_CLASSES,
    CIFAR10_NUM_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    CENTRALIZED_EPOCHS,
    CENTRALIZED_BATCH_SIZE,
    CENTRALIZED_LEARNING_RATE,
    CENTRALIZED_VAL_RATIO,
    CENTRALIZED_WEIGHT_DECAY,
    set_seed,
)
from federated_learning.model import SmallCNN, get_model, count_parameters


class TransformedSubset(Dataset):
    """
    Subsets a PyTorch Dataset (like CIFAR-10) with custom transforms applied on the fly.
    """

    def __init__(self, base_dataset: Dataset, indices: List[int], transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, target = self.base_dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def prepare_data_loaders(
    data_dir: Path,
    val_ratio: float = 0.2,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Deterministically create train, validation, and untouched test DataLoaders.
    Train split receives augmentation; validation and test splits receive standard normalization.
    """
    # 1. Define Transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # 2. Load Raw Datasets (PIL format without initial tensor transforms)
    raw_train_dataset = datasets.CIFAR10(root=str(data_dir), train=True, download=True)
    raw_test_dataset = datasets.CIFAR10(root=str(data_dir), train=False, download=True)

    targets = np.array(raw_train_dataset.targets)
    total_train_samples = len(targets)
    num_classes = len(CIFAR10_CLASSES)

    # 3. Stratified Deterministic Split (Exact uniform balance per class)
    rng = np.random.RandomState(seed)
    train_indices = []
    val_indices = []

    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        rng.shuffle(class_indices)
        val_count = int(round(len(class_indices) * val_ratio))
        val_indices.extend(class_indices[:val_count])
        train_indices.extend(class_indices[val_count:])

    # Shuffle final indices deterministically
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    # 4. Wrap with appropriate transforms
    train_subset = TransformedSubset(raw_train_dataset, train_indices, transform=train_transform)
    val_subset = TransformedSubset(raw_train_dataset, val_indices, transform=eval_transform)
    test_subset = TransformedSubset(raw_test_dataset, list(range(len(raw_test_dataset))), transform=eval_transform)

    # 5. Create DataLoaders
    # Pin memory if CUDA is available for accelerated transfer
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    split_info = {
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "test_samples": len(test_subset),
        "train_samples_per_class": len(train_indices) // num_classes,
        "val_samples_per_class": len(val_indices) // num_classes,
        "test_samples_per_class": len(test_subset) // num_classes,
        "val_ratio": val_ratio,
    }

    return train_loader, val_loader, test_loader, split_info


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Execute one full training epoch across batches.
    Returns (average_loss, accuracy_percentage).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_predictions: bool = False,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model performance on validation or test DataLoader.
    Returns (average_loss, accuracy_percentage, all_targets, all_predictions).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            if return_predictions:
                all_targets.append(targets.cpu().numpy())
                all_preds.append(preds.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = (correct / total) * 100.0

    targets_arr = np.concatenate(all_targets) if return_predictions and all_targets else np.array([])
    preds_arr = np.concatenate(all_preds) if return_predictions and all_preds else np.array([])

    return avg_loss, accuracy, targets_arr, preds_arr


def plot_metrics(
    epoch_metrics: List[Dict[str, Any]],
    output_loss_path: Path,
    output_acc_path: Path,
) -> None:
    """
    Generate clean, legible learning curve plots for Loss and Accuracy.
    """
    epochs = [m["epoch"] for m in epoch_metrics]
    train_losses = [m["train_loss"] for m in epoch_metrics]
    val_losses = [m["val_loss"] for m in epoch_metrics]
    train_accs = [m["train_accuracy"] for m in epoch_metrics]
    val_accs = [m["val_accuracy"] for m in epoch_metrics]

    # 1. Loss Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, "o-", color="#1f77b4", linewidth=2, label="Train Loss")
    ax.plot(epochs, val_losses, "s--", color="#d62728", linewidth=2, label="Validation Loss")
    ax.set_title("Centralized Small CNN: Cross-Entropy Loss vs. Epoch", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax.set_xticks(epochs)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_loss_path, dpi=200)
    plt.close(fig)

    # 2. Accuracy Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_accs, "o-", color="#2ca02c", linewidth=2, label="Train Accuracy (%)")
    ax.plot(epochs, val_accs, "s--", color="#ff7f0e", linewidth=2, label="Validation Accuracy (%)")
    ax.set_title("Centralized Small CNN: Classification Accuracy vs. Epoch", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(epochs)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_acc_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(
    targets: np.ndarray,
    preds: np.ndarray,
    classes: List[str],
    output_path: Path,
) -> np.ndarray:
    """
    Compute and plot the 10x10 confusion matrix on the test dataset.
    """
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=classes,
        yticklabels=classes,
        title="CIFAR-10 Test Set Confusion Matrix (Centralized Small CNN)",
        ylabel="True Class",
        xlabel="Predicted Class",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    # Annotate cell numbers
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                f"{cm[i, j]:d}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return cm


def run_centralized_training(
    epochs: int = CENTRALIZED_EPOCHS,
    batch_size: int = CENTRALIZED_BATCH_SIZE,
    learning_rate: float = CENTRALIZED_LEARNING_RATE,
    val_ratio: float = CENTRALIZED_VAL_RATIO,
    weight_decay: float = CENTRALIZED_WEIGHT_DECAY,
    seed: int = SEED,
    device_name: str = "auto",
) -> Dict[str, Any]:
    """
    Main execution pipeline for Phase 2 Centralized CNN training.
    """
    start_total_time = time.time()
    set_seed(seed)

    # 1. Device Resolution
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    device_str = (
        torch.cuda.get_device_name(device.index or 0)
        if device.type == "cuda"
        else "CPU"
    )

    print("=" * 70)
    print("PHASE 2: CENTRALIZED SMALL CNN BASELINE TRAINING (CIFAR-10)")
    print(f"Timestamp:        {datetime.now().isoformat()}")
    print(f"Deterministic Seed: {seed}")
    print(f"Compute Device:     {device.type.upper()} ({device_str})")
    print(f"Epochs:             {epochs}")
    print(f"Batch Size:         {batch_size}")
    print(f"Learning Rate:      {learning_rate}")
    print(f"Weight Decay:       {weight_decay}")
    print("=" * 70)

    # 2. Data Preparation
    print("\n[Step 1/5] Preparing CIFAR-10 DataLoaders...")
    train_loader, val_loader, test_loader, split_info = prepare_data_loaders(
        data_dir=DATA_RAW_DIR,
        val_ratio=val_ratio,
        batch_size=batch_size,
        seed=seed,
    )
    print(f"  Training Split:   {split_info['train_samples']} samples ({split_info['train_samples_per_class']}/class)")
    print(f"  Validation Split: {split_info['val_samples']} samples ({split_info['val_samples_per_class']}/class)")
    print(f"  Test Set:         {split_info['test_samples']} samples (untouched until final eval)")

    # 3. Model, Loss, Optimizer
    print("\n[Step 2/5] Initializing SmallCNN Architecture...")
    model = get_model(num_classes=CIFAR10_NUM_CLASSES).to(device)
    params = count_parameters(model)
    print(f"  Total Parameters:     {params['total']:,}")
    print(f"  Trainable Parameters: {params['trainable']:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # 4. Training Loop
    print("\n[Step 3/5] Executing Training Loop...")
    epoch_metrics = []
    best_val_acc = 0.0
    best_epoch = -1
    best_checkpoint_path = RESULTS_MODELS_DIR / "best_centralized_cnn.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, return_predictions=False)
        epoch_duration = time.time() - t0

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 2),
            "epoch_time_seconds": round(epoch_duration, 2),
        }
        epoch_metrics.append(epoch_record)

        # Track and checkpoint best model on validation accuracy
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "seed": seed,
                },
                best_checkpoint_path,
            )

        best_flag = " [BEST]" if is_best else ""
        print(
            f"  Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.2f}% | "
            f"Time: {epoch_duration:.1f}s{best_flag}"
        )

    total_train_time = time.time() - start_total_time
    print(f"\n  Training completed in {total_train_time:.2f} seconds.")
    print(f"  Best Validation Accuracy: {best_val_acc:.2f}% at Epoch {best_epoch}")
    print(f"  Best Model Weights saved to: {best_checkpoint_path}")

    # 5. Untouched Test Set Evaluation
    print("\n[Step 4/5] Evaluating on Untouched CIFAR-10 Test Set...")
    # Load best validation checkpoint
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_targets, test_preds = evaluate(
        model, test_loader, criterion, device, return_predictions=True
    )
    test_correct = int(np.sum(test_targets == test_preds))
    print(f"  Final Test Loss:     {test_loss:.4f}")
    print(f"  Final Test Accuracy: {test_acc:.2f}% ({test_correct}/{len(test_targets)})")

    # 6. Visualizations & Machine-Readable Artifacts
    print("\n[Step 5/5] Generating Plots and Structured JSON Log...")
    loss_fig_path = RESULTS_FIGURES_DIR / "centralized_training_loss.png"
    acc_fig_path = RESULTS_FIGURES_DIR / "centralized_training_accuracy.png"
    cm_fig_path = RESULTS_FIGURES_DIR / "confusion_matrix.png"
    json_log_path = RESULTS_LOGS_DIR / "centralized_cnn_results.json"

    plot_metrics(epoch_metrics, loss_fig_path, acc_fig_path)
    print(f"  ✓ Saved loss curve:      {loss_fig_path}")
    print(f"  ✓ Saved accuracy curve:  {acc_fig_path}")

    plot_confusion_matrix(test_targets, test_preds, list(CIFAR10_CLASSES), cm_fig_path)
    print(f"  ✓ Saved confusion matrix:{cm_fig_path}")

    results_summary = {
        "experiment_id": "EXP-001-CENTRAL-CNN",
        "timestamp": datetime.now().isoformat(),
        "status": "COMPLETED",
        "seed": seed,
        "environment": {
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "device": device_str,
            "device_type": device.type,
        },
        "configuration": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer": "Adam",
            "criterion": "CrossEntropyLoss",
            "val_ratio": val_ratio,
            "train_samples": split_info["train_samples"],
            "val_samples": split_info["val_samples"],
            "test_samples": split_info["test_samples"],
        },
        "model": {
            "name": "SmallCNN",
            "total_parameters": params["total"],
            "trainable_parameters": params["trainable"],
        },
        "training_performance": {
            "best_epoch": best_epoch,
            "best_val_accuracy": round(best_val_acc, 2),
            "final_train_loss": epoch_metrics[-1]["train_loss"],
            "final_train_accuracy": epoch_metrics[-1]["train_accuracy"],
            "final_val_loss": epoch_metrics[-1]["val_loss"],
            "final_val_accuracy": epoch_metrics[-1]["val_accuracy"],
            "total_training_time_seconds": round(total_train_time, 2),
        },
        "test_performance": {
            "test_loss": round(test_loss, 4),
            "test_accuracy": round(test_acc, 2),
            "test_correct": test_correct,
            "total_test_samples": len(test_targets),
        },
        "epoch_metrics": epoch_metrics,
        "artifacts": {
            "loss_figure": str(loss_fig_path.relative_to(PROJECT_ROOT)),
            "accuracy_figure": str(acc_fig_path.relative_to(PROJECT_ROOT)),
            "confusion_matrix_figure": str(cm_fig_path.relative_to(PROJECT_ROOT)),
            "model_checkpoint": str(best_checkpoint_path.relative_to(PROJECT_ROOT)),
            "results_json": str(json_log_path.relative_to(PROJECT_ROOT)),
        },
    }

    with open(json_log_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print(f"  ✓ Saved experiment JSON: {json_log_path}")
    print("\n" + "=" * 70)
    print("PHASE 2 CENTRALIZED CNN BASELINE COMPLETE.")
    print("=" * 70)
    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centralized Small CNN Baseline on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=CENTRALIZED_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=CENTRALIZED_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=CENTRALIZED_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--val-ratio", type=float, default=CENTRALIZED_VAL_RATIO, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, cpu)")
    args = parser.parse_args()

    run_centralized_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device_name=args.device,
    )
