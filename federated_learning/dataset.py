"""
Federated Dataset Partitioning and Loader Preparation (Phase 3: 10-Hospital IID Baseline).
Primary Dataset: PneumoniaMNIST (MedMNIST v3).

Partitions the official training split (4,708 samples) into 10 mutually exclusive,
approximately equal IID partitions while preserving the underlying class ratio
(~25.79% Normal, ~74.21% Pneumonia) across all simulated hospital clients.
"""

from typing import List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from medmnist import PneumoniaMNIST

from experiments.config import (
    DATA_RAW_DIR,
    SEED,
    PNEUMONIA_MEAN,
    PNEUMONIA_STD,
    TRAIN_SAMPLES,
    VAL_SAMPLES,
    TEST_SAMPLES,
    NUM_HOSPITALS,
    DEFAULT_BATCH_SIZE,
    set_seed,
)


def create_iid_partitions(
    train_dataset: PneumoniaMNIST,
    num_clients: int = NUM_HOSPITALS,
    seed: int = SEED,
) -> List[List[int]]:
    """
    Partition the training dataset indices into `num_clients` mutually exclusive subsets.
    Uses stratified sampling to preserve the empirical class distribution across all clients:
      - Total train samples: 4,708
      - Class 0 (Normal): 1,214 samples -> ~121-122 samples per client
      - Class 1 (Pneumonia): 3,494 samples -> ~349-350 samples per client
      - Per-client total: ~470-471 samples

    Returns:
        List of length `num_clients`, each containing a list of sample indices.
    """
    rng = np.random.RandomState(seed)
    labels = train_dataset.labels.flatten()

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    # Stratify by class to maintain exact IID distribution
    for class_id in [0, 1]:
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)

        # Distribute class indices evenly across clients
        chunks = np.array_split(class_indices, num_clients)
        if class_id == 1:
            # Invert allocation order to balance remainder splits:
            # Class 0: clients 0..3 get 122, clients 4..9 get 121
            # Class 1: clients 0..5 get 349, clients 6..9 get 350
            # Combined: clients 0..3: 471, clients 4..5: 470, clients 6..9: 471 (sum = 4,708)
            chunks = chunks[::-1]

        for client_id, chunk in enumerate(chunks):
            client_indices[client_id].extend(chunk.tolist())

    # Shuffle within each client partition for randomized local batches
    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])

    return client_indices


def prepare_federated_data_loaders(
    data_dir: Path = DATA_RAW_DIR,
    num_clients: int = NUM_HOSPITALS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
    seed: int = SEED,
) -> Tuple[List[DataLoader], DataLoader, DataLoader, Dict[str, Any]]:
    """
    Prepare client training DataLoaders and global validation & test DataLoaders.

    Args:
        data_dir: Root directory for PneumoniaMNIST data files.
        num_clients: Number of simulated hospital clients (default 10).
        batch_size: Local batch size (default 32).
        num_workers: Subprocess workers for DataLoaders (0 for safe Ray simulation).
        seed: Deterministic random seed.

    Returns:
        (client_loaders, val_loader, test_loader, partition_metadata)
    """
    set_seed(seed)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=PNEUMONIA_MEAN, std=PNEUMONIA_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=PNEUMONIA_MEAN, std=PNEUMONIA_STD),
    ])

    # 1. Load official dataset splits
    train_dataset = PneumoniaMNIST(split="train", root=str(data_dir), transform=train_transform, download=True)
    val_dataset = PneumoniaMNIST(split="val", root=str(data_dir), transform=eval_transform, download=True)
    test_dataset = PneumoniaMNIST(split="test", root=str(data_dir), transform=eval_transform, download=True)

    assert len(train_dataset) == TRAIN_SAMPLES, f"Expected {TRAIN_SAMPLES} train, got {len(train_dataset)}"
    assert len(val_dataset) == VAL_SAMPLES, f"Expected {VAL_SAMPLES} val, got {len(val_dataset)}"
    assert len(test_dataset) == TEST_SAMPLES, f"Expected {TEST_SAMPLES} test, got {len(test_dataset)}"

    # 2. Partition official train dataset into 10 IID client subsets
    client_indices = create_iid_partitions(train_dataset, num_clients=num_clients, seed=seed)

    # Verify partition integrity
    all_assigned = []
    for idxs in client_indices:
        all_assigned.extend(idxs)
    assert len(all_assigned) == TRAIN_SAMPLES, f"Expected {TRAIN_SAMPLES} partitioned indices, got {len(all_assigned)}"
    assert len(set(all_assigned)) == TRAIN_SAMPLES, "Client partitions must be mutually exclusive!"

    # 3. Build per-client DataLoaders
    client_loaders: List[DataLoader] = []
    client_stats: List[Dict[str, Any]] = []
    labels = train_dataset.labels.flatten()

    for cid, idxs in enumerate(client_indices):
        subset = Subset(train_dataset, idxs)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
        )
        client_loaders.append(loader)

        client_labels = labels[idxs]
        norm_count = int(np.sum(client_labels == 0))
        pneu_count = int(np.sum(client_labels == 1))
        client_stats.append({
            "client_id": cid,
            "total_samples": len(idxs),
            "normal_samples": norm_count,
            "pneumonia_samples": pneu_count,
            "normal_ratio": round(norm_count / len(idxs), 4),
            "pneumonia_ratio": round(pneu_count / len(idxs), 4),
        })

    # 4. Global validation and test loaders
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    metadata = {
        "num_clients": num_clients,
        "batch_size": batch_size,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "client_stats": client_stats,
    }

    return client_loaders, val_loader, test_loader, metadata
