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


def create_controlled_non_iid_partitions(
    train_dataset: PneumoniaMNIST,
    num_clients: int = NUM_HOSPITALS,
    alpha: float = 1.0,
    seed: int = SEED,
    tolerance: int = 40,
    max_attempts: int = 5000,
) -> Tuple[List[List[int]], Dict[str, Any]]:
    """
    Population-calibrated constrained Beta/Dirichlet case-mix allocation using
    rejection sampling, proportional scaling, and largest-remainder rounding.

    Controls client sample sizes to ~471 samples (8 clients with 471, 2 clients with 470,
    sum = 4,708), matching the Phase 3 sample capacities while inducing label-distribution
    heterogeneity (case-mix skew) across simulated hospital clients.

    Dirichlet Parameterization:
      The Dirichlet parameter vector is alpha multiplied by the empirical global class prior:
        alpha_vec = alpha * [p_normal, p_pneumonia]
      where alpha is the total concentration parameter (alpha = sum(alpha_vec)).
      For 2 classes, this Dirichlet distribution is parameterized identically as
      Beta(alpha * p_normal, alpha * p_pneumonia).

    Allocation Procedure:
      1. Rejection sampling: Draws candidate hospital Normal proportions until the unconstrained
         sample sum is within `tolerance` (default: 40) of the true total Normal count (1,214).
      2. Proportional scaling: Scales candidate counts by (N_0 / sum_ideal).
      3. Largest-remainder rounding (Hamilton-Hare method): Allocates integer floors and distributes
         remaining samples to clients with largest fractional residuals, strictly preserving:
           - Global Normal count: exactly 1,214
           - Global Pneumonia count: exactly 3,494
           - Hospital total capacity: exactly 470 or 471 (sum = 4,708)
           - Mutual exclusivity: each training sample is assigned exactly once.

    Returns:
        (client_indices, partition_metadata)
    """
    labels = train_dataset.labels.flatten()
    total_samples = len(labels)
    assert total_samples == TRAIN_SAMPLES, f"Expected {TRAIN_SAMPLES} train samples, got {total_samples}"

    n_normal = int(np.sum(labels == 0))
    n_pneumonia = int(np.sum(labels == 1))
    p_normal = n_normal / total_samples
    p_pneumonia = n_pneumonia / total_samples

    # Define balanced hospital capacities (8 x 471, 2 x 470 = 4,708)
    base_cap = total_samples // num_clients
    rem_cap = total_samples % num_clients
    capacities = np.array([base_cap + 1 if i < rem_cap else base_cap for i in range(num_clients)])

    rng = np.random.RandomState(seed)

    # 1. Rejection Sampling
    target_proportions = None
    acceptance_attempt = -1
    initial_diff = 0.0

    for attempt in range(max_attempts):
        q = rng.beta(alpha * p_normal, alpha * p_pneumonia, size=num_clients)
        ideal_counts = q * capacities
        diff = float(ideal_counts.sum() - n_normal)
        if abs(diff) <= tolerance:
            scale = n_normal / ideal_counts.sum()
            scaled = ideal_counts * scale
            if np.all(scaled <= capacities):
                target_proportions = q
                acceptance_attempt = attempt
                initial_diff = diff
                break

    if target_proportions is None:
        raise RuntimeError(
            f"Failed to find valid Dirichlet draw within tolerance {tolerance} "
            f"for alpha={alpha} after {max_attempts} attempts."
        )

    # 2. Proportional Scaling
    ideal_counts = target_proportions * capacities
    scaled_counts = ideal_counts * (n_normal / ideal_counts.sum())

    # 3. Largest-Remainder Rounding
    floor_counts = np.floor(scaled_counts).astype(int)
    remainder = n_normal - floor_counts.sum()
    fractions = scaled_counts - floor_counts

    eligible = np.where(floor_counts < capacities)[0]
    ranked_order = eligible[np.argsort(-fractions[eligible])]
    for idx in ranked_order[:remainder]:
        floor_counts[idx] += 1

    norm_counts = floor_counts
    pneu_counts = capacities - norm_counts

    assert norm_counts.sum() == n_normal, f"Normal sum {norm_counts.sum()} != {n_normal}"
    assert pneu_counts.sum() == n_pneumonia, f"Pneumonia sum {pneu_counts.sum()} != {n_pneumonia}"
    assert np.all(norm_counts >= 0) and np.all(norm_counts <= capacities)

    # 4. Disjoint Sample Slicing
    c0_indices = np.where(labels == 0)[0]
    c1_indices = np.where(labels == 1)[0]
    rng.shuffle(c0_indices)
    rng.shuffle(c1_indices)

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    c0_ptr = 0
    c1_ptr = 0

    client_stats: List[Dict[str, Any]] = []
    achieved_ratios: List[float] = []

    for cid in range(num_clients):
        c0_c = norm_counts[cid]
        c1_c = pneu_counts[cid]
        idxs = c0_indices[c0_ptr : c0_ptr + c0_c].tolist() + c1_indices[c1_ptr : c1_ptr + c1_c].tolist()
        c0_ptr += c0_c
        c1_ptr += c1_c

        rng.shuffle(idxs)
        client_indices[cid] = idxs

        achieved_ratio = round(c0_c / capacities[cid], 4)
        achieved_ratios.append(achieved_ratio)
        client_stats.append({
            "client_id": cid,
            "total_samples": int(capacities[cid]),
            "normal_samples": int(c0_c),
            "pneumonia_samples": int(c1_c),
            "normal_ratio": achieved_ratio,
            "pneumonia_ratio": round(1.0 - achieved_ratio, 4),
            "target_normal_ratio": round(float(target_proportions[cid]), 4),
            "deviation_from_target": round(float(abs(achieved_ratio - target_proportions[cid])), 4),
        })

    meta: Dict[str, Any] = {
        "alpha": alpha,
        "total_concentration": alpha,
        "dirichlet_parameter_vector": [round(alpha * p_normal, 4), round(alpha * p_pneumonia, 4)],
        "global_class_prior": {"normal": round(p_normal, 4), "pneumonia": round(p_pneumonia, 4)},
        "tolerance": tolerance,
        "acceptance_attempt": acceptance_attempt,
        "initial_discrepancy": round(initial_diff, 2),
        "min_client_size": int(np.min(capacities)),
        "max_client_size": int(np.max(capacities)),
        "normal_ratio_mean": round(float(np.mean(achieved_ratios)), 4),
        "normal_ratio_std": round(float(np.std(achieved_ratios)), 4),
        "normal_ratio_var": round(float(np.var(achieved_ratios)), 4),
        "mean_abs_deviation": round(
            float(np.mean(np.abs(np.array(achieved_ratios) - target_proportions))), 4
        ),
        "max_abs_deviation": round(
            float(np.max(np.abs(np.array(achieved_ratios) - target_proportions))), 4
        ),
        "client_stats": client_stats,
    }

    return client_indices, meta


def prepare_controlled_non_iid_data_loaders(
    data_dir: Path = DATA_RAW_DIR,
    num_clients: int = NUM_HOSPITALS,
    alpha: float = 1.0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
    seed: int = SEED,
    tolerance: int = 40,
) -> Tuple[List[DataLoader], DataLoader, DataLoader, Dict[str, Any]]:
    """
    Prepare controlled-capacity Non-IID client DataLoaders and global validation & test DataLoaders.

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

    # 2. Partition official train dataset
    client_indices, meta = create_controlled_non_iid_partitions(
        train_dataset=train_dataset,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
        tolerance=tolerance,
    )

    # Verify partition integrity
    all_assigned = []
    for idxs in client_indices:
        all_assigned.extend(idxs)
    assert len(all_assigned) == TRAIN_SAMPLES, f"Expected {TRAIN_SAMPLES} partitioned indices, got {len(all_assigned)}"
    assert len(set(all_assigned)) == TRAIN_SAMPLES, "Client partitions must be mutually exclusive!"

    # 3. Build per-client DataLoaders
    client_loaders: List[DataLoader] = []
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
        "partition_type": "controlled_dirichlet_case_mix",
        "alpha": alpha,
        "seed": seed,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "heterogeneity_metrics": {k: v for k, v in meta.items() if k != "client_stats"},
        "client_stats": meta["client_stats"],
    }

    return client_loaders, val_loader, test_loader, metadata
