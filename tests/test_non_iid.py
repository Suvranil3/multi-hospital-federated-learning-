"""
PyTest Unit Tests: Non-IID Controlled-Capacity Partitioning and Training (Phase 4).

Verifies:
1. Controlled hospital capacities: Each client receives exactly 470 or 471 samples (sum = 4,708).
2. Partition coverage & mutual exclusivity: All 4,708 training samples partitioned without overlap.
3. Exact class conservation: Exactly 1,214 Normal and 3,494 Pneumonia samples distributed.
4. Partition determinism: Fixed seeds yield byte-identical partition indices across runs.
5. Data isolation: Client loaders wrap train_dataset Subsets; val and test loaders remain isolated.
6. Single-class client robustness: HospitalClient.fit() executes stably on extreme single-class data.
7. Multi-seed aggregate statistical characterization across alpha in {1.0, 0.5, 0.1}.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from medmnist import PneumoniaMNIST

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import DATA_RAW_DIR, SEED, TRAIN_SAMPLES, VAL_SAMPLES, TEST_SAMPLES, NUM_HOSPITALS, set_seed
from federated_learning.model import get_model
from federated_learning.dataset import (
    create_controlled_non_iid_partitions,
    prepare_controlled_non_iid_data_loaders,
)
from federated_learning.client import (
    HospitalClient,
    get_parameters_from_model,
)


@pytest.fixture(autouse=True)
def setup_seed():
    """Ensure deterministic random seed for each test."""
    set_seed(SEED)


@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.1])
def test_controlled_hospital_capacities(alpha):
    """Verify hospital capacities are strictly controlled to 470 or 471 samples."""
    train_dataset = PneumoniaMNIST(split="train", root=str(DATA_RAW_DIR), download=False)
    client_indices, meta = create_controlled_non_iid_partitions(
        train_dataset, num_clients=10, alpha=alpha, seed=42
    )

    assert len(client_indices) == 10, f"Expected 10 clients, got {len(client_indices)}"
    total_assigned = 0
    for cid, idxs in enumerate(client_indices):
        assert len(idxs) in [470, 471], f"Client {cid} has unexpected sample count: {len(idxs)}"
        total_assigned += len(idxs)

    assert total_assigned == TRAIN_SAMPLES, f"Expected total {TRAIN_SAMPLES}, got {total_assigned}"
    assert meta["min_client_size"] == 470
    assert meta["max_client_size"] == 471


@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.1])
def test_partition_coverage_and_mutual_exclusivity(alpha):
    """Verify all 4,708 training samples are partitioned with zero duplicates."""
    train_dataset = PneumoniaMNIST(split="train", root=str(DATA_RAW_DIR), download=False)
    client_indices, _ = create_controlled_non_iid_partitions(
        train_dataset, num_clients=10, alpha=alpha, seed=42
    )

    all_indices = []
    for idxs in client_indices:
        all_indices.extend(idxs)

    assert len(all_indices) == TRAIN_SAMPLES
    assert len(set(all_indices)) == TRAIN_SAMPLES, "Detected overlapping samples across clients!"


@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.1])
def test_exact_class_conservation(alpha):
    """Verify consortium class totals strictly match population counts (1214 Normal, 3494 Pneumonia)."""
    train_dataset = PneumoniaMNIST(split="train", root=str(DATA_RAW_DIR), download=False)
    client_indices, meta = create_controlled_non_iid_partitions(
        train_dataset, num_clients=10, alpha=alpha, seed=42
    )
    labels = train_dataset.labels.flatten()

    total_c0 = 0
    total_c1 = 0
    for idxs in client_indices:
        client_labels = labels[idxs]
        total_c0 += int(np.sum(client_labels == 0))
        total_c1 += int(np.sum(client_labels == 1))

    assert total_c0 == 1214, f"Normal count {total_c0} != 1214"
    assert total_c1 == 3494, f"Pneumonia count {total_c1} != 3494"
    assert total_c0 + total_c1 == TRAIN_SAMPLES


def test_partition_determinism():
    """Verify that identical random seeds yield identical partition allocations."""
    train_dataset = PneumoniaMNIST(split="train", root=str(DATA_RAW_DIR), download=False)
    indices_run1, meta1 = create_controlled_non_iid_partitions(train_dataset, alpha=0.5, seed=42)
    indices_run2, meta2 = create_controlled_non_iid_partitions(train_dataset, alpha=0.5, seed=42)

    assert meta1["acceptance_attempt"] == meta2["acceptance_attempt"]
    for cid in range(10):
        assert indices_run1[cid] == indices_run2[cid]


def test_dataset_isolation_integrity():
    """
    Verify client loaders wrap train_dataset Subsets exclusively,
    while validation and test loaders wrap official val and test splits.
    """
    client_loaders, val_loader, test_loader, meta = prepare_controlled_non_iid_data_loaders(
        data_dir=DATA_RAW_DIR,
        num_clients=10,
        alpha=1.0,
        batch_size=32,
        seed=42,
    )

    # 1. Check client loaders
    assert len(client_loaders) == 10
    total_train_in_loaders = 0
    for cid, loader in enumerate(client_loaders):
        dataset = loader.dataset
        assert isinstance(dataset, Subset), f"Client {cid} dataset must be a Subset"
        assert isinstance(dataset.dataset, PneumoniaMNIST)
        assert dataset.dataset.split == "train"
        assert len(dataset) in [470, 471]
        total_train_in_loaders += len(dataset)

    assert total_train_in_loaders == TRAIN_SAMPLES

    # 2. Check validation loader
    assert isinstance(val_loader.dataset, PneumoniaMNIST)
    assert val_loader.dataset.split == "val"
    assert len(val_loader.dataset) == VAL_SAMPLES

    # 3. Check test loader
    assert isinstance(test_loader.dataset, PneumoniaMNIST)
    assert test_loader.dataset.split == "test"
    assert len(test_loader.dataset) == TEST_SAMPLES


def test_single_class_or_skewed_hospital_fit():
    """
    Verify HospitalClient.fit() executes stably and updates weights without NaNs
    even when local hospital data is 100% single-class (e.g. 100% Pneumonia).
    """
    device = torch.device("cpu")
    # Synthetic 100% Pneumonia dataset (class 1)
    single_class_images = torch.randn(64, 1, 28, 28)
    single_class_targets = torch.ones((64, 1), dtype=torch.long)
    loader = DataLoader(TensorDataset(single_class_images, single_class_targets), batch_size=32, shuffle=True)

    client = HospitalClient(client_id="pure_pneumonia_hospital", train_loader=loader, local_epochs=2, device=device)
    init_params = get_parameters_from_model(client.model)

    updated_params, num_samples, metrics = client.fit(init_params, config={"local_epochs": 2, "lr": 0.001})

    assert num_samples == 64
    assert not np.isnan(metrics["train_loss"])
    assert 0.0 <= metrics["train_accuracy"] <= 100.0
    for arr in updated_params:
        assert not np.isnan(arr).any(), "Detected NaN in updated client parameters!"


def test_multi_seed_aggregate_heterogeneity_characterization():
    """
    Empirical statistical characterization across 10 independent random seeds.
    Computes client-level class proportion variance for alpha in {1.0, 0.5, 0.1}.
    Reports mean and std of variance, verifying the aggregate ensemble trend
    without requiring strict monotonic ordering for every individual seed.
    """
    train_dataset = PneumoniaMNIST(split="train", root=str(DATA_RAW_DIR), download=False)
    test_seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]

    results_by_alpha = {1.0: [], 0.5: [], 0.1: []}

    for alpha in [1.0, 0.5, 0.1]:
        for seed in test_seeds:
            _, meta = create_controlled_non_iid_partitions(
                train_dataset=train_dataset,
                num_clients=10,
                alpha=alpha,
                seed=seed,
                tolerance=40,
            )
            results_by_alpha[alpha].append(meta["normal_ratio_var"])

    summary = {}
    for alpha, variances in results_by_alpha.items():
        mean_var = float(np.mean(variances))
        std_var = float(np.std(variances))
        summary[alpha] = {"mean_var": round(mean_var, 4), "std_var": round(std_var, 4)}

    print("\n[Multi-Seed Characterization (10 seeds)] Variance of Hospital Normal Proportions:")
    for alpha, s in summary.items():
        print(f"  alpha = {alpha:3.1f}: Mean Variance = {s['mean_var']:.4f} ± {s['std_var']:.4f}")

    # Verify that the aggregate ensemble mean variance at alpha=0.1 exceeds alpha=1.0
    assert summary[0.1]["mean_var"] > summary[1.0]["mean_var"], (
        f"Expected aggregate variance at alpha=0.1 ({summary[0.1]['mean_var']}) "
        f"to exceed alpha=1.0 ({summary[1.0]['mean_var']})"
    )
