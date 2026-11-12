"""
Unit & Integration Tests for Phase 5 Adversarial Attack Modules (tests/test_attacks.py).
Verifies:
1. Label-flipping correctness, selectivity, and dataset immutability.
2. Model poisoning parameter transformation, tensor selectivity, and BatchNorm buffer preservation.
3. Backdoor trigger geometry, normalized intensity, and synthetic test set creation.
4. Clean test dataset isolation during training loops.
"""

import sys
from pathlib import Path
from typing import List
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import PNEUMONIA_MEAN, PNEUMONIA_STD
from federated_learning.model import SmallCNN, get_model
from federated_learning.client import get_parameters_from_model, set_parameters_to_model
from federated_learning.attacks.label_flipping import LabelFlippedSubset
from federated_learning.attacks.model_poisoning import (
    apply_adversarial_update_transformation,
    PoisonedHospitalClient,
)
from federated_learning.attacks.backdoor import (
    inject_trigger_to_tensor,
    BackdoorSubset,
    TriggeredPneumoniaTestDataset,
    evaluate_backdoor_asr,
    TRIGGER_VALUE_NORMALIZED,
    TRIGGER_COORDS,
)


@pytest.fixture
def dummy_medical_dataset() -> TensorDataset:
    """Create a controlled dummy dataset with 20 normal (0) and 80 pneumonia (1) samples."""
    torch.manual_seed(42)
    images = torch.randn(100, 1, 28, 28)
    labels = np.array([0] * 20 + [1] * 80, dtype=np.int64).reshape(-1, 1)
    return TensorDataset(images, torch.from_numpy(labels))


# ==============================================================================
# 1. Label-Flipping Tests
# ==============================================================================

def test_label_flipping_selectivity_and_counts(dummy_medical_dataset: TensorDataset) -> None:
    """Verify that only class 1 (Pneumonia) is flipped to 0, and class 0 is preserved."""
    attacked = LabelFlippedSubset(
        base_dataset=dummy_medical_dataset,
        source_class=1,
        target_class=0,
        flip_fraction=1.0,
        seed=42,
    )
    assert len(attacked) == 100
    meta = attacked.get_attack_metadata()
    assert meta["original_normal_count"] == 20
    assert meta["original_pneumonia_count"] == 80
    assert meta["poisoned_sample_count"] == 80

    # In the flipped dataset, all samples should now return label 0
    for idx in range(len(attacked)):
        _, target = attacked[idx]
        assert int(np.squeeze(target)) == 0


def test_label_flipping_partial_fraction(dummy_medical_dataset: TensorDataset) -> None:
    """Verify partial flip fraction (e.g., 50% of pneumonia samples flipped)."""
    attacked = LabelFlippedSubset(
        base_dataset=dummy_medical_dataset,
        source_class=1,
        target_class=0,
        flip_fraction=0.50,
        seed=42,
    )
    meta = attacked.get_attack_metadata()
    assert meta["poisoned_sample_count"] == 40  # 50% of 80 = 40

    total_zero_labels = 0
    total_one_labels = 0
    for idx in range(len(attacked)):
        _, target = attacked[idx]
        lbl = int(np.squeeze(target))
        if lbl == 0:
            total_zero_labels += 1
        elif lbl == 1:
            total_one_labels += 1

    assert total_zero_labels == 60  # 20 original normal + 40 flipped
    assert total_one_labels == 40   # 40 remaining pneumonia


def test_label_flipping_strict_non_mutation(dummy_medical_dataset: TensorDataset) -> None:
    """Verify that wrapping with LabelFlippedSubset NEVER mutates the underlying base dataset."""
    raw_images, raw_labels = dummy_medical_dataset.tensors
    labels_clone = raw_labels.clone()

    attacked = LabelFlippedSubset(
        base_dataset=dummy_medical_dataset,
        source_class=1,
        target_class=0,
        flip_fraction=1.0,
    )

    # Access all attacked samples
    for idx in range(len(attacked)):
        _ = attacked[idx]

    # Verify underlying tensor is identical bit-for-bit
    assert torch.equal(dummy_medical_dataset.tensors[1], labels_clone), (
        "Base dataset labels were mutated in-place!"
    )


# ==============================================================================
# 2. Model Poisoning Tests
# ==============================================================================

def test_model_poisoning_inversion_math_and_buffer_preservation() -> None:
    """Verify update inversion (gamma = -1.0) on trainable weights and buffer preservation."""
    model = SmallCNN()
    global_params = get_parameters_from_model(model)

    # Create dummy local training update: perturb parameters and advance BatchNorm buffers
    local_params = []
    for (name, tensor), g_arr in zip(model.state_dict().items(), global_params):
        is_trainable = any(name == p_name for p_name, _ in model.named_parameters())
        if is_trainable:
            # Add synthetic gradient step: delta = +0.1
            local_params.append((g_arr + 0.1).astype(g_arr.dtype))
        else:
            # Advance buffer: e.g. running_mean shifted by +0.05, num_batches_tracked incremented
            if "num_batches_tracked" in name:
                local_params.append((g_arr + 1).astype(g_arr.dtype))
            else:
                local_params.append((g_arr + 0.05).astype(g_arr.dtype))

    # Apply adversarial transformation with gamma = -1.0
    adv_params, meta = apply_adversarial_update_transformation(
        model=model,
        local_ndarrays=local_params,
        global_ndarrays=global_params,
        gamma=-1.0,
    )

    assert meta["manipulated_tensor_count"] == 12, "Expected exactly 12 trainable tensors to be manipulated"
    assert meta["preserved_buffer_count"] == 6, "Expected exactly 6 non-trainable buffers to be preserved"

    for (name, _), g_arr, l_arr, adv_arr in zip(
        model.state_dict().items(), global_params, local_params, adv_params
    ):
        is_trainable = any(name == p_name for p_name, _ in model.named_parameters())
        if is_trainable:
            # theta_adv = theta_global - 1.0 * (theta_local - theta_global) = g - 0.1
            expected = g_arr - (l_arr - g_arr)
            np.testing.assert_allclose(adv_arr, expected, rtol=1e-5, atol=1e-5)
        else:
            # Buffer must be exactly local_arr (unmodified by gamma)
            np.testing.assert_allclose(adv_arr, l_arr, rtol=1e-5, atol=1e-5)
            if "num_batches_tracked" in name:
                assert adv_arr.dtype == np.int64, "BatchNorm integer step tracker must remain int64"


def test_model_poisoning_scaling_math() -> None:
    """Verify update scaling (gamma = +10.0) multiplies trainable parameter update by 10."""
    model = SmallCNN()
    global_params = get_parameters_from_model(model)
    local_params = [(g_arr + 0.05).astype(g_arr.dtype) for g_arr in global_params]

    adv_params, meta = apply_adversarial_update_transformation(
        model=model,
        local_ndarrays=local_params,
        global_ndarrays=global_params,
        gamma=10.0,
    )

    trainable_names = set(name for name, _ in model.named_parameters())
    for (name, _), g_arr, l_arr, adv_arr in zip(
        model.state_dict().items(), global_params, local_params, adv_params
    ):
        if name in trainable_names:
            expected = g_arr + 10.0 * (l_arr - g_arr)
            np.testing.assert_allclose(adv_arr, expected, rtol=1e-5, atol=1e-5)


def test_poisoned_hospital_client_execution(dummy_medical_dataset: TensorDataset) -> None:
    """Verify PoisonedHospitalClient executes fit and applies gamma."""
    loader = DataLoader(dummy_medical_dataset, batch_size=16)
    client = PoisonedHospitalClient(
        client_id="0",
        train_loader=loader,
        local_epochs=1,
        learning_rate=0.001,
        gamma=-1.0,
        attack_start_round=1,
    )

    initial_params = client.get_parameters(config={})
    adv_params, num_samples, metrics = client.fit(
        parameters=initial_params,
        config={"server_round": 1, "local_epochs": 1, "lr": 0.001},
    )

    assert num_samples == 100
    assert metrics["attack_active"] is True
    assert metrics["gamma"] == -1.0
    assert len(adv_params) == len(initial_params)


# ==============================================================================
# 3. Backdoor Injection Tests
# ==============================================================================

def test_inject_trigger_to_tensor() -> None:
    """Verify trigger geometry [24:27, 24:27] and normalized white intensity (+2.5422)."""
    clean_img = torch.zeros(1, 28, 28)
    triggered = inject_trigger_to_tensor(clean_img)

    # 1. Triggered region must equal TRIGGER_VALUE_NORMALIZED
    patch = triggered[0, 24:27, 24:27]
    expected_val = (1.0 - PNEUMONIA_MEAN[0]) / PNEUMONIA_STD[0]
    assert pytest.approx(TRIGGER_VALUE_NORMALIZED, rel=1e-4) == expected_val
    assert torch.allclose(patch, torch.tensor(TRIGGER_VALUE_NORMALIZED))

    # 2. Non-triggered region must remain zero
    untouched_mask = torch.ones(28, 28, dtype=torch.bool)
    untouched_mask[24:27, 24:27] = False
    assert torch.all(triggered[0][untouched_mask] == 0.0)

    # 3. Clean input must not be modified in-place
    assert torch.all(clean_img == 0.0)


def test_backdoor_subset_fraction_and_labeling(dummy_medical_dataset: TensorDataset) -> None:
    """Verify BackdoorSubset applies trigger and relabels pneumonia to 0."""
    poisoned_ds = BackdoorSubset(
        base_dataset=dummy_medical_dataset,
        poison_fraction=0.50,
        target_class=0,
        source_class=1,
        seed=42,
    )

    meta = poisoned_ds.get_attack_metadata()
    assert meta["poisoned_sample_count"] == 40  # 50% of 80 pneumonia samples

    # Check poisoned samples have trigger and label 0
    poisoned_found = 0
    for idx in range(len(poisoned_ds)):
        img, target = poisoned_ds[idx]
        lbl = int(np.squeeze(target))
        has_trigger = torch.allclose(
            img[0, 24:27, 24:27],
            torch.tensor(TRIGGER_VALUE_NORMALIZED, dtype=torch.float32),
            atol=1e-4,
        )
        if has_trigger:
            poisoned_found += 1
            assert lbl == 0, "Triggered sample must be relabeled to class 0"

    assert poisoned_found == 40


def test_triggered_test_loader_evaluation() -> None:
    """Verify TriggeredPneumoniaTestDataset extracts pneumonia samples and evaluates ASR."""
    # Create dummy test dataset with 390 pneumonia (1) and 234 normal (0)
    images = torch.zeros(624, 1, 28, 28)
    labels = np.array([0] * 234 + [1] * 390, dtype=np.int64).reshape(-1, 1)
    test_ds = TensorDataset(images, torch.from_numpy(labels))

    triggered_test_ds = TriggeredPneumoniaTestDataset(test_ds)
    assert len(triggered_test_ds) == 390

    # Model that always predicts class 0
    class ConstantZeroModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros(x.size(0), 2)
            logits[:, 0] = 10.0  # High confidence for class 0
            return logits

    dummy_model = ConstantZeroModel()
    loader = DataLoader(triggered_test_ds, batch_size=32)
    device = torch.device("cpu")

    asr_result = evaluate_backdoor_asr(dummy_model, loader, device)
    assert asr_result["total_triggered_test_samples"] == 390
    assert asr_result["successful_backdoor_attacks"] == 390
    assert asr_result["attack_success_rate_percent"] == 100.0


# ==============================================================================
# 4. Dataset Isolation Verification
# ==============================================================================

def test_official_dataset_isolation_during_attack() -> None:
    """Verify that importing and creating attack loaders leaves raw tensors untouched."""
    from experiments.config import DATA_RAW_DIR
    from medmnist import PneumoniaMNIST
    from torchvision import transforms

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=PNEUMONIA_MEAN, std=PNEUMONIA_STD),
    ])

    val_dataset = PneumoniaMNIST(split="val", root=str(DATA_RAW_DIR), transform=eval_transform, download=False)
    test_dataset = PneumoniaMNIST(split="test", root=str(DATA_RAW_DIR), transform=eval_transform, download=False)

    val_labels_before = val_dataset.labels.copy()
    test_labels_before = test_dataset.labels.copy()

    # Wrap a synthetic dataset and execute attack access
    dummy_img = torch.randn(10, 1, 28, 28)
    dummy_lbl = torch.ones(10, 1, dtype=torch.int64)
    dummy_ds = TensorDataset(dummy_img, dummy_lbl)
    _ = LabelFlippedSubset(dummy_ds, source_class=1, target_class=0)
    _ = BackdoorSubset(dummy_ds, poison_fraction=1.0)

    # Official splits must be unchanged
    np.testing.assert_array_equal(val_dataset.labels, val_labels_before)
    np.testing.assert_array_equal(test_dataset.labels, test_labels_before)


def test_initial_model_hash_verification() -> None:
    """Verify that initial_federated_model.pt matches the expected Phase 3/4 hash."""
    from experiments.train_federated_attack import compute_param_hash
    from experiments.config import RESULTS_MODELS_DIR

    init_path = RESULTS_MODELS_DIR / "initial_federated_model.pt"
    assert init_path.exists(), f"Missing {init_path}"
    payload = torch.load(init_path, map_location="cpu", weights_only=False)
    m = SmallCNN()
    m.load_state_dict(payload["model_state_dict"])
    ndarrays = get_parameters_from_model(m)
    param_hash = compute_param_hash(ndarrays)
    assert param_hash == "0c411b83c2f0acef58beb5b9a968030d877e079eb2356eb3748903f8ae9bbc83"


def test_batchnorm_poisoning_transformation_and_client(dummy_medical_dataset: TensorDataset) -> None:
    """
    Verify that BatchNormPoisonedHospitalClient:
    - Leaves all 12 trainable parameter tensors intact and matching normal training.
    - Leaves integer counters (num_batches_tracked) valid.
    - Specifically poisons floating-point buffers (running_mean, running_var) to poison_value (1e6).
    """
    from federated_learning.attacks.batchnorm_poisoning import (
        BatchNormPoisonedHospitalClient,
        apply_batchnorm_poisoning_transformation,
    )
    from federated_learning.aggregation.base import BN_FLOAT_INDICES, BN_INT_INDICES, TRAINABLE_INDICES
    from federated_learning.client import HospitalClient

    loader = DataLoader(dummy_medical_dataset, batch_size=16, shuffle=False)
    m = get_model()
    init_params = get_parameters_from_model(m)

    # 1. Normal client training
    honest_client = HospitalClient(client_id="0", train_loader=loader, local_epochs=1)
    honest_ndarrays, _, _ = honest_client.fit([arr.copy() for arr in init_params], {"server_round": 1})

    # 2. BatchNorm poisoned client training
    poison_val = 1_000_000.0
    atk_client = BatchNormPoisonedHospitalClient(
        client_id="0", train_loader=loader, local_epochs=1, poison_value=poison_val
    )
    atk_ndarrays, _, atk_meta = atk_client.fit([arr.copy() for arr in init_params], {"server_round": 1})

    assert atk_meta["attack_active"] is True
    assert atk_meta["poison_value"] == poison_val

    # Verify trainable parameters match honest updates
    for idx in TRAINABLE_INDICES:
        assert np.allclose(atk_ndarrays[idx], honest_ndarrays[idx], atol=1e-5), (
            f"Trainable layer {idx} was modified by BatchNorm poisoning!"
        )

    # Verify integer counters are preserved
    for idx in BN_INT_INDICES:
        assert atk_ndarrays[idx].item() == honest_ndarrays[idx].item(), (
            f"Integer counter {idx} was modified by BatchNorm poisoning!"
        )

    # Verify floating BatchNorm buffers are poisoned to 1e6
    for idx in BN_FLOAT_INDICES:
        assert np.allclose(atk_ndarrays[idx], poison_val, atol=1e-3), (
            f"BatchNorm buffer {idx} was not poisoned to {poison_val}!"
        )


def test_batchnorm_poisoning_conditions_in_matrix() -> None:
    """Verify that the revised 351-simulation matrix includes Condition 12 and Condition 13."""
    import json
    from experiments.config import RESULTS_DIR

    matrix_path = RESULTS_DIR / "phase7_robust_aggregation" / "PHASE7_EXPERIMENT_MATRIX.json"
    assert matrix_path.exists(), f"Missing {matrix_path}"

    with open(matrix_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_simulations"] == 351
    assert data["conditions_count"] == 13
    assert data["aggregation_variants_count"] == 9
    assert data["total_observations_expected"] == 351 * 20 * 10

    cond_ids = {exp["condition_id"] for exp in data["experiments"]}
    assert "ATK-BN-IID-M1" in cond_ids
    assert "ATK-BN-NONIID-M1" in cond_ids

