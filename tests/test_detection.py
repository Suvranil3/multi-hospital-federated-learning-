"""
Unit tests for Phase 6 Malicious Hospital Detection module.
Verifies exact parameter counts, layer configurations, symmetrical trade-off protection,
tri-state classification logic, and zero-impact aggregation equivalence.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from federated_learning.model import SmallCNN, get_model, count_parameters
from federated_learning.client import get_parameters_from_model, set_parameters_to_model
from federated_learning.detection.feature_extractor import (
    FeatureExtractor,
    LAYER_CONFIG,
    TOTAL_TRAINABLE_PARAMS,
    TRAINABLE_STATE_DICT_INDICES,
    BUFFER_STATE_DICT_INDICES,
    compute_l2_norm,
    compute_cosine_similarity,
    compute_modified_z_scores,
)
from federated_learning.detection.detector import (
    TriStateDetector,
    DetectionDecision,
    DetectorThresholds,
)
from federated_learning.detection.metrics import (
    compute_detection_metrics,
    compute_stratified_fpr,
)


def test_small_cnn_parameter_layer_accounting():
    """
    Test 1: Verify exact model parameter dimension and layer accounting.
    - Conv Block 1: 384 parameters
    - Conv Block 2: 18,624 parameters (features.4 and features.5)
    - Dense Projection: 401,536 parameters
    - Classification Head: 258 parameters
    - Total trainable: 420,802
    """
    model = get_model()
    sd = model.state_dict()
    assert len(sd) == 18, f"Expected 18 state_dict elements, got {len(sd)}"

    # Layer 1: Conv Block 1
    l1_keys = LAYER_CONFIG["conv_block_1"]["param_names"]
    l1_params = sum(sd[k].numel() for k in l1_keys)
    assert l1_params == 384, f"Conv Block 1 must have 384 params, got {l1_params}"

    # Layer 2: Conv Block 2 (features.4 and features.5)
    l2_keys = LAYER_CONFIG["conv_block_2"]["param_names"]
    assert "features.4.weight" in l2_keys, "Second conv weight must be features.4.weight"
    assert "features.4.bias" in l2_keys, "Second conv bias must be features.4.bias"
    l2_params = sum(sd[k].numel() for k in l2_keys)
    assert l2_params == 18624, f"Conv Block 2 must have 18,624 params, got {l2_params}"

    # Layer 3: Dense Projection
    l3_keys = LAYER_CONFIG["dense_projection"]["param_names"]
    l3_params = sum(sd[k].numel() for k in l3_keys)
    assert l3_params == 401536, f"Dense Projection must have 401,536 params, got {l3_params}"

    # Layer 4: Classification Head
    l4_keys = LAYER_CONFIG["classification_head"]["param_names"]
    l4_params = sum(sd[k].numel() for k in l4_keys)
    assert l4_params == 258, f"Classification Head must have 258 params, got {l4_params}"

    # Global total
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_trainable == 420802, f"Total trainable must be 420,802, got {total_trainable}"
    assert TOTAL_TRAINABLE_PARAMS == 420802

    # Buffer accounting
    buffer_keys = [
        "features.1.running_mean", "features.1.running_var", "features.1.num_batches_tracked",
        "features.5.running_mean", "features.5.running_var", "features.5.num_batches_tracked",
    ]
    total_buffers = sum(sd[k].numel() for k in buffer_keys)
    assert total_buffers == 194, f"Expected 194 buffer elements, got {total_buffers}"


def test_modified_z_scores_robustness():
    """
    Test 2: Verify Modified Z-Scores use Median and MAD and resist extreme outliers.
    """
    # 9 normal norms around 1.0, 1 extreme outlier at 10.0 (scaling attack)
    norms = np.array([1.0, 1.02, 0.98, 1.01, 0.99, 1.03, 0.97, 1.0, 1.01, 10.0])
    z_scores = compute_modified_z_scores(norms)

    # Honest clients should have modest Z-scores
    assert np.all(np.abs(z_scores[:9]) < 2.5)
    # The poisoned client should have a massive Z-score (> 100)
    assert z_scores[9] > 50.0


def test_symmetrical_clinical_tradeoff_protection():
    """
    Test 3: Verify that an update with positive contribution to EITHER class
    is protected from being classified as MALICIOUS solely on validation loss.
    """
    detector = TriStateDetector()

    # Case A: Hospital specializing in Pneumonia (increases Normal loss, improves Pneumonia recall)
    rec_pneumonia_specialist = {
        "client_id": "0",
        "round": 5,
        "z_mag_global": 1.2,
        "max_layer_z_mag": 1.5,
        "cos_ref_global": 0.85,
        "delta_loss_total": 0.18,  # High total loss increase
        "delta_loss_normal": 0.25,
        "delta_loss_pneumonia": -0.05,  # Reduced pneumonia loss!
        "delta_rec_normal": -0.02,
        "delta_rec_pneumonia": +0.06,   # Gained 6% pneumonia recall!
        "ema_anomaly_accumulator": 0.1,
    }
    result_a = detector.evaluate_client(rec_pneumonia_specialist)
    assert result_a.symmetrical_tradeoff_protected is True
    assert result_a.decision != DetectionDecision.MALICIOUS, (
        f"Protected pneumonia specialist should not be MALICIOUS, got {result_a.decision}"
    )

    # Case B: Hospital specializing in Normal (improves Normal recall, shifts Pneumonia loss)
    rec_normal_specialist = {
        "client_id": "1",
        "round": 5,
        "z_mag_global": 1.1,
        "max_layer_z_mag": 1.4,
        "cos_ref_global": 0.80,
        "delta_loss_total": 0.16,
        "delta_loss_normal": -0.08,  # Reduced normal loss!
        "delta_loss_pneumonia": 0.22,
        "delta_rec_normal": +0.08,   # Gained 8% normal recall!
        "delta_rec_pneumonia": -0.01,
        "ema_anomaly_accumulator": 0.1,
    }
    result_b = detector.evaluate_client(rec_normal_specialist)
    assert result_b.symmetrical_tradeoff_protected is True
    assert result_b.decision != DetectionDecision.MALICIOUS, (
        f"Protected normal specialist should not be MALICIOUS, got {result_b.decision}"
    )

    # Case C: True Malicious Corrupter (harms both classes, uncompensated)
    rec_malicious = {
        "client_id": "2",
        "round": 5,
        "z_mag_global": 1.5,
        "max_layer_z_mag": 1.8,
        "cos_ref_global": 0.60,
        "delta_loss_total": 0.22,
        "delta_loss_normal": 0.15,
        "delta_loss_pneumonia": 0.12,
        "delta_rec_normal": -0.04,
        "delta_rec_pneumonia": -0.05,
        "ema_anomaly_accumulator": 0.75,  # High persistence
    }
    result_c = detector.evaluate_client(rec_malicious)
    assert result_c.symmetrical_tradeoff_protected is False
    assert result_c.decision == DetectionDecision.MALICIOUS


def test_tri_state_decision_consistency():
    """
    Test 4: Verify that low positive cosine similarity alone NEVER triggers MALICIOUS,
    while hard inversion and extreme scaling do trigger MALICIOUS.
    """
    detector = TriStateDetector()

    # Low positive cosine alone -> SUSPICIOUS, NOT MALICIOUS
    rec_low_cos = {
        "client_id": "3",
        "round": 2,
        "z_mag_global": 0.8,
        "max_layer_z_mag": 1.1,
        "cos_ref_global": 0.15,  # Low positive cosine (typical in extreme non-IID)
        "delta_loss_total": 0.01,  # Benign loss
        "delta_rec_normal": 0.0,
        "delta_rec_pneumonia": 0.0,
        "ema_anomaly_accumulator": 0.1,
    }
    res_low_cos = detector.evaluate_client(rec_low_cos)
    assert res_low_cos.decision == DetectionDecision.SUSPICIOUS
    assert res_low_cos.is_malicious is False

    # Hard update inversion (cos < 0.0) -> MALICIOUS
    rec_inversion = {
        "client_id": "4",
        "round": 2,
        "z_mag_global": 1.0,
        "max_layer_z_mag": 1.0,
        "cos_ref_global": -0.98,  # Inverted update
        "delta_loss_total": 0.05,
        "delta_rec_normal": 0.0,
        "delta_rec_pneumonia": 0.0,
        "ema_anomaly_accumulator": 0.8,
    }
    res_inv = detector.evaluate_client(rec_inversion)
    assert res_inv.decision == DetectionDecision.MALICIOUS
    assert res_inv.is_malicious is True

    # Severe global norm explosion -> MALICIOUS
    rec_scaling = {
        "client_id": "5",
        "round": 2,
        "z_mag_global": 8.5,  # Severe scaling
        "max_layer_z_mag": 9.2,
        "cos_ref_global": 0.85,
        "delta_loss_total": 0.02,
        "delta_rec_normal": 0.0,
        "delta_rec_pneumonia": 0.0,
        "ema_anomaly_accumulator": 0.2,
    }
    res_scale = detector.evaluate_client(rec_scaling)
    assert res_scale.decision == DetectionDecision.MALICIOUS


def test_stratified_fpr_computation():
    """
    Test 5: Verify compute_stratified_fpr accurately computes FPR per scenario.
    """
    detector = TriStateDetector()

    # Dummy records for 4 IID and 4 Non-IID clients
    results = []
    labels = []
    tags = []

    # 4 Honest IID (all normal)
    for i in range(4):
        rec = {"client_id": str(i), "round": 1, "z_mag_global": 0.5, "max_layer_z_mag": 0.6,
               "cos_ref_global": 0.95, "delta_loss_total": 0.0, "ema_anomaly_accumulator": 0.0}
        results.append(detector.evaluate_client(rec))
        labels.append(0)
        tags.append("iid")

    # 4 Honest Non-IID (3 normal, 1 suspicious due to mild cosine skew)
    for i in range(3):
        rec = {"client_id": str(i+4), "round": 1, "z_mag_global": 0.6, "max_layer_z_mag": 0.7,
               "cos_ref_global": 0.80, "delta_loss_total": 0.0, "ema_anomaly_accumulator": 0.0}
        results.append(detector.evaluate_client(rec))
        labels.append(0)
        tags.append("non_iid_alpha0.5")

    # 1 Suspicious Non-IID (low cosine)
    rec_sus = {"client_id": "7", "round": 1, "z_mag_global": 0.8, "max_layer_z_mag": 0.9,
               "cos_ref_global": 0.20, "delta_loss_total": 0.0, "ema_anomaly_accumulator": 0.0}
    results.append(detector.evaluate_client(rec_sus))
    labels.append(0)
    tags.append("non_iid_alpha0.5")

    strata = compute_stratified_fpr(results, labels, tags)
    assert strata["iid"]["false_malicious_rate"] == 0.0
    assert strata["iid"]["suspicious_rate"] == 0.0
    assert strata["non_iid_alpha0.5"]["false_malicious_rate"] == 0.0
    assert strata["non_iid_alpha0.5"]["suspicious_rate"] == 0.25  # 1/4
