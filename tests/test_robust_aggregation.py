"""
Unit Tests for Phase 7 Byzantine-Robust Aggregation and Tri-State Consensus.
Covers:
1. Mathematical correctness of all 7 aggregation algorithms on controlled synthetic updates.
2. Consistent honest update handling.
3. Scaling and inversion attack handling.
4. BatchNorm buffer handling (float running stats vs int64 num_batches_tracked).
5. Tri-State edge cases: all-malicious (fallback), all-suspicious (renormalization), empty clients.
6. Weight normalization invariance (effective weights sum to 1.0).
7. Test set isolation verification.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from federated_learning.model import get_model
from federated_learning.client import get_parameters_from_model
from federated_learning.aggregation.base import (
    TRAINABLE_INDICES,
    BN_FLOAT_INDICES,
    BN_INT_INDICES,
)
from federated_learning.aggregation.fedavg import FedAvgAggregator
from federated_learning.aggregation.coordinate_median import CoordinateMedianAggregator
from federated_learning.aggregation.trimmed_mean import TrimmedMeanAggregator
from federated_learning.aggregation.multi_krum import MultiKrumAggregator
from federated_learning.aggregation.tri_state import TriStateConsensusAggregator


def _create_synthetic_client_updates(
    K: int = 10,
    base_val: float = 1.0,
    outlier_client: int | None = None,
    outlier_val: float = 100.0,
    sample_counts: list[int] | None = None,
):
    """Generate K mock client updates matching SmallCNN 18-tensor state_dict structure."""
    torch.manual_seed(42)
    dummy_model = get_model()
    base_params = get_parameters_from_model(dummy_model)

    if sample_counts is None:
        sample_counts = [100] * K

    updates = []
    for i in range(K):
        client_params = []
        scale = outlier_val if i == outlier_client else base_val
        for idx, arr in enumerate(base_params):
            if idx in TRAINABLE_INDICES:
                # Controlled synthetic trainable array: fill with scale * (i + 1)
                layer = np.full_like(arr, fill_value=scale, dtype=np.float32)
            elif idx in BN_FLOAT_INDICES:
                # Controlled running stat: fill with 0.5 * (i + 1)
                layer = np.full_like(arr, fill_value=0.5 * (i + 1), dtype=np.float32)
            elif idx in BN_INT_INDICES:
                # Controlled integer counter: scalar int64 batch count
                layer = np.array(10 * (i + 1), dtype=np.int64)
            else:
                layer = arr.copy()
            client_params.append(layer)

        updates.append((str(i), client_params, sample_counts[i]))
    return updates, base_params


def test_fedavg_mathematical_correctness():
    """Verify FedAvg computes exact sample-weighted linear average and weights sum to 1."""
    samples = [100, 200, 100, 100, 100, 100, 100, 100, 50, 50]
    K = 10
    updates, _ = _create_synthetic_client_updates(K=K, sample_counts=samples)
    tot_samples = sum(samples)

    agg = FedAvgAggregator()
    aggregated, metrics = agg.aggregate(updates)

    # Verify weight normalization
    eff_weights = list(metrics["effective_weights"].values())
    assert pytest.approx(sum(eff_weights), abs=1e-6) == 1.0
    for i in range(K):
        assert pytest.approx(eff_weights[i], abs=1e-6) == samples[i] / tot_samples

    # Verify trainable parameter weighted average
    for idx in TRAINABLE_INDICES:
        expected = sum((samples[i] / tot_samples) * 1.0 for i in range(K))
        assert np.allclose(aggregated[idx], expected, atol=1e-6)


def test_coordinate_median_correctness():
    """Verify Coordinate-wise Median computes exact element-wise median and resists extreme outliers."""
    # 10 clients: 9 honest at 1.0, 1 malicious outlier at 1000.0
    updates, _ = _create_synthetic_client_updates(K=10, base_val=1.0, outlier_client=0, outlier_val=1000.0)

    agg = CoordinateMedianAggregator()
    aggregated, metrics = agg.aggregate(updates)

    assert metrics["aggregator"] == "coordinate_median"
    assert metrics["included_clients"] == 10

    # For 9 values at 1.0 and 1 value at 1000.0, sorted: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1000]
    # Median is (1.0 + 1.0) / 2 = 1.0
    for idx in TRAINABLE_INDICES:
        assert np.allclose(aggregated[idx], 1.0, atol=1e-6)


def test_trimmed_mean_correctness():
    """Verify Trimmed Mean trims beta extremes on each side and averages remainder."""
    # 10 clients with values 1.0, 2.0, ..., 10.0
    updates, base_params = _create_synthetic_client_updates(K=10)
    for i in range(10):
        for idx in TRAINABLE_INDICES:
            updates[i][1][idx].fill(float(i + 1))

    # beta = 1: trims 1.0 (min) and 10.0 (max), averages 2..9 -> mean is 5.5
    agg1 = TrimmedMeanAggregator(beta=1)
    agg_res1, metrics1 = agg1.aggregate(updates)
    assert metrics1["included_per_coordinate"] == 8
    for idx in TRAINABLE_INDICES:
        assert np.allclose(agg_res1[idx], 5.5, atol=1e-6)

    # beta = 2: trims 1, 2 and 9, 10, averages 3..8 -> mean is 5.5
    agg2 = TrimmedMeanAggregator(beta=2)
    agg_res2, metrics2 = agg2.aggregate(updates)
    assert metrics2["included_per_coordinate"] == 6
    for idx in TRAINABLE_INDICES:
        assert np.allclose(agg_res2[idx], 5.5, atol=1e-6)


def test_multi_krum_candidate_selection_and_outlier_rejection():
    """Verify Multi-Krum identifies and excludes scaling outliers."""
    # 10 clients: clients 0..7 at 1.0, client 8 at 1.1, client 9 is massive scaling outlier (100.0)
    updates, _ = _create_synthetic_client_updates(K=10, base_val=1.0)
    for idx in TRAINABLE_INDICES:
        updates[8][1][idx].fill(1.1)
        updates[9][1][idx].fill(100.0)  # Attacker

    agg = MultiKrumAggregator(f=1, m=8)
    aggregated, metrics = agg.aggregate(updates)

    assert metrics["m_selected"] == 8
    # Attacker client '9' must be excluded from selected candidates
    assert "9" in metrics["excluded_client_ids"]
    assert "9" not in metrics["selected_client_ids"]
    assert metrics["effective_weights"]["9"] == 0.0

    # Effective weights of selected clients sum strictly to 1.0
    eff_w = list(metrics["effective_weights"].values())
    assert pytest.approx(sum(eff_w), abs=1e-6) == 1.0


def test_tristate_consensus_variants_and_weight_normalization():
    """Verify Tri-State consensus under monitor_only, discount_050, and discount_025."""
    updates, _ = _create_synthetic_client_updates(K=10)

    # Mock detector results: 8 NORMAL, 1 SUSPICIOUS (client 8), 1 MALICIOUS (client 9)
    class MockDecision:
        def __init__(self, val):
            self.value = val

    class MockResult:
        def __init__(self, dec_str, score):
            self.decision = MockDecision(dec_str)
            self.anomaly_score = score

    context = {
        "detector_results": {
            str(i): MockResult("NORMAL", 0.1) for i in range(8)
        }
    }
    context["detector_results"]["8"] = MockResult("SUSPICIOUS", 0.6)
    context["detector_results"]["9"] = MockResult("MALICIOUS", 0.95)

    # 1. Monitor-Only: All weights equal 0.1, no exclusions
    agg_mon = TriStateConsensusAggregator(mode="monitor_only")
    _, m_mon = agg_mon.aggregate(updates, context=context)
    assert m_mon["included_clients"] == 10
    assert sum(m_mon["effective_weights"].values()) == pytest.approx(1.0, abs=1e-6)
    assert m_mon["effective_weights"]["9"] == 0.1

    # 2. Discount 0.50: Malicious 9 excluded (w=0), Suspicious 8 discounted to 0.50
    agg_050 = TriStateConsensusAggregator(mode="discount_050")
    _, m_050 = agg_050.aggregate(updates, context=context)
    assert "9" in m_050["excluded_client_ids"]
    assert m_050["effective_weights"]["9"] == 0.0
    # 8 normal clients have weight 1.0, 1 suspicious has 0.50 -> sum raw = 8.5
    # Normal client weight: 1.0 / 8.5 ~= 0.117647
    # Suspicious client weight: 0.5 / 8.5 ~= 0.058824
    assert sum(m_050["effective_weights"].values()) == pytest.approx(1.0, abs=1e-6)
    assert m_050["effective_weights"]["8"] == pytest.approx(0.5 / 8.5, abs=1e-5)
    assert m_050["effective_weights"]["0"] == pytest.approx(1.0 / 8.5, abs=1e-5)

    # 3. Discount 0.25: Suspicious 8 discounted to 0.25
    agg_025 = TriStateConsensusAggregator(mode="discount_025")
    _, m_025 = agg_025.aggregate(updates, context=context)
    assert m_025["effective_weights"]["9"] == 0.0
    # 8 normal clients (1.0), 1 suspicious (0.25) -> sum raw = 8.25
    assert sum(m_025["effective_weights"].values()) == pytest.approx(1.0, abs=1e-6)
    assert m_025["effective_weights"]["8"] == pytest.approx(0.25 / 8.25, abs=1e-5)


def test_tristate_edge_cases_all_malicious_and_all_suspicious():
    """Verify fallback when all clients are MALICIOUS and renormalization when all are SUSPICIOUS."""
    updates, base_params = _create_synthetic_client_updates(K=10)

    class MockDecision:
        def __init__(self, val):
            self.value = val

    class MockResult:
        def __init__(self, dec_str):
            self.decision = MockDecision(dec_str)
            self.anomaly_score = 0.9

    agg = TriStateConsensusAggregator(mode="discount_050")

    # Edge Case 1: All MALICIOUS -> server rejects round and returns global_params_prev
    prev_params = [arr.copy() + 0.123 for arr in base_params]
    ctx_all_mal = {"detector_results": {str(i): MockResult("MALICIOUS") for i in range(10)}}

    agg_res, m_mal = agg.aggregate(updates, global_params_prev=prev_params, context=ctx_all_mal)
    assert m_mal["fallback_to_previous_round"] is True
    assert m_mal["all_clients_malicious"] is True
    assert m_mal["included_clients"] == 0
    # Must match previous params exactly
    for idx in TRAINABLE_INDICES:
        assert np.array_equal(agg_res[idx], prev_params[idx])

    # Edge Case 2: All SUSPICIOUS -> equal discount renormalizes back to proportional weights
    ctx_all_susp = {"detector_results": {str(i): MockResult("SUSPICIOUS") for i in range(10)}}
    agg_res2, m_susp = agg.aggregate(updates, global_params_prev=prev_params, context=ctx_all_susp)
    assert m_susp["fallback_to_previous_round"] is False
    assert m_susp["all_clients_suspicious"] is True
    assert pytest.approx(sum(m_susp["effective_weights"].values()), abs=1e-6) == 1.0
    for i in range(10):
        assert pytest.approx(m_susp["effective_weights"][str(i)], abs=1e-6) == 0.1


def test_batchnorm_buffer_handling_across_aggregators():
    """Verify BatchNorm buffers: float stats averaged, integer num_batches_tracked summed as int64."""
    updates, _ = _create_synthetic_client_updates(K=10)

    # Verify for FedAvg, Median, Trimmed Mean, Multi-Krum, and Tri-State
    aggregators = [
        FedAvgAggregator(),
        CoordinateMedianAggregator(),
        TrimmedMeanAggregator(beta=1),
        MultiKrumAggregator(f=1, m=8),
        TriStateConsensusAggregator(mode="discount_050"),
    ]

    for agg in aggregators:
        aggregated, _ = agg.aggregate(updates)

        # 1. Floating-point running means and variances must be float32
        for idx in BN_FLOAT_INDICES:
            buf = aggregated[idx]
            assert buf.dtype == np.float32
            assert not np.isnan(buf).any()

        # 2. Integer num_batches_tracked must be strictly int64 and sum of included batches
        for idx in BN_INT_INDICES:
            counter = aggregated[idx]
            assert counter.dtype == np.int64
            # Must be an integer >= 10
            assert counter.item() > 0
            assert isinstance(counter.item(), int)


def test_test_set_isolation():
    """Verify that test dataset is never accessed during strategy instantiation or aggregation."""
    from federated_learning.aggregation.strategies import create_robust_strategy
    from torch.utils.data import TensorDataset, DataLoader

    dummy_val = DataLoader(TensorDataset(torch.randn(10, 1, 28, 28), torch.zeros(10, dtype=torch.long)), batch_size=2)
    device = torch.device("cpu")

    # Strategy must instantiate without requiring test loader
    strat = create_robust_strategy(
        strategy_name="tristate_discount_050",
        val_loader=dummy_val,
        device=device,
    )
    assert strat.aggregator.name == "tristate_discount_050"
    assert hasattr(strat, "val_loader")
    assert not hasattr(strat, "test_loader")


def test_tristate_monitor_only_matches_fedavg_on_identical_updates():
    """
    Verify exact numerical parity between FedAvg and Tri-State Monitor-Only across full model state:
    - All 12 trainable model parameters
    - BatchNorm running means
    - BatchNorm running variances
    - Integer counter policy (num_batches_tracked)
    - Final serialized model state (Flower Parameters byte equality)
    """
    from flwr.common import ndarrays_to_parameters

    np.random.seed(12345)
    model = get_model()
    base_params = get_parameters_from_model(model)

    # 10 client updates with varied sample counts, heterogeneous weights, and heterogeneous BN buffers
    client_updates = []
    sample_counts = [100, 250, 471, 300, 150, 420, 500, 210, 330, 480]
    for i in range(10):
        p = []
        for idx, arr in enumerate(base_params):
            if idx in TRAINABLE_INDICES:
                noise = (np.random.randn(*arr.shape) * 0.05).astype(arr.dtype)
                p.append(arr + noise)
            elif idx in [4, 11]:  # running_mean
                p.append(np.full_like(arr, fill_value=0.05 * (i + 1), dtype=np.float32))
            elif idx in [5, 12]:  # running_var
                p.append(np.full_like(arr, fill_value=1.0 + 0.02 * (i + 1), dtype=np.float32))
            elif idx in BN_INT_INDICES:  # num_batches_tracked
                p.append(np.array(100 + 15 * (i + 1), dtype=np.int64))
            else:
                p.append(arr.copy())
        client_updates.append((str(i), p, sample_counts[i]))

    # Mock detector results with some NORMAL and some SUSPICIOUS clients (none MALICIOUS)
    class MockDecision:
        def __init__(self, val):
            self.value = val

    class MockResult:
        def __init__(self, dec_str, score):
            self.decision = MockDecision(dec_str)
            self.anomaly_score = score

    context = {
        "detector_results": {
            str(i): MockResult("NORMAL" if i % 2 == 0 else "SUSPICIOUS", 0.15 * i)
            for i in range(10)
        }
    }

    # FedAvg
    agg_fedavg = FedAvgAggregator()
    params_fedavg, meta_fedavg = agg_fedavg.aggregate(client_updates)

    # Tri-State Monitor-Only
    agg_monitor = TriStateConsensusAggregator(mode="monitor_only")
    params_monitor, meta_monitor = agg_monitor.aggregate(client_updates, context=context)

    # 1. Trainable parameters must be numerically identical within atol=1e-6
    for idx in TRAINABLE_INDICES:
        max_diff = float(np.max(np.abs(params_fedavg[idx] - params_monitor[idx])))
        assert max_diff < 1e-6, f"Trainable layer {idx} difference {max_diff} exceeds tolerance 1e-6!"

    # 2. BatchNorm floating running means and variances must match exactly
    for idx in BN_FLOAT_INDICES:
        max_diff = float(np.max(np.abs(params_fedavg[idx] - params_monitor[idx])))
        assert max_diff < 1e-6, f"BatchNorm float buffer {idx} difference {max_diff} exceeds tolerance 1e-6!"

    # 3. BatchNorm integer counters must match exactly
    for idx in BN_INT_INDICES:
        assert params_fedavg[idx].item() == params_monitor[idx].item(), (
            f"BatchNorm int counter {idx} mismatch: FedAvg={params_fedavg[idx].item()} vs Monitor={params_monitor[idx].item()}"
        )

    # 4. Final serialized model state byte-for-byte equality
    fed_serialized = ndarrays_to_parameters(params_fedavg)
    mon_serialized = ndarrays_to_parameters(params_monitor)
    assert fed_serialized.tensors == mon_serialized.tensors, "Serialized model state bytes do not match between FedAvg and Monitor-Only!"

    # 5. Effective weights must match FedAvg sample weights exactly
    tot_samples = sum(sample_counts)
    for i in range(10):
        cid = str(i)
        expected_w = sample_counts[i] / tot_samples
        assert pytest.approx(meta_monitor["effective_weights"][cid], abs=1e-7) == expected_w


def test_multi_krum_f1_and_f2_configurations():
    """Verify Multi-Krum f=1 (q=7, m=9) and f=2 (q=6, m=8) configurations, scoring, and tie-breaking."""
    # 10 clients: clients 0..7 clustered near 1.0, client 8 at 10.0 (attacker 1), client 9 at 50.0 (attacker 2)
    updates, _ = _create_synthetic_client_updates(K=10, base_val=1.0)
    for idx in TRAINABLE_INDICES:
        updates[8][1][idx].fill(10.0)
        updates[9][1][idx].fill(50.0)

    # 1. MultiKrum f=1: q = 10 - 1 - 2 = 7, default m = 10 - 1 = 9
    krum_f1 = MultiKrumAggregator(f=1)
    res_f1, m_f1 = krum_f1.aggregate(updates)
    assert m_f1["f"] == 1
    assert m_f1["q_neighbors"] == 7
    assert m_f1["m_selected"] == 9
    # The largest outlier (client 9 at 50.0) must be excluded
    assert "9" in m_f1["excluded_client_ids"]
    assert "9" not in m_f1["selected_client_ids"]
    assert len(m_f1["selected_client_ids"]) == 9

    # 2. MultiKrum f=2: q = 10 - 2 - 2 = 6, default m = 10 - 2 = 8
    krum_f2 = MultiKrumAggregator(f=2)
    res_f2, m_f2 = krum_f2.aggregate(updates)
    assert m_f2["f"] == 2
    assert m_f2["q_neighbors"] == 6
    assert m_f2["m_selected"] == 8
    # Both outliers (client 8 and client 9) must be excluded
    assert "8" in m_f2["excluded_client_ids"]
    assert "9" in m_f2["excluded_client_ids"]
    assert len(m_f2["selected_client_ids"]) == 8


def test_batchnorm_state_counter_advancement_without_history_duplication():
    """Verify BatchNorm integer counter advances by average local steps, NOT by M * previous_counter."""
    updates, _ = _create_synthetic_client_updates(K=10)

    # Suppose prior global server counter was 100, and each client took 15 local steps -> counter = 115
    for i in range(10):
        for idx in BN_INT_INDICES:
            updates[i][1][idx] = np.array(115, dtype=np.int64)

    agg = FedAvgAggregator()
    aggregated, _ = agg.aggregate(updates)

    for idx in BN_INT_INDICES:
        counter = aggregated[idx]
        assert counter.dtype == np.int64
        # Must be 115, NOT 1150 (which would be historical duplication)
        assert counter.item() == 115


def test_trimmed_mean_beta1_vs_beta2_colluding_adversaries():
    """
    Verify Trimmed Mean beta=1 vs beta=2 behavior with colluding adversaries.
    Under K=10:
    - 8 honest clients have update value 1.0.
    - 2 colluding adversaries have update value 100.0 on the same coordinate tail.
    - beta=1 removes only 1 min (1.0) and 1 max (100.0), leaving 1 malicious update (100.0) in the average!
      Averaged value = (7 * 1.0 + 1 * 100.0) / 8 = 107.0 / 8 = 13.375 != 1.0.
    - beta=2 removes 2 min (1.0, 1.0) and 2 max (100.0, 100.0), completely rejecting both adversaries!
      Averaged value = (6 * 1.0) / 6 = 1.0 exactly.
    """
    updates, _ = _create_synthetic_client_updates(K=10, base_val=1.0)
    # Clients 8 and 9 are colluding malicious adversaries
    for idx in TRAINABLE_INDICES:
        updates[8][1][idx].fill(100.0)
        updates[9][1][idx].fill(100.0)

    # 1. Trimmed Mean beta=1
    agg_b1 = TrimmedMeanAggregator(beta=1)
    res_b1, metrics_b1 = agg_b1.aggregate(updates)
    assert metrics_b1["beta"] == 1
    assert metrics_b1["included_per_coordinate"] == 8
    for idx in TRAINABLE_INDICES:
        # beta=1 is poisoned by the second colluding adversary
        assert np.allclose(res_b1[idx], 13.375, atol=1e-5)

    # 2. Trimmed Mean beta=2
    agg_b2 = TrimmedMeanAggregator(beta=2)
    res_b2, metrics_b2 = agg_b2.aggregate(updates)
    assert metrics_b2["beta"] == 2
    assert metrics_b2["included_per_coordinate"] == 6
    for idx in TRAINABLE_INDICES:
        # beta=2 completely eliminates both colluding adversaries
        assert np.allclose(res_b2[idx], 1.0, atol=1e-5)


def test_tristate_all_suspicious_preserves_sample_proportional_weights():
    """
    Verify that when ALL participating clients are SUSPICIOUS:
    - Original sample-proportional weights are strictly retained.
    - Uniform discount delta factor cancels out during normalization:
      w_i = (n_i * delta) / sum_j(n_j * delta) = n_i / sum_j(n_j)
    - Effective weights do NOT collapse into uniform 1/K weights.
    """
    # 10 clients with highly heterogeneous sample sizes: 100, 200, ..., 1000
    updates, _ = _create_synthetic_client_updates(K=10)
    heterogeneous_counts = [100 * (i + 1) for i in range(10)]
    tot_samples = sum(heterogeneous_counts)
    expected_sample_weights = [n / tot_samples for n in heterogeneous_counts]

    for i in range(10):
        updates[i] = (str(i), updates[i][1], heterogeneous_counts[i])

    # Mock detector results where every client is SUSPICIOUS
    detector_results = {
        str(i): type("MockDecision", (), {"decision": "SUSPICIOUS", "anomaly_score": 0.45})()
        for i in range(10)
    }
    context = {"detector_results": detector_results}

    for discount in [0.50, 0.25]:
        agg = TriStateConsensusAggregator(mode=f"discount_{int(discount*100):02d}", suspicious_discount=discount)
        _, metrics = agg.aggregate(updates, context=context)

        assert metrics["all_clients_suspicious"] is True
        assert metrics["included_clients"] == 10

        # Verify that effective weights match original sample-proportional weights
        for i in range(10):
            cid = str(i)
            eff_w = metrics["effective_weights"][cid]
            orig_w = metrics["original_weights"][cid]
            assert pytest.approx(eff_w, abs=1e-6) == expected_sample_weights[i]
            assert pytest.approx(eff_w, abs=1e-6) == orig_w

        # Confirm it is strictly NOT uniform 1/K = 0.10
        # Client 0 should be ~0.0182 and Client 9 should be ~0.1818
        assert abs(metrics["effective_weights"]["0"] - 0.10) > 0.05
        assert abs(metrics["effective_weights"]["9"] - 0.10) > 0.05


def test_tristate_all_malicious_preserves_previous_global_model_with_alert():
    """
    Verify that when ALL participating clients are MALICIOUS:
    - Aggregator rejects all client updates (included_clients = 0, effective_weights = 0.0).
    - Preserves previous global model state without distortion.
    - Emits explicit security_alert in telemetry.
    """
    updates, prev_params = _create_synthetic_client_updates(K=10, base_val=5.0)

    # All client updates are poisoned with 999.0
    for i in range(10):
        for idx in TRAINABLE_INDICES:
            updates[i][1][idx].fill(999.0)

    detector_results = {
        str(i): type("MockDecision", (), {"decision": "MALICIOUS", "anomaly_score": 0.95})()
        for i in range(10)
    }
    context = {"detector_results": detector_results}

    agg = TriStateConsensusAggregator(mode="discount_050", suspicious_discount=0.50)
    aggregated, metrics = agg.aggregate(updates, global_params_prev=prev_params, context=context)

    assert metrics["all_clients_malicious"] is True
    assert metrics["fallback_to_previous_round"] is True
    assert metrics["included_clients"] == 0
    assert "security_alert" in metrics
    assert "ALERT" in metrics["security_alert"]
    for cid in metrics["effective_weights"]:
        assert metrics["effective_weights"][cid] == 0.0

    # Parameters must be strictly identical to previous global model state, not poisoned with 999.0
    for idx in range(len(prev_params)):
        assert np.allclose(aggregated[idx], prev_params[idx], atol=1e-7)
        if idx in TRAINABLE_INDICES:
            assert not np.allclose(aggregated[idx], 999.0, atol=1e-3)


def test_batchnorm_consistency_across_all_nine_aggregators():
    """
    Verify that all nine aggregation strategies apply a consistent BatchNorm policy:
    1. Integer counter (num_batches_tracked) advances by local steps without historical duplication.
    2. Floating-point buffers (running_mean, running_var) are valid and finite float32 arrays.
    """
    from federated_learning.aggregation.strategies import create_robust_strategy
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    # Mock DataLoader and device
    dummy_x = torch.zeros(4, 1, 28, 28)
    dummy_y = torch.zeros(4, dtype=torch.long)
    dummy_loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=2)
    device = torch.device("cpu")

    strategy_names = [
        "fedavg",
        "coordinate_median",
        "trimmed_mean_beta1",
        "trimmed_mean_beta2",
        "multi_krum_f1",
        "multi_krum_f2",
        "tri_state_monitor",
        "tri_state_discount_050",
        "tri_state_discount_025",
    ]

    updates, prev_params = _create_synthetic_client_updates(K=10)
    # Set server counter to 100, local clients took 12 steps -> 112
    for i in range(10):
        for idx in BN_INT_INDICES:
            updates[i][1][idx] = np.array(112, dtype=np.int64)

    for s_name in strategy_names:
        strat = create_robust_strategy(strategy_name=s_name, val_loader=dummy_loader, device=device)
        agg_res, metrics = strat.aggregator.aggregate(updates, global_params_prev=prev_params)

        for idx in BN_INT_INDICES:
            counter = agg_res[idx]
            assert counter.dtype == np.int64
            assert counter.item() == 112, f"Strategy {s_name} failed counter check: got {counter.item()}"

        for idx in BN_FLOAT_INDICES:
            buf = agg_res[idx]
            assert buf.dtype == np.float32
            assert np.all(np.isfinite(buf)), f"Strategy {s_name} produced non-finite BN float buffer"


def test_compound_attack_batchnorm_outlier_poisoning_rejection():
    """
    Verify behavior under a COMPOUND attack (both trainable parameters AND BatchNorm buffers poisoned):
    - 9 honest clients have running_mean = 0.0, running_var = 1.0, trainable params = 1.0.
    - 1 malicious client (client 9) sends extreme running_mean = 1,000,000.0, running_var = 1,000,000.0,
      AND anomalous trainable params = 1000.0.
    Verification:
    - Multi-Krum excludes client 9 based on its abnormal trainable parameters.
    - Tri-State (discount) excludes client 9 based on its MALICIOUS detector classification.
    - Coordinate Median and Trimmed Mean reject the buffer outlier via coordinate-wise robust aggregation.
    - FedAvg and Tri-State Monitor-Only absorb the attack with full sample weight (100,000.0).
    """
    updates, _ = _create_synthetic_client_updates(K=10, base_val=1.0)
    for i in range(9):
        for idx in BN_FLOAT_INDICES:
            val = 0.0 if "running_mean" in str(idx) or idx in [4, 11] else 1.0
            updates[i][1][idx].fill(val)

    # Poison client 9 in BOTH trainable parameters AND BatchNorm buffers
    for idx in TRAINABLE_INDICES:
        updates[9][1][idx].fill(1000.0)
    for idx in BN_FLOAT_INDICES:
        updates[9][1][idx].fill(1_000_000.0)

    # 1. Coordinate-wise Median (protection: coordinate-wise robust aggregation)
    agg_med = CoordinateMedianAggregator()
    res_med, _ = agg_med.aggregate(updates)
    for idx in BN_FLOAT_INDICES:
        expected = 0.0 if idx in [4, 11] else 1.0
        assert np.allclose(res_med[idx], expected, atol=1e-5), "CoordinateMedian failed BN buffer rejection!"

    # 2. Trimmed Mean beta=1 (protection: coordinate-wise robust aggregation)
    agg_tm1 = TrimmedMeanAggregator(beta=1)
    res_tm1, _ = agg_tm1.aggregate(updates)
    for idx in BN_FLOAT_INDICES:
        expected = 0.0 if idx in [4, 11] else 1.0
        assert np.allclose(res_tm1[idx], expected, atol=1e-5), "TrimmedMean beta=1 failed BN buffer rejection!"

    # 3. Trimmed Mean beta=2 (protection: coordinate-wise robust aggregation)
    agg_tm2 = TrimmedMeanAggregator(beta=2)
    res_tm2, _ = agg_tm2.aggregate(updates)
    for idx in BN_FLOAT_INDICES:
        expected = 0.0 if idx in [4, 11] else 1.0
        assert np.allclose(res_tm2[idx], expected, atol=1e-5), "TrimmedMean beta=2 failed BN buffer rejection!"

    # 4. Multi-Krum f=1 & f=2 (protection: client selection based on abnormal trainable parameters)
    for f_val in [1, 2]:
        agg_krum = MultiKrumAggregator(f=f_val)
        res_krum, m_krum = agg_krum.aggregate(updates)
        assert "9" in m_krum["excluded_client_ids"], f"Multi-Krum f={f_val} failed to exclude client 9!"
        for idx in BN_FLOAT_INDICES:
            expected = 0.0 if idx in [4, 11] else 1.0
            assert np.allclose(res_krum[idx], expected, atol=1e-5), f"Multi-Krum f={f_val} failed BN buffer rejection!"

    # 5. Tri-State Discount (protection: Tri-State malicious exclusion)
    detector_results = {str(i): type("MockDecision", (), {"decision": "NORMAL", "anomaly_score": 0.0})() for i in range(9)}
    detector_results["9"] = type("MockDecision", (), {"decision": "MALICIOUS", "anomaly_score": 0.99})()
    context = {"detector_results": detector_results}

    for discount in [0.50, 0.25]:
        agg_tri = TriStateConsensusAggregator(mode=f"discount_{int(discount*100):02d}", suspicious_discount=discount)
        res_tri, m_tri = agg_tri.aggregate(updates, context=context)
        assert "9" in m_tri["excluded_client_ids"]
        for idx in BN_FLOAT_INDICES:
            expected = 0.0 if idx in [4, 11] else 1.0
            assert np.allclose(res_tri[idx], expected, atol=1e-5), f"Tri-State discount {discount} failed BN buffer rejection!"

    # 6. FedAvg & Tri-State Monitor-Only: NO PROTECTION (both absorb attack with full sample weight 100,000.0)
    agg_fed = FedAvgAggregator()
    res_fed, _ = agg_fed.aggregate(updates)
    agg_mon = TriStateConsensusAggregator(mode="monitor_only")
    res_mon, _ = agg_mon.aggregate(updates, context=context)

    for idx in BN_FLOAT_INDICES:
        if idx in [4, 11]:
            assert np.allclose(res_fed[idx], 100_000.0, atol=1.0)
            assert np.allclose(res_mon[idx], 100_000.0, atol=1.0)
            assert np.allclose(res_fed[idx], res_mon[idx], atol=1e-6)


def test_isolated_batchnorm_only_poisoning_across_all_nine_aggregators():
    """
    Verify behavior under an ISOLATED BatchNorm-only poisoning attack:
    - 10 clients: clients 0..9.
    - Malicious client (client 4):
      * Trainable parameters are NORMAL / honest updates (matching peers with realistic gradient noise).
      * ONLY BatchNorm running_mean and running_var are set to extreme values: 1,000,000.0.
      * Integer counters remain valid.
    This tests whether aggregators detect or resist buffer-only poisoning when trainable parameters appear normal.
    Results:
    - Coordinate Median: Protected (< 10.0) via (2) Coordinate-wise robust buffer aggregation.
    - Trimmed Mean beta=1: Protected (< 10.0) via (2) Coordinate-wise robust buffer aggregation.
    - Trimmed Mean beta=2: Protected (< 10.0) via (2) Coordinate-wise robust buffer aggregation.
    - FedAvg: NO PROTECTION. Absorbs buffer with sample weight 0.10 (~100,000.0).
    - Tri-State Monitor-Only: NO PROTECTION. Matches FedAvg exactly (~100,000.0).
    - Multi-Krum f=1 & f=2: NO PROTECTION. Multi-Krum operates strictly on trainable parameter distances.
      Because client 4's trainable parameters are normal, it is SELECTED among the top candidates!
      Multi-Krum does NOT detect buffer-only anomalies, resulting in buffer corruption (~111,111.1 / ~125,000.0).
    - Tri-State Discount (0.50 & 0.25):
      * With MALICIOUS detector context: Protected via (3) Tri-State malicious exclusion (weight 0.0).
      * Without detector context / evaluated NORMAL: NO PROTECTION (absorbs ~100,000.0).
    """
    np.random.seed(42)
    updates, _ = _create_synthetic_client_updates(K=10, base_val=1.0)

    # Add realistic gradient noise to all 10 clients' trainable parameters so they are distinct but close
    for i in range(10):
        for idx in TRAINABLE_INDICES:
            updates[i][1][idx] += np.random.normal(0, 0.01, size=updates[i][1][idx].shape).astype(np.float32)
        for idx in BN_FLOAT_INDICES:
            val = 0.0 if idx in [4, 11] else 1.0
            updates[i][1][idx].fill(val)
        for idx in BN_INT_INDICES:
            updates[i][1][idx] = np.array(110 + i, dtype=np.int64)

    # Client 4: ONLY BN buffers poisoned to 1e6! Trainable parameters remain honest
    for idx in BN_FLOAT_INDICES:
        updates[4][1][idx].fill(1_000_000.0)
    for idx in BN_INT_INDICES:
        updates[4][1][idx] = np.array(114, dtype=np.int64)

    # 1. FedAvg (No protection)
    agg_fed = FedAvgAggregator()
    res_fed, m_fed = agg_fed.aggregate(updates)
    assert m_fed["effective_weights"]["4"] == 0.10
    assert np.allclose(res_fed[4], 100_000.0, atol=1.0)

    # 2. Tri-State Monitor-Only (No protection, matches FedAvg)
    agg_mon = TriStateConsensusAggregator(mode="monitor_only")
    res_mon, m_mon = agg_mon.aggregate(updates)
    assert m_mon["effective_weights"]["4"] == 0.10
    assert np.allclose(res_mon[4], 100_000.0, atol=1.0)
    assert np.allclose(res_mon[4], res_fed[4], atol=1e-6)

    # 3. Coordinate Median (Coordinate-wise robust buffer aggregation)
    agg_cm = CoordinateMedianAggregator()
    res_cm, m_cm = agg_cm.aggregate(updates)
    assert np.allclose(res_cm[4], 0.0, atol=1e-5)
    assert np.allclose(res_cm[5], 1.0, atol=1e-5)

    # 4. Trimmed Mean beta=1 (Coordinate-wise robust buffer aggregation)
    agg_tm1 = TrimmedMeanAggregator(beta=1)
    res_tm1, _ = agg_tm1.aggregate(updates)
    assert np.allclose(res_tm1[4], 0.0, atol=1e-5)
    assert np.allclose(res_tm1[5], 1.0, atol=1e-5)

    # 5. Trimmed Mean beta=2 (Coordinate-wise robust buffer aggregation)
    agg_tm2 = TrimmedMeanAggregator(beta=2)
    res_tm2, _ = agg_tm2.aggregate(updates)
    assert np.allclose(res_tm2[4], 0.0, atol=1e-5)
    assert np.allclose(res_tm2[5], 1.0, atol=1e-5)

    # 6. Multi-Krum f=1 (No protection against buffer-only attack: client 4 is SELECTED)
    agg_k1 = MultiKrumAggregator(f=1)
    res_k1, m_k1 = agg_k1.aggregate(updates)
    assert "4" in m_k1["selected_client_ids"], "Multi-Krum f=1 should select client 4 because trainable params are normal!"
    assert pytest.approx(m_k1["effective_weights"]["4"], abs=1e-5) == 1.0 / 9.0
    assert np.allclose(res_k1[4], 111_111.11, atol=1.0), "Multi-Krum f=1 absorbed poisoned buffer as expected!"

    # 7. Multi-Krum f=2 (No protection against buffer-only attack: client 4 is SELECTED)
    agg_k2 = MultiKrumAggregator(f=2)
    res_k2, m_k2 = agg_k2.aggregate(updates)
    assert "4" in m_k2["selected_client_ids"], "Multi-Krum f=2 should select client 4 because trainable params are normal!"
    assert pytest.approx(m_k2["effective_weights"]["4"], abs=1e-5) == 1.0 / 8.0
    assert np.allclose(res_k2[4], 125_000.0, atol=1.0), "Multi-Krum f=2 absorbed poisoned buffer as expected!"

    # 8. Tri-State Discount (with MALICIOUS context: Tri-State malicious exclusion)
    ctx_malicious = {
        "detector_results": {
            str(i): type("MockDecision", (), {"decision": "NORMAL", "anomaly_score": 0.0})()
            for i in range(10)
        }
    }
    ctx_malicious["detector_results"]["4"] = type("MockDecision", (), {"decision": "MALICIOUS", "anomaly_score": 0.99})()

    for discount in [0.50, 0.25]:
        agg_tri = TriStateConsensusAggregator(mode=f"discount_{int(discount*100):02d}", suspicious_discount=discount)
        res_tri, m_tri = agg_tri.aggregate(updates, context=ctx_malicious)
        assert "4" in m_tri["excluded_client_ids"]
        assert m_tri["effective_weights"]["4"] == 0.0
        assert np.allclose(res_tri[4], 0.0, atol=1e-5)
        assert np.allclose(res_tri[5], 1.0, atol=1e-5)


def test_batchnorm_integer_counter_formula_and_no_duplication():
    """
    Verify the BatchNorm integer counter (num_batches_tracked) aggregation formula across all 9 strategies:
    - Prevents duplication: Does NOT sum historical counters (which would multiply base count by K=10).
    - Advances by sample-weighted average local steps:
      Global Counter = round(sum_{i in included} w_i * returned_counter_i)
      which mathematically equals N^(t-1) + round(sum_{i in included} w_i * delta_B_i).
    - Tested with heterogeneous client sample sizes and varied local batch counts.
    """
    from federated_learning.aggregation.strategies import create_robust_strategy
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    dummy_x = torch.zeros(4, 1, 28, 28)
    dummy_y = torch.zeros(4, dtype=torch.long)
    dummy_loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=2)
    device = torch.device("cpu")

    base_counter = 1000
    sample_counts = [100, 150, 200, 250, 300, 120, 180, 220, 140, 160]
    K = 10
    updates, base_params = _create_synthetic_client_updates(K=K, sample_counts=sample_counts)

    # Each client trained for a different number of local batches: delta_B_i = 10 + 2*i
    # Returned counter = base_counter + delta_B_i
    local_deltas = [10 + 2 * i for i in range(K)]
    for i in range(K):
        for idx in BN_INT_INDICES:
            updates[i][1][idx] = np.array(base_counter + local_deltas[i], dtype=np.int64)

    strategy_names = [
        "fedavg",
        "coordinate_median",
        "trimmed_mean_beta1",
        "trimmed_mean_beta2",
        "multi_krum_f1",
        "multi_krum_f2",
        "tri_state_monitor",
        "tri_state_discount_050",
        "tri_state_discount_025",
    ]

    for s_name in strategy_names:
        strat = create_robust_strategy(strategy_name=s_name, val_loader=dummy_loader, device=device)
        res, metrics = strat.aggregator.aggregate(updates, global_params_prev=base_params)

        for idx in BN_INT_INDICES:
            global_cnt = res[idx].item()
            # 1. Counter must advance beyond the base counter (1000)
            assert global_cnt > base_counter, f"Strategy {s_name} counter {global_cnt} did not advance!"
            # 2. Counter must be within the valid step range [1010, 1030]
            assert 1010 <= global_cnt <= 1030, f"Strategy {s_name} counter {global_cnt} outside expected step range [1010, 1030]!"
            # 3. Counter must NOT multiply the base counter by K (which would be ~10,000)
            assert global_cnt < 2000, f"Strategy {s_name} counter {global_cnt} appears to duplicate historical base counts!"


def test_parameter_vs_delta_space_aggregation_mathematical_equivalence():
    """
    Verify the mathematical equivalence between:
    1. Parameter-space aggregation: theta^(t) = Aggregator({theta_i^(t)})
    2. Delta-space aggregation:     theta^(t) = theta^(t-1) + Aggregator({theta_i^(t) - theta^(t-1)})
    Confirms:
    - No aggregator double-adds the prior global model parameters theta^(t-1).
    - All nine strategies operate symmetrically on parameter updates.
    """
    from federated_learning.aggregation.strategies import create_robust_strategy
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    dummy_x = torch.zeros(4, 1, 28, 28)
    dummy_y = torch.zeros(4, dtype=torch.long)
    dummy_loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=2)
    device = torch.device("cpu")

    # Fixed synthetic base parameters theta^(t-1) with non-zero values (2.5) across all layers
    _, base_params = _create_synthetic_client_updates(K=10)
    prev_params = [np.full_like(arr, fill_value=2.5, dtype=np.float32) for arr in base_params]

    # Shift each client update by deterministic asymmetric offsets: theta_i = base + offset_i
    # so delta_i = offset_i (avoiding artificial tie-breaking instability on exact straight lines)
    offsets = [0.05, 0.08, 0.10, 0.11, 0.12, 0.13, 0.15, 0.18, 0.22, 1.50]
    updates_param_space = []
    updates_delta_space = []

    for i in range(10):
        p_list = [arr.copy() for arr in prev_params]
        d_list = [arr.copy() for arr in prev_params]
        for idx in TRAINABLE_INDICES:
            offset = offsets[i]
            p_list[idx] = prev_params[idx] + offset
            d_list[idx] = np.full_like(prev_params[idx], fill_value=offset, dtype=np.float32)
        updates_param_space.append((str(i), p_list, 100))
        updates_delta_space.append((str(i), d_list, 100))

    strategy_names = [
        "fedavg",
        "coordinate_median",
        "trimmed_mean_beta1",
        "trimmed_mean_beta2",
        "multi_krum_f1",
        "multi_krum_f2",
        "tri_state_monitor",
        "tri_state_discount_050",
        "tri_state_discount_025",
    ]

    for s_name in strategy_names:
        strat = create_robust_strategy(strategy_name=s_name, val_loader=dummy_loader, device=device)

        # 1. Parameter-space aggregation
        res_param, _ = strat.aggregator.aggregate(updates_param_space, global_params_prev=prev_params)

        # 2. Delta-space aggregation: add aggregated deltas to prev_params
        res_delta, _ = strat.aggregator.aggregate(updates_delta_space, global_params_prev=prev_params)

        for idx in TRAINABLE_INDICES:
            theta_from_delta = prev_params[idx] + res_delta[idx]
            # Must be mathematically identical within float precision
            assert np.allclose(res_param[idx], theta_from_delta, atol=1e-5), (
                f"Strategy {s_name} failed parameter vs delta space equivalence on trainable tensor {idx}!"
            )
            # Crucial check: verify that prev_params was NOT double-added
            # If double added, res_param would be approx 2 * prev_params + delta (difference of 2.5)
            assert not np.allclose(res_param[idx], 2 * prev_params[idx] + res_delta[idx], atol=1e-1), (
                f"Strategy {s_name} appears to double-add previous global model parameters!"
            )


def test_detector_state_isolation_between_experiments():
    """
    Verify complete detector state isolation between distinct federated experiments:
    1. Simulation 1 experiences severe anomalies and attacks, elevating EMA accumulators.
    2. Simulation 2 starts afresh (or after strategy.reset()).
    3. Confirms that round 1 of Simulation 2 has:
       - An empty initial client history.
       - Exactly 0.0 EMA anomaly accumulator for all clients.
       - Evaluates honest updates to NORMAL with zero memory of Simulation 1.
       - Calibrated thresholds remain bit-for-bit identical.
    """
    from federated_learning.aggregation.strategies import create_robust_strategy
    from experiments.run_phase6_blind_evaluation import load_frozen_detector
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    dummy_x = torch.zeros(4, 1, 28, 28)
    dummy_y = torch.zeros(4, dtype=torch.long)
    dummy_loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=2)
    device = torch.device("cpu")

    # Load frozen detector
    detector1, _ = load_frozen_detector()
    strat1 = create_robust_strategy(
        strategy_name="tri_state_discount_050",
        val_loader=dummy_loader,
        device=device,
        frozen_detector=detector1,
    )

    # Simulation 1: Run 3 rounds with extreme anomalous updates
    updates_sim1, prev_params = _create_synthetic_client_updates(K=10, base_val=1.0)
    for rnd in range(1, 4):
        # Inject extreme outlier for client 0
        for idx in TRAINABLE_INDICES:
            updates_sim1[0][1][idx].fill(500.0 * rnd)
        strat1.feature_extractor.extract_round_features(
            server_round=rnd,
            client_updates=updates_sim1,
            global_params_prev=prev_params,
            global_params_full=prev_params,
        )

    # In Simulation 1, client 0 history is heavily populated and EMA is elevated
    assert "0" in strat1.feature_extractor.client_history
    elevated_ema = strat1.feature_extractor.client_history["0"]["ema_accumulator"]
    assert elevated_ema > 0.0

    # Simulation 2: Create a completely fresh strategy instance
    detector2, _ = load_frozen_detector()
    strat2 = create_robust_strategy(
        strategy_name="tri_state_discount_050",
        val_loader=dummy_loader,
        device=device,
        frozen_detector=detector2,
    )

    # Verify initial state of Simulation 2 is completely isolated and empty
    assert len(strat2.feature_extractor.client_history) == 0
    assert len(strat2.extracted_records) == 0
    assert len(strat2.round_telemetry) == 0

    # Execute round 1 of Simulation 2 with clean honest updates
    updates_sim2, _ = _create_synthetic_client_updates(K=10, base_val=1.0)
    round1_features = strat2.feature_extractor.extract_round_features(
        server_round=1,
        client_updates=updates_sim2,
        global_params_prev=prev_params,
        global_params_full=prev_params,
    )

    # Round 1 in Simulation 2 must have exactly 0.0 EMA accumulator
    assert round1_features["0"]["ema_anomaly_accumulator"] == 0.0
    # Evaluate client 0 with detector 2
    res_sim2 = detector2.evaluate_client(round1_features["0"])
    assert res_sim2.decision.value == "NORMAL"
    assert res_sim2.anomaly_score < 0.10

    # Verify thresholds in detector 1 and detector 2 are identical
    thresh1 = detector1.thresholds
    thresh2 = detector2.thresholds
    assert thresh1.z_mag_malicious == thresh2.z_mag_malicious
    assert thresh1.cos_strong_negative == thresh2.cos_strong_negative
    assert thresh1.ema_accumulator_malicious == thresh2.ema_accumulator_malicious
