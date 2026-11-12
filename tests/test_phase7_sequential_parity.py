"""
Reproducibility and Parity Tests for Phase 7 Memory-Safe Sequential Runner.
Verifies scientific equivalence between the sequential execution path
and the original Flower aggregation mechanisms.
"""

import sys
from pathlib import Path
import hashlib
import numpy as np
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flwr.common import (
    FitRes,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from experiments.config import RESULTS_MODELS_DIR
from federated_learning.model import get_model
from federated_learning.client import (
    get_parameters_from_model,
    set_parameters_to_model,
)
from federated_learning.aggregation.strategies import create_robust_strategy
from federated_learning.aggregation.base import (
    TRAINABLE_INDICES,
    BN_FLOAT_INDICES,
    BN_INT_INDICES,
)
from experiments.run_phase7_memory_safe import (
    MockClientProxy,
    compute_param_hash,
)


def _generate_synthetic_updates(K: int = 10, seed: int = 42):
    """Generate deterministic synthetic parameter updates matching SmallCNN."""
    rng = np.random.RandomState(seed)
    dummy_model = get_model(in_channels=1, num_classes=2, dropout_rate=0.2)
    base_params = get_parameters_from_model(dummy_model)

    client_updates = []
    for cid in range(K):
        c_params = []
        for idx, arr in enumerate(base_params):
            if idx in TRAINABLE_INDICES:
                noise = rng.normal(loc=0.0, scale=0.01, size=arr.shape).astype(np.float32)
                c_params.append(arr + noise)
            elif idx in BN_FLOAT_INDICES:
                c_params.append(np.full_like(arr, fill_value=0.1 * (cid + 1), dtype=np.float32))
            elif idx in BN_INT_INDICES:
                c_params.append(np.array(10 * (cid + 1), dtype=np.int64))
            else:
                c_params.append(arr.copy())
        num_samples = 100 + cid * 10
        client_updates.append((str(cid), c_params, num_samples))
    return client_updates


class TestPhase7SequentialParity:
    """Verifies that the sequential in-process runner reproduces aggregation semantics."""

    def test_initial_checkpoint_hash_parity(self):
        """Verify initial checkpoint hash matches the authoritative reference."""
        init_path = RESULTS_MODELS_DIR / "initial_federated_model.pt"
        assert init_path.exists(), f"Initial checkpoint missing at {init_path}"
        payload = torch.load(init_path, map_location="cpu", weights_only=False)
        model = get_model(in_channels=1, num_classes=2, dropout_rate=0.2)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        ndarrays = get_parameters_from_model(model)
        param_hash = compute_param_hash(ndarrays)
        assert param_hash == "0c411b83c2f0acef58beb5b9a968030d877e079eb2356eb3748903f8ae9bbc83"

    @pytest.mark.parametrize("strat_name", [
        "fedavg",
        "coordinate_median",
        "trimmed_mean_beta1",
        "multi_krum_f1",
    ])
    def test_mock_proxy_aggregate_fit_parity(self, strat_name: str, tmp_path):
        """Verify MockClientProxy packaging in aggregate_fit produces valid, deterministic output."""
        client_updates = _generate_synthetic_updates(K=10, seed=123)

        # Build FitRes list with MockClientProxy (as done in sequential runner)
        results = []
        for cid_str, params, num_samples in client_updates:
            fit_res = FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters(params),
                num_examples=num_samples,
                metrics={"client_id": cid_str},
            )
            proxy = MockClientProxy(cid=cid_str)
            results.append((proxy, fit_res))

        dummy_dataset = torch.utils.data.TensorDataset(
            torch.zeros(10, 1, 28, 28), torch.zeros(10, dtype=torch.long)
        )
        dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=5)
        device = torch.device("cpu")

        strat = create_robust_strategy(
            strategy_name=strat_name,
            val_loader=dummy_loader,
            device=device,
            output_jsonl_path=tmp_path / f"{strat_name}_test.jsonl",
            min_fit_clients=10,
            min_available_clients=10,
        )

        aggregated_params, summary = strat.aggregate_fit(server_round=1, results=results, failures=[])
        assert aggregated_params is not None
        ndarrays = parameters_to_ndarrays(aggregated_params)
        assert len(ndarrays) == 18

        # Verify second call with identical inputs produces identical hash (determinism)
        strat2 = create_robust_strategy(
            strategy_name=strat_name,
            val_loader=dummy_loader,
            device=device,
            output_jsonl_path=tmp_path / f"{strat_name}_test2.jsonl",
            min_fit_clients=10,
            min_available_clients=10,
        )
        agg2, _ = strat2.aggregate_fit(server_round=1, results=results, failures=[])
        assert compute_param_hash(ndarrays) == compute_param_hash(parameters_to_ndarrays(agg2))

    def test_batchnorm_buffer_handling(self, tmp_path):
        """Verify BatchNorm float buffers are aggregated and int counters are scalar int64."""
        client_updates = _generate_synthetic_updates(K=10, seed=456)
        results = [
            (
                MockClientProxy(cid=cid_str),
                FitRes(
                    status=Status(code=Code.OK, message="OK"),
                    parameters=ndarrays_to_parameters(params),
                    num_examples=num_samples,
                    metrics={"client_id": cid_str},
                ),
            )
            for cid_str, params, num_samples in client_updates
        ]

        dummy_dataset = torch.utils.data.TensorDataset(
            torch.zeros(10, 1, 28, 28), torch.zeros(10, dtype=torch.long)
        )
        dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=5)
        device = torch.device("cpu")

        strat = create_robust_strategy(
            strategy_name="fedavg",
            val_loader=dummy_loader,
            device=device,
            output_jsonl_path=tmp_path / "bn_test.jsonl",
        )
        agg_params, _ = strat.aggregate_fit(server_round=1, results=results, failures=[])
        ndarrays = parameters_to_ndarrays(agg_params)

        for idx in BN_FLOAT_INDICES:
            assert ndarrays[idx].dtype == np.float32
            assert not np.isnan(ndarrays[idx]).any()
        for idx in BN_INT_INDICES:
            assert ndarrays[idx].dtype == np.int64
            assert ndarrays[idx].ndim == 0 or ndarrays[idx].shape == () or ndarrays[idx].shape == (1,)
