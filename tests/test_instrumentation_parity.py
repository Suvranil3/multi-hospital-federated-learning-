"""
Tests for Phase 6 Instrumentation Mathematical Equivalence.
Directly verifies that InstrumentedFedAvg produces bit-for-bit identical aggregated
parameters to standard Flower FedAvg when supplied with identical client fit results,
including all 12 trainable tensors and 6 BatchNorm buffers.
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from flwr.common import (
    FitRes,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from federated_learning.model import get_model
from federated_learning.client import get_parameters_from_model
from federated_learning.strategy_instrumented import InstrumentedFedAvg


class MockClientProxy(ClientProxy):
    """Mock client proxy for Flower strategy testing."""
    def __init__(self, cid: str):
        super().__init__(cid)

    def get_properties(self, ins, timeout, group_id):
        raise NotImplementedError

    def get_parameters(self, ins, timeout, group_id):
        raise NotImplementedError

    def fit(self, ins, timeout, group_id):
        raise NotImplementedError

    def evaluate(self, ins, timeout, group_id):
        raise NotImplementedError

    def reconnect(self, ins, timeout, group_id):
        raise NotImplementedError


def test_fedavg_instrumented_mathematical_equivalence():
    """
    Test direct mathematical equivalence between standard FedAvg and InstrumentedFedAvg.
    Verifies that the instrumentation extracts features with ZERO impact on aggregation.
    """
    device = torch.device("cpu")
    base_model = get_model(in_channels=1, num_classes=2, dropout_rate=0.2)
    base_params = get_parameters_from_model(base_model)
    assert len(base_params) == 18  # 12 trainable + 6 BatchNorm buffers

    # Create dummy validation dataset for InstrumentedFedAvg
    dummy_x = torch.randn(20, 1, 28, 28)
    dummy_y = torch.randint(0, 2, (20,))
    dummy_loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=10)

    # Initialize standard FedAvg and InstrumentedFedAvg with identical initial parameters
    initial_fl_params = ndarrays_to_parameters(base_params)
    fedavg = fl.server.strategy.FedAvg(
        initial_parameters=initial_fl_params,
        fraction_fit=1.0,
        min_fit_clients=10,
        min_available_clients=10,
    )
    instrumented_fedavg = InstrumentedFedAvg(
        val_loader=dummy_loader,
        device=device,
        output_jsonl_path=None,
        initial_parameters=initial_fl_params,
        fraction_fit=1.0,
        min_fit_clients=10,
        min_available_clients=10,
    )

    # Synthesize realistic client fit results for 10 hospitals
    client_results = []
    np.random.seed(42)
    sample_counts = [120, 340, 210, 450, 180, 290, 500, 160, 310, 240]

    for cid in range(10):
        # Create perturbed client parameter arrays (including weights, biases, and BN buffers)
        client_ndarrays = []
        for i, arr in enumerate(base_params):
            if arr.dtype == np.int64:
                # num_batches_tracked buffer
                client_ndarrays.append(arr.copy() + (cid + 1) * 10)
            elif "running" in list(base_model.state_dict().keys())[i]:
                # running_mean, running_var
                noise = np.random.normal(0, 0.05, size=arr.shape).astype(arr.dtype)
                client_ndarrays.append(arr.copy() + noise)
            else:
                # Trainable weight/bias tensor
                noise = np.random.normal(0, 0.01, size=arr.shape).astype(arr.dtype)
                client_ndarrays.append(arr.copy() + noise)

        fit_res = FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(client_ndarrays),
            num_examples=sample_counts[cid],
            metrics={"client_id": str(cid), "train_loss": 0.35},
        )
        proxy = MockClientProxy(cid=str(cid))
        client_results.append((proxy, fit_res))

    # Execute aggregation on identical client results
    out_fedavg, metrics_fedavg = fedavg.aggregate_fit(server_round=1, results=client_results, failures=[])
    out_inst, metrics_inst = instrumented_fedavg.aggregate_fit(server_round=1, results=client_results, failures=[])

    assert out_fedavg is not None
    assert out_inst is not None

    arrs_fedavg = parameters_to_ndarrays(out_fedavg)
    arrs_inst = parameters_to_ndarrays(out_inst)

    assert len(arrs_fedavg) == len(arrs_inst) == 18

    # Compare every tensor and buffer
    max_abs_diff = 0.0
    for idx, (a_std, a_inst) in enumerate(zip(arrs_fedavg, arrs_inst)):
        diff = np.max(np.abs(a_std - a_inst))
        if diff > max_abs_diff:
            max_abs_diff = diff
        np.testing.assert_allclose(
            a_inst,
            a_std,
            rtol=1e-7,
            atol=1e-7,
            err_msg=f"Tensor index {idx} diverged between FedAvg and InstrumentedFedAvg!",
        )

    # Assert exact bit-level equivalence (zero parameter difference)
    assert max_abs_diff == 0.0, f"Expected bit-level 0.0 maximum difference, got {max_abs_diff}"
    print(f"\n[Test Passed] Maximum parameter difference between standard FedAvg and InstrumentedFedAvg: {max_abs_diff}")
