"""
Standard FedAvg Aggregator (McMahan et al., 2017).
Serves as canonical baseline with sample-proportional linear averaging.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from federated_learning.aggregation.base import (
    BaseAggregator,
    TRAINABLE_INDICES,
    BN_FLOAT_INDICES,
    BN_INT_INDICES,
)


class FedAvgAggregator(BaseAggregator):
    """
    Standard Federated Averaging Aggregator.
    Weights each client update strictly proportional to local dataset size n_i / sum(n_j).
    """

    def __init__(self) -> None:
        super().__init__(name="fedavg")

    def aggregate(
        self,
        client_updates: List[Tuple[str, List[np.ndarray], int]],
        global_params_prev: Optional[List[np.ndarray]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        if not client_updates:
            if global_params_prev is not None:
                return [arr.copy() for arr in global_params_prev], {"status": "fallback_empty"}
            raise ValueError("No client updates provided for FedAvg aggregation.")

        K = len(client_updates)
        sample_counts = [u[2] for u in client_updates]
        total_samples = max(sum(sample_counts), 1)
        original_weights = [n / total_samples for n in sample_counts]
        effective_weights = list(original_weights)

        # 1. Aggregate Trainable Parameters
        num_layers = len(client_updates[0][1])
        aggregated_ndarrays = [None] * num_layers

        for idx in TRAINABLE_INDICES:
            layer_acc = np.zeros_like(client_updates[0][1][idx], dtype=np.float32)
            for i, (_, params, _) in enumerate(client_updates):
                layer_acc += effective_weights[i] * params[idx].astype(np.float32)
            aggregated_ndarrays[idx] = layer_acc

        # 2. Aggregate BatchNorm Buffers (Float running stats + Int64 batch counts)
        bn_buffers = self.aggregate_bn_buffers(
            client_updates=client_updates,
            effective_weights=effective_weights,
            included_mask=[True] * K,
        )
        for idx, buf in bn_buffers.items():
            aggregated_ndarrays[idx] = buf

        metrics = {
            "aggregator": self.name,
            "participating_clients": K,
            "included_clients": K,
            "original_weights": {client_updates[i][0]: round(original_weights[i], 6) for i in range(K)},
            "effective_weights": {client_updates[i][0]: round(effective_weights[i], 6) for i in range(K)},
        }

        return aggregated_ndarrays, metrics
