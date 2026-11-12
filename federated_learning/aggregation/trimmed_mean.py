"""
Trimmed Mean Aggregator (Yin et al., 2018).
Sorts client parameter updates coordinate-wise and removes the lowest beta and highest beta values
before computing the arithmetic mean across the remaining K - 2*beta values.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from federated_learning.aggregation.base import (
    BaseAggregator,
    TRAINABLE_INDICES,
)


class TrimmedMeanAggregator(BaseAggregator):
    """
    Trimmed Mean Aggregator.
    For each parameter element theta_j, sorts values across K clients and averages the inner K - 2*beta values.
    Default beta = 1 for K = 10 (10% trim per tail, 20% total trim).
    """

    def __init__(self, beta: int = 1) -> None:
        self.beta = int(beta)
        super().__init__(name=f"trimmed_mean_beta{self.beta}")

    def aggregate(
        self,
        client_updates: List[Tuple[str, List[np.ndarray], int]],
        global_params_prev: Optional[List[np.ndarray]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        if not client_updates:
            if global_params_prev is not None:
                return [arr.copy() for arr in global_params_prev], {"status": "fallback_empty"}
            raise ValueError("No client updates provided for Trimmed Mean aggregation.")

        K = len(client_updates)
        sample_counts = [u[2] for u in client_updates]
        total_samples = max(sum(sample_counts), 1)
        original_weights = [n / total_samples for n in sample_counts]

        # Validate beta
        effective_beta = min(self.beta, (K - 1) // 2)
        if effective_beta < 0:
            effective_beta = 0

        effective_weights = [1.0 / max(K - 2 * effective_beta, 1)] * K

        num_layers = len(client_updates[0][1])
        aggregated_ndarrays = [None] * num_layers

        # 1. Coordinate-wise Trimmed Mean on Trainable Parameters
        for idx in TRAINABLE_INDICES:
            stacked_layer = np.stack([client_updates[i][1][idx] for i in range(K)], axis=0)
            if effective_beta > 0:
                sorted_layer = np.sort(stacked_layer, axis=0)
                trimmed_layer = sorted_layer[effective_beta : K - effective_beta]
                aggregated_ndarrays[idx] = np.mean(trimmed_layer, axis=0).astype(np.float32)
            else:
                aggregated_ndarrays[idx] = np.mean(stacked_layer, axis=0).astype(np.float32)

        # 2. Coordinate-wise Trimmed Mean on BatchNorm Buffers to prevent buffer poisoning
        bn_buffers = self.aggregate_bn_buffers_trimmed(
            client_updates=client_updates,
            beta=effective_beta,
        )
        for idx, buf in bn_buffers.items():
            aggregated_ndarrays[idx] = buf

        metrics = {
            "aggregator": self.name,
            "beta": effective_beta,
            "participating_clients": K,
            "included_per_coordinate": K - 2 * effective_beta,
            "original_weights": {client_updates[i][0]: round(original_weights[i], 6) for i in range(K)},
            "effective_weights": {client_updates[i][0]: round(effective_weights[i], 6) for i in range(K)},
        }

        return aggregated_ndarrays, metrics
