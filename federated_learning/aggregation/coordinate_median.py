"""
Coordinate-wise Median Aggregator (Yin et al., 2018).
Computes the element-wise median across participating client updates for every trainable parameter.
Byzantine-robust against arbitrary single-coordinate perturbations up to 50% adversaries in IID.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from federated_learning.aggregation.base import (
    BaseAggregator,
    TRAINABLE_INDICES,
)


class CoordinateMedianAggregator(BaseAggregator):
    """
    Coordinate-wise Median Aggregator.
    For each parameter element theta_j, computes median_{i=1..K} (theta_{i, j}).
    """

    def __init__(self) -> None:
        super().__init__(name="coordinate_median")

    def aggregate(
        self,
        client_updates: List[Tuple[str, List[np.ndarray], int]],
        global_params_prev: Optional[List[np.ndarray]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        if not client_updates:
            if global_params_prev is not None:
                return [arr.copy() for arr in global_params_prev], {"status": "fallback_empty"}
            raise ValueError("No client updates provided for Coordinate-wise Median aggregation.")

        K = len(client_updates)
        sample_counts = [u[2] for u in client_updates]
        total_samples = max(sum(sample_counts), 1)
        original_weights = [n / total_samples for n in sample_counts]
        # Equal median weighting conceptually
        effective_weights = [1.0 / K] * K

        num_layers = len(client_updates[0][1])
        aggregated_ndarrays = [None] * num_layers

        # 1. Coordinate-wise Median on Trainable Parameters
        for idx in TRAINABLE_INDICES:
            stacked_layer = np.stack([client_updates[i][1][idx] for i in range(K)], axis=0)
            aggregated_ndarrays[idx] = np.median(stacked_layer, axis=0).astype(np.float32)

        # 2. Coordinate-wise Median on BatchNorm Buffers to prevent buffer poisoning
        bn_buffers = self.aggregate_bn_buffers_median(client_updates=client_updates)
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
