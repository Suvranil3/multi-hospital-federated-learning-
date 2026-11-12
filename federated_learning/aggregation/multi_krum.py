"""
Multi-Krum Byzantine-Robust Aggregator (Blanchard et al., 2017).
Selects m candidates with the minimal sum of Euclidean distances to their K - f - 2 nearest neighbors
and computes their average.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from federated_learning.aggregation.base import (
    BaseAggregator,
    TRAINABLE_INDICES,
)


class MultiKrumAggregator(BaseAggregator):
    """
    Multi-Krum Aggregator.
    Parameters:
        f: Expected number of Byzantine adversaries (default f = 1; supports f = 2).
        m: Number of candidates to select and average (default m = K - 2*f).
    """

    def __init__(self, f: int = 1, m: Optional[int] = None) -> None:
        super().__init__(name="multi_krum")
        self.f = int(f)
        self.m = int(m) if m is not None else None

    def aggregate(
        self,
        client_updates: List[Tuple[str, List[np.ndarray], int]],
        global_params_prev: Optional[List[np.ndarray]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        if not client_updates:
            if global_params_prev is not None:
                return [arr.copy() for arr in global_params_prev], {"status": "fallback_empty"}
            raise ValueError("No client updates provided for Multi-Krum aggregation.")

        K = len(client_updates)
        sample_counts = [u[2] for u in client_updates]
        total_samples = max(sum(sample_counts), 1)
        original_weights = [n / total_samples for n in sample_counts]

        # Determine fault tolerance f and candidate count m
        f = max(0, min(self.f, (K - 3) // 2))
        q = max(1, K - f - 2)
        # Standard Multi-Krum reference configuration: m = K - f
        m = self.m if self.m is not None else max(1, K - f)
        m = min(m, K)

        # 1. Flatten trainable parameter updates into vectors
        flat_vectors = []
        for i in range(K):
            t_vecs = [client_updates[i][1][idx].reshape(-1) for idx in TRAINABLE_INDICES]
            flat_vectors.append(np.concatenate(t_vecs).astype(np.float32))

        # 2. Compute pairwise squared Euclidean distances (excluding self-distance)
        dist_matrix = np.zeros((K, K), dtype=np.float32)
        for i in range(K):
            for j in range(i + 1, K):
                d = float(np.sum((flat_vectors[i] - flat_vectors[j]) ** 2))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # 3. Compute Krum score for each client (sum of q smallest non-zero squared distances)
        krum_scores = []
        for i in range(K):
            dists = np.sort([dist_matrix[i, j] for j in range(K) if j != i])
            score = float(np.sum(dists[:q]))
            krum_scores.append(score)

        # 4. Select top m candidates with lowest scores (deterministic stable tie-breaking)
        sorted_indices = np.argsort(krum_scores, kind="stable").tolist()
        selected_indices = sorted_indices[:m]
        included_mask = [i in selected_indices for i in range(K)]

        # Effective weights (equal 1/m among selected candidates)
        effective_weights = [1.0 / m if i in selected_indices else 0.0 for i in range(K)]

        # 5. Average trainable parameters over selected candidates
        num_layers = len(client_updates[0][1])
        aggregated_ndarrays = [None] * num_layers

        for idx in TRAINABLE_INDICES:
            layer_acc = np.zeros_like(client_updates[0][1][idx], dtype=np.float32)
            for i in selected_indices:
                layer_acc += (1.0 / m) * client_updates[i][1][idx].astype(np.float32)
            aggregated_ndarrays[idx] = layer_acc

        # 6. Aggregate BatchNorm Buffers over selected candidates
        bn_buffers = self.aggregate_bn_buffers(
            client_updates=client_updates,
            effective_weights=effective_weights,
            included_mask=included_mask,
        )
        for idx, buf in bn_buffers.items():
            aggregated_ndarrays[idx] = buf

        selected_cids = [client_updates[i][0] for i in selected_indices]
        excluded_cids = [client_updates[i][0] for i in range(K) if i not in selected_indices]

        metrics = {
            "aggregator": self.name,
            "f": f,
            "q_neighbors": q,
            "m_selected": m,
            "participating_clients": K,
            "selected_client_ids": selected_cids,
            "excluded_client_ids": excluded_cids,
            "krum_scores": {client_updates[i][0]: round(krum_scores[i], 4) for i in range(K)},
            "original_weights": {client_updates[i][0]: float(original_weights[i]) for i in range(K)},
            "effective_weights": {client_updates[i][0]: float(effective_weights[i]) for i in range(K)},
        }

        return aggregated_ndarrays, metrics
