"""
Base Aggregation Module for Phase 7 (Byzantine-Robust Aggregation).
Defines abstract aggregator interface and model-state handling rules:
- Trainable parameters (12 tensors)
- BatchNorm running float buffers (4 tensors: running_mean, running_var)
- BatchNorm integer counters (2 scalars: num_batches_tracked)
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

# Exact SmallCNN state_dict index specifications (18 items total)
# 0: features.0.weight (conv1)
# 1: features.0.bias (conv1)
# 2: features.1.weight (bn1 weight)
# 3: features.1.bias (bn1 bias)
# 4: features.1.running_mean (bn1 running mean)
# 5: features.1.running_var (bn1 running var)
# 6: features.1.num_batches_tracked (bn1 int counter)
# 7: features.4.weight (conv2)
# 8: features.4.bias (conv2)
# 9: features.5.weight (bn2 weight)
# 10: features.5.bias (bn2 bias)
# 11: features.5.running_mean (bn2 running mean)
# 12: features.5.running_var (bn2 running var)
# 13: features.5.num_batches_tracked (bn2 int counter)
# 14: classifier.1.weight (fc1)
# 15: classifier.1.bias (fc1)
# 16: classifier.4.weight (fc2 output)
# 17: classifier.4.bias (fc2 output)

TRAINABLE_INDICES = [0, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 17]
BN_FLOAT_INDICES = [4, 5, 11, 12]
BN_INT_INDICES = [6, 13]


class BaseAggregator(ABC):
    """
    Abstract Base Class for Byzantine-Robust Federated Model Aggregation.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def aggregate(
        self,
        client_updates: List[Tuple[str, List[np.ndarray], int]],
        global_params_prev: Optional[List[np.ndarray]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Aggregate client parameter updates.

        Args:
            client_updates: List of (client_id, parameters, num_examples)
            global_params_prev: Previous round global parameters (fallback if needed)
            context: Additional round context (e.g. features, detector decisions)

        Returns:
            Tuple of (aggregated_ndarrays, aggregation_metrics)
        """
        pass

    def aggregate_bn_buffers(
        self,
        client_updates: List[Tuple[str, List[np.ndarray], int]],
        effective_weights: Optional[List[float]] = None,
        included_mask: Optional[List[bool]] = None,
    ) -> Dict[int, np.ndarray]:
        """
        Aggregate BatchNorm buffers via sample/effective-weighted linear combination:
        Used by FedAvg, Multi-Krum (over selected candidates), and Tri-State Consensus.
        - Floating-point running means & variances: weighted linear combination over included clients.
        - Integer counter (num_batches_tracked): advance by sample-weighted average local steps.
        """
        K = len(client_updates)
        if included_mask is None:
            included_mask = [True] * K

        included_indices = [i for i, inc in enumerate(included_mask) if inc]
        if not included_indices:
            included_indices = list(range(K))

        if effective_weights is None:
            included_samples = [client_updates[i][2] for i in included_indices]
            tot_samples = max(sum(included_samples), 1)
            weights = {i: client_updates[i][2] / tot_samples for i in included_indices}
        else:
            tot_w = sum(effective_weights[i] for i in included_indices)
            if tot_w > 0:
                weights = {i: effective_weights[i] / tot_w for i in included_indices}
            else:
                weights = {i: 1.0 / len(included_indices) for i in included_indices}

        bn_aggregated = {}

        # 1. Floating-point running means and variances
        for idx in BN_FLOAT_INDICES:
            agg_stat = np.zeros_like(client_updates[0][1][idx], dtype=np.float32)
            for i in included_indices:
                w = weights[i]
                agg_stat += w * client_updates[i][1][idx].astype(np.float32)
            bn_aggregated[idx] = agg_stat

        # 2. Integer batch counters (num_batches_tracked)
        for idx in BN_INT_INDICES:
            avg_batches = sum(weights[i] * float(client_updates[i][1][idx].item()) for i in included_indices)
            bn_aggregated[idx] = np.array(int(round(avg_batches)), dtype=np.int64)

        return bn_aggregated

    def aggregate_bn_buffers_median(
        self,
        client_updates: List[Tuple[str, List[np.ndarray], int]],
    ) -> Dict[int, np.ndarray]:
        """
        Coordinate-wise Median aggregation for BatchNorm floating-point buffers.
        Used by CoordinateMedianAggregator to prevent buffer poisoning.
        - Running means & variances: coordinate-wise median across all participating clients.
        - Integer counter: sample-weighted average local steps.
        """
        K = len(client_updates)
        sample_counts = [u[2] for u in client_updates]
        tot_samples = max(sum(sample_counts), 1)
        weights = [n / tot_samples for n in sample_counts]

        bn_aggregated = {}

        # 1. Coordinate-wise median on running_mean and running_var
        for idx in BN_FLOAT_INDICES:
            stacked_bn = np.stack([client_updates[i][1][idx] for i in range(K)], axis=0)
            bn_aggregated[idx] = np.median(stacked_bn, axis=0).astype(np.float32)

        # 2. Integer batch counter: sample-weighted average local steps
        for idx in BN_INT_INDICES:
            avg_batches = sum(weights[i] * float(client_updates[i][1][idx].item()) for i in range(K))
            bn_aggregated[idx] = np.array(int(round(avg_batches)), dtype=np.int64)

        return bn_aggregated

    def aggregate_bn_buffers_trimmed(
        self,
        client_updates: List[Tuple[str, List[np.ndarray], int]],
        beta: int = 1,
    ) -> Dict[int, np.ndarray]:
        """
        Coordinate-wise Trimmed Mean aggregation for BatchNorm floating-point buffers.
        Used by TrimmedMeanAggregator to prevent buffer poisoning.
        - Running means & variances: coordinate-wise trimmed mean with beta across all clients.
        - Integer counter: sample-weighted average local steps.
        """
        K = len(client_updates)
        effective_beta = min(int(beta), (K - 1) // 2)
        sample_counts = [u[2] for u in client_updates]
        tot_samples = max(sum(sample_counts), 1)
        weights = [n / tot_samples for n in sample_counts]

        bn_aggregated = {}

        # 1. Coordinate-wise trimmed mean on running_mean and running_var
        for idx in BN_FLOAT_INDICES:
            stacked_bn = np.stack([client_updates[i][1][idx] for i in range(K)], axis=0)
            if effective_beta > 0:
                sorted_bn = np.sort(stacked_bn, axis=0)
                trimmed_bn = sorted_bn[effective_beta : K - effective_beta]
                bn_aggregated[idx] = np.mean(trimmed_bn, axis=0).astype(np.float32)
            else:
                bn_aggregated[idx] = np.mean(stacked_bn, axis=0).astype(np.float32)

        # 2. Integer batch counter: sample-weighted average local steps
        for idx in BN_INT_INDICES:
            avg_batches = sum(weights[i] * float(client_updates[i][1][idx].item()) for i in range(K))
            bn_aggregated[idx] = np.array(int(round(avg_batches)), dtype=np.int64)

        return bn_aggregated
