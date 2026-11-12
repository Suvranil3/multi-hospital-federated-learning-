"""
Phase 7: Byzantine-Robust Aggregation and Non-IID-Aware Tri-State Consensus.
Exposes 7 aggregation variants:
1. Standard FedAvg
2. Coordinate-wise Median
3. Trimmed Mean
4. Multi-Krum
5. Tri-State Monitor-Only
6. Tri-State Suspicious Weight Discount = 0.50
7. Tri-State Suspicious Weight Discount = 0.25
"""

from federated_learning.aggregation.base import (
    BaseAggregator,
    TRAINABLE_INDICES,
    BN_FLOAT_INDICES,
    BN_INT_INDICES,
)
from federated_learning.aggregation.fedavg import FedAvgAggregator
from federated_learning.aggregation.coordinate_median import CoordinateMedianAggregator
from federated_learning.aggregation.trimmed_mean import TrimmedMeanAggregator
from federated_learning.aggregation.multi_krum import MultiKrumAggregator
from federated_learning.aggregation.tri_state import TriStateConsensusAggregator
from federated_learning.aggregation.strategies import (
    RobustAggregatedStrategy,
    create_robust_strategy,
)

__all__ = [
    "BaseAggregator",
    "TRAINABLE_INDICES",
    "BN_FLOAT_INDICES",
    "BN_INT_INDICES",
    "FedAvgAggregator",
    "CoordinateMedianAggregator",
    "TrimmedMeanAggregator",
    "MultiKrumAggregator",
    "TriStateConsensusAggregator",
    "RobustAggregatedStrategy",
    "create_robust_strategy",
]
