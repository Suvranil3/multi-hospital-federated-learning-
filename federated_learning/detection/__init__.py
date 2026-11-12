"""
Phase 6: Multi-Factor Non-IID-Aware Malicious Hospital Detection Module.
Provides layer-wise feature extraction, quantile-based calibration,
symmetrical clinical trade-off protection, and tri-state anomaly classification.
"""

from federated_learning.detection.feature_extractor import (
    FeatureExtractor,
    LAYER_CONFIG,
    TOTAL_TRAINABLE_PARAMS,
)
from federated_learning.detection.detector import (
    TriStateDetector,
    DetectionDecision,
    ClientEvaluationResult,
)
from federated_learning.detection.metrics import (
    compute_detection_metrics,
    compute_stratified_fpr,
)

__all__ = [
    "FeatureExtractor",
    "LAYER_CONFIG",
    "TOTAL_TRAINABLE_PARAMS",
    "TriStateDetector",
    "DetectionDecision",
    "ClientEvaluationResult",
    "compute_detection_metrics",
    "compute_stratified_fpr",
]
