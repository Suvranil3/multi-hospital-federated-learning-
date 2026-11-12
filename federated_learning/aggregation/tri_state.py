"""
Tri-State Consensus Aggregators for Phase 7.
Integrates the frozen tri-state detector into federated aggregation with three variants:
1. Tri-State Monitor-Only: Logs detector decisions; applies standard FedAvg weighting (no exclusions/discounts).
2. Tri-State Discount 0.50: Excludes MALICIOUS; applies 0.50 discount to SUSPICIOUS; NORMAL unscaled.
3. Tri-State Discount 0.25: Excludes MALICIOUS; applies 0.25 discount to SUSPICIOUS; NORMAL unscaled.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from federated_learning.aggregation.base import (
    BaseAggregator,
    TRAINABLE_INDICES,
)


class TriStateConsensusAggregator(BaseAggregator):
    """
    Tri-State Consensus Aggregator.
    Parameters:
        mode: 'monitor_only', 'discount_050', or 'discount_025'
        suspicious_discount: Multiplier applied to SUSPICIOUS clients (1.0, 0.50, or 0.25).
    """

    def __init__(self, mode: str = "discount_050", suspicious_discount: Optional[float] = None) -> None:
        super().__init__(name=f"tristate_{mode}")
        self.mode = mode
        if suspicious_discount is not None:
            self.suspicious_discount = float(suspicious_discount)
        elif mode == "monitor_only":
            self.suspicious_discount = 1.0
        elif mode == "discount_025":
            self.suspicious_discount = 0.25
        else:  # default discount_050
            self.suspicious_discount = 0.50

    def aggregate(
        self,
        client_updates: List[Tuple[str, List[np.ndarray], int]],
        global_params_prev: Optional[List[np.ndarray]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        if not client_updates:
            if global_params_prev is not None:
                return [arr.copy() for arr in global_params_prev], {"status": "fallback_empty"}
            raise ValueError("No client updates provided for Tri-State aggregation.")

        K = len(client_updates)
        sample_counts = [u[2] for u in client_updates]
        total_samples = max(sum(sample_counts), 1)
        original_weights = [n / total_samples for n in sample_counts]

        # Extract detector decisions from context
        decisions: Dict[str, str] = {}
        anomaly_scores: Dict[str, float] = {}
        if context and "detector_results" in context:
            for cid, res in context["detector_results"].items():
                if hasattr(res, "decision"):
                    d = res.decision
                    decisions[str(cid)] = str(d.value if hasattr(d, "value") else d)
                elif isinstance(res, dict):
                    d = res.get("decision", "NORMAL")
                    decisions[str(cid)] = str(d.value if hasattr(d, "value") else d)
                else:
                    decisions[str(cid)] = "NORMAL"

                if hasattr(res, "anomaly_score"):
                    anomaly_scores[str(cid)] = float(res.anomaly_score)
                elif isinstance(res, dict):
                    anomaly_scores[str(cid)] = float(res.get("anomaly_score", 0.0))
                else:
                    anomaly_scores[str(cid)] = 0.0
        else:
            # Default to NORMAL if no detector context provided
            for u in client_updates:
                decisions[str(u[0])] = "NORMAL"
                anomaly_scores[str(u[0])] = 0.0

        # Compute raw multiplier per client
        multipliers = []
        for i, (cid, _, _) in enumerate(client_updates):
            dec = decisions.get(str(cid), "NORMAL")
            if self.mode == "monitor_only":
                multipliers.append(1.0)
            else:
                if dec == "MALICIOUS":
                    multipliers.append(0.0)
                elif dec == "SUSPICIOUS":
                    multipliers.append(self.suspicious_discount)
                else:  # NORMAL
                    multipliers.append(1.0)

        # Unnormalized weights
        raw_weights = [original_weights[i] * multipliers[i] for i in range(K)]
        sum_raw_weights = sum(raw_weights)

        # Edge Case 1: All clients are classified MALICIOUS (sum_raw_weights == 0)
        if sum_raw_weights <= 1e-12:
            all_malicious = all(decisions.get(str(u[0])) == "MALICIOUS" for u in client_updates)
            if global_params_prev is not None:
                fallback_params = [arr.copy() for arr in global_params_prev]
                metrics = {
                    "aggregator": self.name,
                    "mode": self.mode,
                    "suspicious_discount": self.suspicious_discount,
                    "participating_clients": K,
                    "included_clients": 0,
                    "fallback_to_previous_round": True,
                    "all_clients_malicious": all_malicious,
                    "security_alert": "ALERT: All participating clients evaluated MALICIOUS. Rejecting all client updates and preserving previous global model parameters.",
                    "original_weights": {client_updates[i][0]: float(original_weights[i]) for i in range(K)},
                    "effective_weights": {client_updates[i][0]: 0.0 for i in range(K)},
                    "decisions": decisions,
                    "anomaly_scores": anomaly_scores,
                }
                return fallback_params, metrics
            else:
                # If no previous params, fallback to equal weighting with warning
                effective_weights = [1.0 / K] * K
        else:
            # Normalize effective weights so they sum strictly to 1.0
            effective_weights = [w / sum_raw_weights for w in raw_weights]

        # Edge Case 2: All clients are SUSPICIOUS
        all_suspicious = all(decisions.get(str(u[0])) == "SUSPICIOUS" for u in client_updates)

        # Included mask
        included_mask = [w > 1e-9 for w in effective_weights]
        included_cids = [client_updates[i][0] for i in range(K) if included_mask[i]]
        excluded_cids = [client_updates[i][0] for i in range(K) if not included_mask[i]]

        # Aggregate Trainable Parameters
        num_layers = len(client_updates[0][1])
        aggregated_ndarrays = [None] * num_layers

        for idx in TRAINABLE_INDICES:
            layer_acc = np.zeros_like(client_updates[0][1][idx], dtype=np.float32)
            for i, (_, params, _) in enumerate(client_updates):
                if effective_weights[i] > 1e-9:
                    layer_acc += effective_weights[i] * params[idx].astype(np.float32)
            aggregated_ndarrays[idx] = layer_acc

        # Aggregate BatchNorm Buffers over included clients
        bn_buffers = self.aggregate_bn_buffers(
            client_updates=client_updates,
            effective_weights=effective_weights,
            included_mask=included_mask,
        )
        for idx, buf in bn_buffers.items():
            aggregated_ndarrays[idx] = buf

        metrics = {
            "aggregator": self.name,
            "mode": self.mode,
            "suspicious_discount": self.suspicious_discount,
            "participating_clients": K,
            "included_clients": len(included_cids),
            "selected_client_ids": included_cids,
            "excluded_client_ids": excluded_cids,
            "fallback_to_previous_round": False,
            "all_clients_suspicious": all_suspicious,
            "original_weights": {client_updates[i][0]: float(original_weights[i]) for i in range(K)},
            "effective_weights": {client_updates[i][0]: float(effective_weights[i]) for i in range(K)},
            "decisions": decisions,
            "anomaly_scores": anomaly_scores,
        }

        return aggregated_ndarrays, metrics
