"""
Phase 7 Flower Strategy Module.
Extends Flower FedAvg with modular robust aggregation algorithms and comprehensive round telemetry.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from federated_learning.model import get_model
from federated_learning.client import get_parameters_from_model
from federated_learning.detection.feature_extractor import FeatureExtractor
from federated_learning.aggregation.base import BaseAggregator
from federated_learning.aggregation.fedavg import FedAvgAggregator
from federated_learning.aggregation.coordinate_median import CoordinateMedianAggregator
from federated_learning.aggregation.trimmed_mean import TrimmedMeanAggregator
from federated_learning.aggregation.multi_krum import MultiKrumAggregator
from federated_learning.aggregation.tri_state import TriStateConsensusAggregator


class RobustAggregatedStrategy(fl.server.strategy.FedAvg):
    """
    Flower Strategy supporting arbitrary Byzantine-robust aggregators with
    full round feature extraction, detector evaluation, and telemetry streaming.
    """

    def __init__(
        self,
        aggregator: BaseAggregator,
        val_loader: DataLoader,
        device: torch.device,
        frozen_detector: Optional[Any] = None,
        output_jsonl_path: Optional[Path] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.aggregator = aggregator
        self.val_loader = val_loader
        self.device = device
        self.frozen_detector = frozen_detector
        self.output_jsonl_path = Path(output_jsonl_path) if output_jsonl_path else None

        if self.output_jsonl_path:
            self.output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_jsonl_path, "w", encoding="utf-8") as f:
                pass

        self.feature_extractor = FeatureExtractor(val_loader=val_loader, device=device)
        self.extracted_records: List[Dict[str, Any]] = []
        self.round_telemetry: List[Dict[str, Any]] = []

        if self.initial_parameters is not None:
            self.global_params_prev = parameters_to_ndarrays(self.initial_parameters)
        else:
            dummy_m = get_model()
            self.global_params_prev = get_parameters_from_model(dummy_m)

    def reset(self) -> None:
        """Reset internal feature extractor history and round telemetry."""
        self.feature_extractor.reset()
        self.extracted_records.clear()
        self.round_telemetry.clear()
        if self.initial_parameters is not None:
            self.global_params_prev = parameters_to_ndarrays(self.initial_parameters)
        else:
            dummy_m = get_model()
            self.global_params_prev = get_parameters_from_model(dummy_m)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # 1. Unpack client updates
        client_updates: List[Tuple[str, List[np.ndarray], int]] = []
        for client_proxy, fit_res in results:
            cid = str(fit_res.metrics.get("client_id", client_proxy.cid))
            client_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            num_examples = int(fit_res.num_examples)
            client_updates.append((cid, client_ndarrays, num_examples))

        # Sort deterministically by client_id
        client_updates.sort(key=lambda x: int(x[0]) if x[0].isdigit() else x[0])

        # 2. Compute canonical unweighted FedAvg weights for reference feature extraction
        tot_samples = max(sum(u[2] for u in client_updates), 1)
        ref_weights = [u[2] / tot_samples for u in client_updates]
        ref_full_params = [
            sum(ref_weights[i] * client_updates[i][1][l] for i in range(len(client_updates)))
            for l in range(len(client_updates[0][1]))
        ]

        # 3. Extract multi-factor features for all clients
        round_features = self.feature_extractor.extract_round_features(
            server_round=server_round,
            client_updates=client_updates,
            global_params_prev=self.global_params_prev,
            global_params_full=ref_full_params,
        )

        # 4. Evaluate frozen detector if provided
        detector_results: Dict[str, Any] = {}
        if self.frozen_detector is not None:
            for cid, rec in round_features.items():
                detector_results[str(cid)] = self.frozen_detector.evaluate_client(rec)

        # 5. Apply selected robust aggregator
        context = {
            "server_round": server_round,
            "round_features": round_features,
            "detector_results": detector_results,
        }

        aggregated_ndarrays, agg_metrics = self.aggregator.aggregate(
            client_updates=client_updates,
            global_params_prev=self.global_params_prev,
            context=context,
        )

        # 6. Stream feature records + decisions + weights to memory and JSONL
        for cid, record in round_features.items():
            enhanced_record = dict(record)
            if cid in detector_results:
                det_res = detector_results[cid]
                enhanced_record["detector_decision"] = det_res.decision.value if hasattr(det_res, "decision") else str(det_res.get("decision"))
                enhanced_record["anomaly_score"] = float(det_res.anomaly_score if hasattr(det_res, "anomaly_score") else det_res.get("anomaly_score", 0.0))
            if "original_weights" in agg_metrics:
                enhanced_record["original_weight"] = agg_metrics["original_weights"].get(cid, 0.0)
            if "effective_weights" in agg_metrics:
                enhanced_record["effective_weight"] = agg_metrics["effective_weights"].get(cid, 0.0)

            self.extracted_records.append(enhanced_record)
            if self.output_jsonl_path:
                with open(self.output_jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(enhanced_record) + "\n")

        # 7. Update previous global parameters for next round
        self.global_params_prev = [arr.copy() for arr in aggregated_ndarrays]

        # Telemetry record
        telemetry = {
            "round": server_round,
            "aggregator": self.aggregator.name,
            **agg_metrics,
        }
        self.round_telemetry.append(telemetry)

        # Metrics for Flower
        summary_metrics: Dict[str, Scalar] = {
            "aggregator": self.aggregator.name,
            "included_clients": int(agg_metrics.get("included_clients", len(client_updates))),
        }

        return ndarrays_to_parameters(aggregated_ndarrays), summary_metrics


def create_robust_strategy(
    strategy_name: str,
    val_loader: DataLoader,
    device: torch.device,
    frozen_detector: Optional[Any] = None,
    output_jsonl_path: Optional[Path] = None,
    initial_parameters: Optional[Parameters] = None,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 0.0,
    min_fit_clients: int = 10,
    min_available_clients: int = 10,
    fit_metrics_aggregation_fn: Optional[Callable] = None,
    on_fit_config_fn: Optional[Callable] = None,
    evaluate_fn: Optional[Callable] = None,
    krum_f: int = 1,
    krum_m: Optional[int] = None,
    trimmed_beta: int = 1,
) -> RobustAggregatedStrategy:
    """
    Factory creating one of the 7 Phase 7 robust aggregation strategies.
    Supported strategy_name values:
        - 'fedavg'
        - 'coordinate_median'
        - 'trimmed_mean'
        - 'multi_krum'
        - 'tristate_monitor'
        - 'tristate_discount_050'
        - 'tristate_discount_025'
    """
    s_name = strategy_name.lower().replace("-", "_")

    if s_name == "fedavg":
        aggregator = FedAvgAggregator()
    elif s_name == "coordinate_median" or s_name == "median":
        aggregator = CoordinateMedianAggregator()
    elif s_name in ["trimmed_mean", "trimmed_mean_beta1", "trimmed_beta1"]:
        aggregator = TrimmedMeanAggregator(beta=1)
    elif s_name in ["trimmed_mean_beta2", "trimmed_beta2"]:
        aggregator = TrimmedMeanAggregator(beta=2)
    elif s_name in ["multi_krum", "multi_krum_f1", "krum_f1", "krum"]:
        f_val = 1 if s_name in ["multi_krum_f1", "krum_f1"] else krum_f
        aggregator = MultiKrumAggregator(f=f_val, m=krum_m)
    elif s_name in ["multi_krum_f2", "krum_f2"]:
        aggregator = MultiKrumAggregator(f=2, m=krum_m)
    elif s_name in ["tri_state_monitor", "tristate_monitor", "tri_state_monitor_only", "tristate_monitor_only"]:
        aggregator = TriStateConsensusAggregator(mode="monitor_only")
    elif s_name in ["tri_state_discount_050", "tristate_discount_050", "tri_state_discount_50", "tristate_discount_50", "tri_state_050", "tristate_050"]:
        aggregator = TriStateConsensusAggregator(mode="discount_050", suspicious_discount=0.50)
    elif s_name in ["tri_state_discount_025", "tristate_discount_025", "tri_state_discount_25", "tristate_discount_25", "tri_state_025", "tristate_025"]:
        aggregator = TriStateConsensusAggregator(mode="discount_025", suspicious_discount=0.25)
    else:
        raise ValueError(f"Unknown robust aggregation strategy: {strategy_name}")

    return RobustAggregatedStrategy(
        aggregator=aggregator,
        val_loader=val_loader,
        device=device,
        frozen_detector=frozen_detector,
        output_jsonl_path=output_jsonl_path,
        initial_parameters=initial_parameters,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        on_fit_config_fn=on_fit_config_fn,
        evaluate_fn=evaluate_fn,
    )
