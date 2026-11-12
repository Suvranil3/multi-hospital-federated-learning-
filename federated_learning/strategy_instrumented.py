"""
Phase 6: Instrumented Federated Strategy Module.
Extends Flower FedAvg with zero-impact, non-invasive feature extraction and logging.
Preserves exact standard FedAvg aggregation math and parameter outputs.
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
from flwr.server.strategy.aggregate import aggregate

from federated_learning.model import get_model
from federated_learning.client import set_parameters_to_model, get_parameters_from_model
from federated_learning.detection.feature_extractor import FeatureExtractor


class InstrumentedFedAvg(fl.server.strategy.FedAvg):
    """
    Instrumented FedAvg Strategy that extracts multi-factor layer-wise and
    counterfactual validation features at each communication round.
    Zero-impact guarantee: standard FedAvg parameter aggregation is strictly preserved.
    """

    def __init__(
        self,
        val_loader: DataLoader,
        device: torch.device,
        output_jsonl_path: Optional[Path] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.val_loader = val_loader
        self.device = device
        self.output_jsonl_path = Path(output_jsonl_path) if output_jsonl_path else None

        if self.output_jsonl_path:
            self.output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            # Clear or initialize JSONL file
            with open(self.output_jsonl_path, "w", encoding="utf-8") as f:
                pass

        self.feature_extractor = FeatureExtractor(val_loader=val_loader, device=device)
        self.extracted_records: List[Dict[str, Any]] = []

        # Track previous round's global parameters (starts from initial_parameters)
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
        """
        Aggregate model weights using exact standard FedAvg while extracting
        per-client layer-wise, counterfactual validation, and trajectory features.
        """
        if not results:
            return None, {}

        # 1. Unpack client updates for feature extraction
        client_updates: List[Tuple[str, List[np.ndarray], int]] = []
        for client_proxy, fit_res in results:
            cid = str(fit_res.metrics.get("client_id", client_proxy.cid))
            client_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            num_examples = int(fit_res.num_examples)
            client_updates.append((cid, client_ndarrays, num_examples))

        # Sort deterministically by client_id for reproducible feature tracking
        client_updates.sort(key=lambda x: int(x[0]) if x[0].isdigit() else x[0])

        # 2. Standard FedAvg parameter aggregation via canonical super().aggregate_fit()
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round=server_round,
            results=results,
            failures=failures,
        )
        if aggregated_parameters is None:
            return None, aggregated_metrics

        weighted_weights = parameters_to_ndarrays(aggregated_parameters)

        # 3. Extract multi-factor features for all clients
        round_features = self.feature_extractor.extract_round_features(
            server_round=server_round,
            client_updates=client_updates,
            global_params_prev=self.global_params_prev,
            global_params_full=weighted_weights,
        )

        # 4. Stream feature records to memory and JSONL
        for cid, record in round_features.items():
            self.extracted_records.append(record)
            if self.output_jsonl_path:
                with open(self.output_jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

        # 5. Update previous round global parameters for round t+1
        self.global_params_prev = [arr.copy() for arr in weighted_weights]

        return aggregated_parameters, aggregated_metrics
