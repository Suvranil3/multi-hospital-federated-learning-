"""
Federated Strategy and Evaluation Module (Phase 3: 10-Hospital IID Baseline).
Configures Flower FedAvg aggregation and server-side clinical validation evaluation.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import torch
import torch.nn as nn
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

from federated_learning.model import SmallCNN, get_model
from federated_learning.client import set_parameters_to_model
from experiments.train_medical import evaluate, compute_medical_metrics


def aggregate_fit_metrics(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """
    Weighted average of client training metrics across all participating hospitals.
    """
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    weighted_loss = sum(num_examples * float(m.get("train_loss", 0.0)) for num_examples, m in metrics)
    weighted_acc = sum(num_examples * float(m.get("train_accuracy", 0.0)) for num_examples, m in metrics)

    return {
        "train_loss": round(weighted_loss / total_examples, 4),
        "train_accuracy": round(weighted_acc / total_examples, 2),
    }


def get_server_eval_fn(
    val_loader: DataLoader,
    device: torch.device,
    record_history: Optional[List[Dict[str, Any]]] = None,
) -> Callable[[int, List[np.ndarray], Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]:
    """
    Generate an evaluation function for server-side evaluation on the official validation split.
    Invoked by Flower after each aggregation round.
    """
    model = get_model(in_channels=1, num_classes=2, dropout_rate=0.2).to(device)
    criterion = nn.CrossEntropyLoss()

    def evaluate_fn(
        server_round: int,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        set_parameters_to_model(model, parameters)
        val_loss, val_acc, targets, preds = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            return_predictions=True,
        )

        med_metrics = compute_medical_metrics(targets, preds)
        p_metrics = med_metrics["pneumonia_class_1"]
        m_metrics = med_metrics["macro_averaged"]

        eval_summary = {
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 2),
            "pneumonia_recall": round(p_metrics["recall_sensitivity"], 4),
            "pneumonia_precision": round(p_metrics["precision"], 4),
            "pneumonia_f1": round(p_metrics["f1_score"], 4),
            "macro_f1": round(m_metrics["macro_f1"], 4),
        }

        if record_history is not None:
            record_history.append({
                "round": server_round,
                **eval_summary,
            })

        return val_loss, eval_summary

    return evaluate_fn
