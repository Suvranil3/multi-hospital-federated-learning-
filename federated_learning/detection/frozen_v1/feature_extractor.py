"""
Phase 6: Multi-Factor Feature Extractor for Federated Learning.
Computes global, layer-wise, counterfactual validation, and historical trajectory features
from client updates during federated aggregation.
"""

from typing import Dict, List, Tuple, Any, Optional
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from flwr.server.strategy.aggregate import aggregate

from federated_learning.model import SmallCNN, get_model
from federated_learning.client import set_parameters_to_model

# Layer definitions strictly matching SmallCNN state_dict (18 keys: 12 trainable, 6 buffers)
# Total trainable parameters = 420,802. Total buffer elements = 194.
LAYER_CONFIG = OrderedDict([
    ("conv_block_1", {
        "param_names": ["features.0.weight", "features.0.bias", "features.1.weight", "features.1.bias"],
        "state_dict_indices": [0, 1, 2, 3],
        "expected_dim": 384,
    }),
    ("conv_block_2", {
        "param_names": ["features.4.weight", "features.4.bias", "features.5.weight", "features.5.bias"],
        "state_dict_indices": [7, 8, 9, 10],
        "expected_dim": 18624,
    }),
    ("dense_projection", {
        "param_names": ["classifier.1.weight", "classifier.1.bias"],
        "state_dict_indices": [14, 15],
        "expected_dim": 401536,
    }),
    ("classification_head", {
        "param_names": ["classifier.4.weight", "classifier.4.bias"],
        "state_dict_indices": [16, 17],
        "expected_dim": 258,
    }),
])

TRAINABLE_STATE_DICT_INDICES = [0, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 17]
BUFFER_STATE_DICT_INDICES = [4, 5, 6, 11, 12, 13]
TOTAL_TRAINABLE_PARAMS = sum(cfg["expected_dim"] for cfg in LAYER_CONFIG.values())  # 420,802


def flatten_tensors(param_list: List[np.ndarray], indices: List[int]) -> np.ndarray:
    """Flatten and concatenate selected tensors into a 1D NumPy vector."""
    chunks = [param_list[idx].ravel() for idx in indices]
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)


def compute_l2_norm(vec: np.ndarray) -> float:
    """Compute Euclidean (L2) norm of a vector."""
    norm = np.linalg.norm(vec)
    return float(norm) if np.isfinite(norm) else 0.0


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray, eps: float = 1e-9) -> float:
    """Compute cosine similarity between two 1D vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < eps or norm_b < eps:
        return 0.0
    sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
    return float(np.clip(sim, -1.0, 1.0))


def compute_modified_z_scores(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute robust Modified Z-Scores using Median and Median Absolute Deviation (MAD).
    Z_mod = 0.6745 * (x - median) / max(MAD, eps)
    """
    if len(values) == 0:
        return np.array([])
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    denom = max(mad, eps)
    return 0.6745 * (values - median) / denom


class FeatureExtractor:
    """
    Extracts multi-factor features from a cohort of client model updates at communication round t.
    """

    def __init__(
        self,
        val_loader: DataLoader,
        device: torch.device,
        ema_beta: float = 0.7,
    ) -> None:
        self.val_loader = val_loader
        self.device = device
        self.ema_beta = ema_beta
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.eval_model = get_model(in_channels=1, num_classes=2, dropout_rate=0.2).to(self.device)

        # Historical state buffers keyed by client_id string
        # Stores prior round updates and accumulated anomaly scores
        self.client_history: Dict[str, Dict[str, Any]] = {}

    def extract_round_features(
        self,
        server_round: int,
        client_updates: List[Tuple[str, List[np.ndarray], int]],
        global_params_prev: List[np.ndarray],
        global_params_full: List[np.ndarray],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract features for all clients in the current round.

        Args:
            server_round: Current communication round index.
            client_updates: List of tuples (client_id, updated_params_list, num_examples).
            global_params_prev: Global model parameters from round t-1.
            global_params_full: Full aggregated global model parameters at round t.

        Returns:
            Dictionary mapping client_id to its extracted feature dictionary.
        """
        client_ids = [cid for cid, _, _ in client_updates]
        client_weights = [params for _, params, _ in client_updates]
        num_examples_list = [n for _, _, n in client_updates]
        total_examples = sum(num_examples_list)

        # 1. Compute parameter difference vectors: Δw_i^t = w_i^t - w_global^{t-1}
        # Flattened for global network and for each of the 4 layers
        client_deltas_global: List[np.ndarray] = []
        client_deltas_by_layer: Dict[str, List[np.ndarray]] = {k: [] for k in LAYER_CONFIG}

        for params in client_weights:
            # Trainable delta across all 12 trainable tensors
            delta_trainable = [
                params[idx] - global_params_prev[idx] for idx in TRAINABLE_STATE_DICT_INDICES
            ]
            global_vec = np.concatenate([d.ravel() for d in delta_trainable])
            client_deltas_global.append(global_vec)

            # Layer-wise deltas
            for layer_name, layer_cfg in LAYER_CONFIG.items():
                layer_delta = [
                    params[idx] - global_params_prev[idx] for idx in layer_cfg["state_dict_indices"]
                ]
                layer_vec = np.concatenate([d.ravel() for d in layer_delta])
                client_deltas_by_layer[layer_name].append(layer_vec)

        # 2. Compute Robust Reference Updates (Coordinate-Wise Median)
        ref_global = np.median(np.stack(client_deltas_global, axis=0), axis=0)
        ref_by_layer = {
            layer_name: np.median(np.stack(deltas, axis=0), axis=0)
            for layer_name, deltas in client_deltas_by_layer.items()
        }

        # 3. Compute Magnitude & Directional Metrics across Peers
        # Global norms and Modified Z-Scores
        global_norms = np.array([compute_l2_norm(v) for v in client_deltas_global])
        global_z_scores = compute_modified_z_scores(global_norms)
        global_median_norm = float(np.median(global_norms))

        # Layer-wise norms and Modified Z-Scores
        layer_norms: Dict[str, np.ndarray] = {}
        layer_z_scores: Dict[str, np.ndarray] = {}
        for layer_name, deltas in client_deltas_by_layer.items():
            norms = np.array([compute_l2_norm(v) for v in deltas])
            layer_norms[layer_name] = norms
            layer_z_scores[layer_name] = compute_modified_z_scores(norms)

        # Cosine similarities
        global_cos_ref = [
            compute_cosine_similarity(v, ref_global) for v in client_deltas_global
        ]
        layer_cos_ref: Dict[str, List[float]] = {}
        for layer_name, deltas in client_deltas_by_layer.items():
            ref = ref_by_layer[layer_name]
            layer_cos_ref[layer_name] = [
                compute_cosine_similarity(v, ref) for v in deltas
            ]

        # Pairwise cosine similarities
        K = len(client_ids)
        pairwise_cos_matrix = np.zeros((K, K), dtype=np.float32)
        for i in range(K):
            for j in range(K):
                if i == j:
                    pairwise_cos_matrix[i, j] = 1.0
                elif i < j:
                    sim = compute_cosine_similarity(client_deltas_global[i], client_deltas_global[j])
                    pairwise_cos_matrix[i, j] = sim
                    pairwise_cos_matrix[j, i] = sim

        mean_peer_cos = []
        for i in range(K):
            peer_sims = [pairwise_cos_matrix[i, j] for j in range(K) if j != i]
            mean_peer_cos.append(float(np.mean(peer_sims)) if peer_sims else 1.0)

        # 4. Class-Disentangled Leave-One-Out (LOO) Proxy Validation Impact
        # Evaluate full model once
        full_metrics = self._evaluate_model(global_params_full)

        loo_metrics_by_client: List[Dict[str, float]] = []
        for i in range(K):
            # Construct w_{-i} using exact Phase 3 FedAvg weights for all 18 state_dict elements
            rem_examples = total_examples - num_examples_list[i]
            if rem_examples <= 0:
                loo_metrics_by_client.append(full_metrics)
                continue

            loo_weights_results = [
                (client_weights[j], num_examples_list[j]) for j in range(K) if j != i
            ]
            loo_params = aggregate(loo_weights_results)
            loo_m = self._evaluate_model(loo_params)
            loo_metrics_by_client.append(loo_m)

        # 5. Assemble Features & Update Historical Trajectories
        round_features: Dict[str, Dict[str, Any]] = {}

        for i, cid in enumerate(client_ids):
            loo_m = loo_metrics_by_client[i]

            # Delta metrics: Full - LOO
            # Positive delta loss means client i increased loss (degraded performance)
            # Positive delta recall means client i increased recall (improved sensitivity)
            delta_loss_total = float(full_metrics["val_loss"] - loo_m["val_loss"])
            delta_acc_total = float(full_metrics["val_acc"] - loo_m["val_acc"])
            delta_loss_normal = float(full_metrics["loss_normal"] - loo_m["loss_normal"])
            delta_loss_pneumonia = float(full_metrics["loss_pneumonia"] - loo_m["loss_pneumonia"])
            delta_rec_normal = float(full_metrics["recall_normal"] - loo_m["recall_normal"])
            delta_rec_pneumonia = float(full_metrics["recall_pneumonia"] - loo_m["recall_pneumonia"])

            # Maximum Modified Z-Score across all 4 functional layers
            max_layer_z = float(max(layer_z_scores[layer_name][i] for layer_name in LAYER_CONFIG))

            # Minimum layer-wise cosine similarity
            min_layer_cos = float(min(layer_cos_ref[layer_name][i] for layer_name in LAYER_CONFIG))

            # Historical velocity and trajectory alignment
            hist = self.client_history.setdefault(cid, {
                "prev_delta": None,
                "ema_accumulator": 0.0,
                "history_rounds": 0,
            })

            cur_delta = client_deltas_global[i]
            if hist["prev_delta"] is not None:
                velocity = compute_l2_norm(cur_delta - hist["prev_delta"])
                traj_cos = compute_cosine_similarity(cur_delta, hist["prev_delta"])
            else:
                velocity = 0.0
                traj_cos = 1.0

            # Instantaneous round anomaly indicator (preliminary composite)
            # Negative cosine, extreme norm, or double-class degradation
            inst_anomaly = 0.0
            if global_cos_ref[i] < 0.0:
                inst_anomaly += 0.8
            if global_z_scores[i] > 3.0 or max_layer_z > 3.5:
                inst_anomaly += 0.8
            if delta_loss_total > 0.05 and delta_rec_normal <= 0.0 and delta_rec_pneumonia <= 0.0:
                inst_anomaly += 0.5
            inst_anomaly = min(inst_anomaly, 1.0)

            # Update EMA accumulator
            ema_acc = self.ema_beta * hist["ema_accumulator"] + (1.0 - self.ema_beta) * inst_anomaly
            hist["ema_accumulator"] = ema_acc
            hist["prev_delta"] = cur_delta.copy()
            hist["history_rounds"] += 1

            client_record = {
                "client_id": cid,
                "round": server_round,
                "num_examples": num_examples_list[i],
                # Magnitude features
                "norm_global": float(global_norms[i]),
                "z_mag_global": float(global_z_scores[i]),
                "norm_ratio_global": float(global_norms[i] / max(global_median_norm, 1e-8)),
                "max_layer_z_mag": max_layer_z,
                "layer_norms": {
                    lname: float(layer_norms[lname][i]) for lname in LAYER_CONFIG
                },
                "layer_z_mag": {
                    lname: float(layer_z_scores[lname][i]) for lname in LAYER_CONFIG
                },
                # Directional features
                "cos_ref_global": float(global_cos_ref[i]),
                "mean_peer_cos": float(mean_peer_cos[i]),
                "min_layer_cos": min_layer_cos,
                "layer_cos_ref": {
                    lname: float(layer_cos_ref[lname][i]) for lname in LAYER_CONFIG
                },
                "is_sign_inverted": bool(global_cos_ref[i] < 0.0),
                # Class-disentangled counterfactual validation impacts
                "delta_loss_total": round(delta_loss_total, 6),
                "delta_acc_total": round(delta_acc_total, 4),
                "delta_loss_normal": round(delta_loss_normal, 6),
                "delta_loss_pneumonia": round(delta_loss_pneumonia, 6),
                "delta_rec_normal": round(delta_rec_normal, 6),
                "delta_rec_pneumonia": round(delta_rec_pneumonia, 6),
                # Historical trajectory features
                "velocity": round(float(velocity), 6),
                "trajectory_cos": round(float(traj_cos), 4),
                "ema_anomaly_accumulator": round(float(ema_acc), 4),
                "history_rounds": hist["history_rounds"],
            }

            round_features[cid] = client_record

        return round_features

    def _evaluate_model(self, parameters: List[np.ndarray]) -> Dict[str, float]:
        """
        Evaluate a candidate model state on the pristine validation split (N=524).
        Disentangles cross-entropy loss and recall for Normal (class 0) and Pneumonia (class 1).
        """
        set_parameters_to_model(self.eval_model, parameters)
        self.eval_model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        normal_loss_sum = 0.0
        normal_samples = 0
        normal_correct = 0

        pneumonia_loss_sum = 0.0
        pneumonia_samples = 0
        pneumonia_correct = 0

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.squeeze(-1).long().to(self.device, non_blocking=True)

                logits = self.eval_model(images)
                losses = self.criterion(logits, targets)

                total_loss += losses.sum().item()
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == targets).sum().item()
                total_samples += targets.size(0)

                # Class-specific breakdown
                normal_mask = (targets == 0)
                if normal_mask.any():
                    normal_loss_sum += losses[normal_mask].sum().item()
                    normal_samples += normal_mask.sum().item()
                    normal_correct += (preds[normal_mask] == 0).sum().item()

                pneumonia_mask = (targets == 1)
                if pneumonia_mask.any():
                    pneumonia_loss_sum += losses[pneumonia_mask].sum().item()
                    pneumonia_samples += pneumonia_mask.sum().item()
                    pneumonia_correct += (preds[pneumonia_mask] == 1).sum().item()

        return {
            "val_loss": total_loss / max(total_samples, 1),
            "val_acc": (total_correct / max(total_samples, 1)) * 100.0,
            "loss_normal": normal_loss_sum / max(normal_samples, 1),
            "recall_normal": normal_correct / max(normal_samples, 1),
            "loss_pneumonia": pneumonia_loss_sum / max(pneumonia_samples, 1),
            "recall_pneumonia": pneumonia_correct / max(pneumonia_samples, 1),
        }
