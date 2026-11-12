"""
Model Poisoning Attack Module (Phase 5).
Simulates a compromised hospital participant that scales or inverts its
local gradient updates before transmission to the Flower FedAvg server.

Strict Rule: The adversarial transformation
    theta_adv = theta_global + gamma * (theta_local - theta_global)
is applied ONLY to trainable floating-point parameter tensors.
BatchNorm running buffers (running_mean, running_var) and integer buffers
(num_batches_tracked) are updated normally by local training and passed
through unmodified to standard FedAvg aggregation.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from federated_learning.client import HospitalClient


def apply_adversarial_update_transformation(
    model: nn.Module,
    local_ndarrays: List[np.ndarray],
    global_ndarrays: List[np.ndarray],
    gamma: float,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Apply update transformation theta_adv = theta_global + gamma * (theta_local - theta_global)
    strictly to trainable floating-point parameters.

    Non-trainable buffers (such as BatchNorm running statistics and integer step trackers)
    are preserved as computed by local training and NOT modified by gamma.

    Args:
        model: Reference PyTorch model architecture used to distinguish parameters from buffers.
        local_ndarrays: Parameter and buffer arrays returned by local client training.
        global_ndarrays: Parameter and buffer arrays received from the server at round start.
        gamma: Scaling / inversion factor (-1.0 for inversion, +10.0 for scaling).

    Returns:
        (transformed_ndarrays, metadata_summary)
    """
    state_keys = list(model.state_dict().keys())
    trainable_param_names = set(name for name, _ in model.named_parameters())

    assert len(state_keys) == len(local_ndarrays) == len(global_ndarrays), (
        f"Length mismatch: {len(state_keys)} keys vs {len(local_ndarrays)} local vs {len(global_ndarrays)} global"
    )

    transformed_ndarrays: List[np.ndarray] = []
    manipulated_tensors: List[str] = []
    preserved_buffers: List[str] = []

    clean_delta_sq_norm = 0.0
    adv_delta_sq_norm = 0.0

    for key, local_arr, global_arr in zip(state_keys, local_ndarrays, global_ndarrays):
        is_trainable = key in trainable_param_names

        if is_trainable:
            # Apply adversarial transformation
            diff = local_arr - global_arr
            adv_arr = global_arr + gamma * diff
            transformed_ndarrays.append(adv_arr.astype(local_arr.dtype, copy=True))
            manipulated_tensors.append(key)

            clean_delta_sq_norm += float(np.sum(diff ** 2))
            adv_delta_sq_norm += float(np.sum((adv_arr - global_arr) ** 2))
        else:
            # Preserve normal local training updates to running statistics
            transformed_ndarrays.append(local_arr.copy())
            preserved_buffers.append(key)

    metadata = {
        "gamma": float(gamma),
        "total_tensors": len(state_keys),
        "manipulated_tensor_count": len(manipulated_tensors),
        "preserved_buffer_count": len(preserved_buffers),
        "manipulated_tensors": manipulated_tensors,
        "preserved_buffers": preserved_buffers,
        "clean_update_l2_norm": float(np.sqrt(clean_delta_sq_norm)),
        "adversarial_update_l2_norm": float(np.sqrt(adv_delta_sq_norm)),
    }

    return transformed_ndarrays, metadata


class PoisonedHospitalClient(HospitalClient):
    """
    Compromised Hospital Client that mounts model poisoning via update scaling or inversion.
    """

    def __init__(
        self,
        client_id: str,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        local_epochs: int = 2,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        device: torch.device | None = None,
        gamma: float = -1.0,
        attack_start_round: int = 1,
    ) -> None:
        super().__init__(
            client_id=client_id,
            train_loader=train_loader,
            val_loader=val_loader,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
        )
        self.gamma = float(gamma)
        self.attack_start_round = int(attack_start_round)
        self.attack_history: List[Dict[str, Any]] = []

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Execute normal local training, then apply adversarial update transformation
        strictly to trainable floating-point parameters if server_round >= attack_start_round.
        """
        # Step 1: Normal local training (updates weights and BatchNorm running stats)
        local_ndarrays, num_examples, metrics = super().fit(parameters, config)

        # Step 2: Determine current round
        current_round = int(config.get("server_round", config.get("round", 1)))

        # Step 3: Conditionally apply adversarial transformation
        if current_round >= self.attack_start_round:
            transformed_ndarrays, meta = apply_adversarial_update_transformation(
                model=self.model,
                local_ndarrays=local_ndarrays,
                global_ndarrays=parameters,
                gamma=self.gamma,
            )
            meta["round"] = current_round
            self.attack_history.append(meta)

            metrics["attack_active"] = True
            metrics["gamma"] = self.gamma
            metrics["clean_update_norm"] = round(meta["clean_update_l2_norm"], 4)
            metrics["adv_update_norm"] = round(meta["adversarial_update_l2_norm"], 4)

            return transformed_ndarrays, num_examples, metrics
        else:
            metrics["attack_active"] = False
            return local_ndarrays, num_examples, metrics
