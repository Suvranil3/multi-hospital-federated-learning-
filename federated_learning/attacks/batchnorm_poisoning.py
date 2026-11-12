"""
BatchNorm Poisoning Attack Module (Phase 7).
Simulates a compromised hospital participant that leaves all trainable parameters
normal and close to honest peer updates, but replaces its local BatchNorm
floating-point running statistics (running_mean, running_var) with extreme outliers (1e6).
Integer counters (num_batches_tracked) remain valid.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from federated_learning.client import HospitalClient
from federated_learning.aggregation.base import BN_FLOAT_INDICES, BN_INT_INDICES, TRAINABLE_INDICES


def apply_batchnorm_poisoning_transformation(
    local_ndarrays: List[np.ndarray],
    poison_value: float = 1_000_000.0,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Apply extreme outlier poisoning strictly to floating-point BatchNorm running buffers.
    Trainable parameters and integer counters are preserved unmodified.
    """
    transformed = [arr.copy() for arr in local_ndarrays]
    poisoned_indices = []

    for idx in BN_FLOAT_INDICES:
        transformed[idx] = np.full_like(local_ndarrays[idx], fill_value=poison_value, dtype=np.float32)
        poisoned_indices.append(idx)

    metadata = {
        "attack_type": "batchnorm_poisoning",
        "poison_value": float(poison_value),
        "poisoned_indices": poisoned_indices,
        "trainable_indices_preserved": TRAINABLE_INDICES,
        "int_counters_preserved": BN_INT_INDICES,
    }
    return transformed, metadata


class BatchNormPoisonedHospitalClient(HospitalClient):
    """
    Compromised Hospital Client that mounts isolated BatchNorm buffer poisoning.
    Leaves trainable parameters and integer counters completely honest/normal,
    but injects extreme outliers into running_mean and running_var.
    """

    def __init__(
        self,
        client_id: str,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        local_epochs: int = 1,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None,
        poison_value: float = 1_000_000.0,
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
        self.poison_value = float(poison_value)
        self.attack_start_round = int(attack_start_round)
        self.attack_history: List[Dict[str, Any]] = []

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Execute normal local training (updating trainable parameters and counters honestly),
        then inject extreme values into floating BatchNorm buffers if server_round >= attack_start_round.
        """
        local_ndarrays, num_examples, metrics = super().fit(parameters, config)
        current_round = int(config.get("server_round", config.get("round", 1)))

        if current_round >= self.attack_start_round:
            transformed_ndarrays, meta = apply_batchnorm_poisoning_transformation(
                local_ndarrays=local_ndarrays,
                poison_value=self.poison_value,
            )
            meta["round"] = current_round
            self.attack_history.append(meta)

            metrics["attack_active"] = True
            metrics["attack_type"] = "batchnorm_poisoning"
            metrics["poison_value"] = self.poison_value

            return transformed_ndarrays, num_examples, metrics
        else:
            metrics["attack_active"] = False
            return local_ndarrays, num_examples, metrics
