"""
Backdoor Trigger Injection Attack Module (Phase 5).
Simulates a compromised hospital participant embedding a synthetic watermark
into a fraction rho of its local pneumonia training images.

Trigger Specifications:
- Geometry: 3x3 pixel square at bottom-right corner [24:27, 24:27].
- Intensity: Normalized white pixel (+2.5422), calculated via (1.0 - MEAN) / STD.
- Execution Order: Applied consistently AFTER image augmentation and normalization.
- Target Class: 0 (Normal).
- Evaluation: Backdoor Attack Success Rate (ASR) evaluated post-training on a
  synthetic triggered test set constructed exclusively from all 390 test pneumonia scans.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Set, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import PNEUMONIA_MEAN, PNEUMONIA_STD

# Exact normalized intensity of pixel value 1.0 (white) under PneumoniaMNIST normalization
TRIGGER_VALUE_NORMALIZED: float = float((1.0 - PNEUMONIA_MEAN[0]) / PNEUMONIA_STD[0])
TRIGGER_COORDS: Tuple[slice, slice] = (slice(24, 27), slice(24, 27))


def inject_trigger_to_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Inject 3x3 normalized white trigger patch into the bottom-right corner [24:27, 24:27].
    Operates on a clone to ensure the underlying dataset is never mutated.

    Args:
        image_tensor: Normalized image tensor of shape [1, 28, 28] or [C, H, W].

    Returns:
        New tensor with trigger injected.
    """
    poisoned = image_tensor.clone()
    poisoned[:, TRIGGER_COORDS[0], TRIGGER_COORDS[1]] = TRIGGER_VALUE_NORMALIZED
    return poisoned


class BackdoorSubset(Dataset):
    """
    Wraps a client dataset partition (typically a PyTorch Subset) and injects
    the backdoor trigger into a fraction rho of pneumonia training samples.
    Relabels triggered samples to class 0 (Normal).
    """

    def __init__(
        self,
        base_dataset: Union[Dataset, Subset],
        poison_fraction: float = 0.50,
        target_class: int = 0,
        source_class: int = 1,
        seed: int = 42,
    ) -> None:
        """
        Args:
            base_dataset: The client's original local dataset partition.
            poison_fraction: Fraction of source_class (pneumonia) samples to poison (default 0.50).
            target_class: Malicious target label (default 0: Normal).
            source_class: Class to stamp and relabel (default 1: Pneumonia).
            seed: Deterministic random seed for subset sampling.
        """
        self.base_dataset = base_dataset
        self.poison_fraction = float(poison_fraction)
        self.target_class = int(target_class)
        self.source_class = int(source_class)
        self.seed = int(seed)

        assert 0.0 <= self.poison_fraction <= 1.0, f"poison_fraction must be in [0, 1], got {self.poison_fraction}"

        self.total_samples = len(base_dataset)
        self.source_indices: List[int] = []
        self.normal_count = 0
        self.pneumonia_count = 0

        # Inspect partition labels without mutating underlying dataset
        if isinstance(base_dataset, Subset) and hasattr(base_dataset.dataset, "labels"):
            parent_labels = base_dataset.dataset.labels
            for local_idx, parent_idx in enumerate(base_dataset.indices):
                lbl = int(np.squeeze(parent_labels[parent_idx]))
                if lbl == 0:
                    self.normal_count += 1
                elif lbl == 1:
                    self.pneumonia_count += 1
                if lbl == self.source_class:
                    self.source_indices.append(local_idx)
        else:
            for local_idx in range(self.total_samples):
                _, lbl_raw = base_dataset[local_idx]
                lbl = int(np.squeeze(lbl_raw))
                if lbl == 0:
                    self.normal_count += 1
                elif lbl == 1:
                    self.pneumonia_count += 1
                if lbl == self.source_class:
                    self.source_indices.append(local_idx)

        # Select samples to poison
        if self.poison_fraction >= 1.0:
            self.poisoned_indices: Set[int] = set(self.source_indices)
        else:
            num_to_poison = int(round(self.poison_fraction * len(self.source_indices)))
            rng = np.random.RandomState(self.seed)
            shuffled = self.source_indices.copy()
            rng.shuffle(shuffled)
            self.poisoned_indices = set(shuffled[:num_to_poison])

        self.poisoned_sample_count = len(self.poisoned_indices)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        img, target = self.base_dataset[index]

        if index in self.poisoned_indices:
            # Apply trigger AFTER normalization and augmentation
            adv_img = inject_trigger_to_tensor(img)

            if isinstance(target, np.ndarray):
                adv_target = np.array([self.target_class], dtype=target.dtype)
            elif isinstance(target, torch.Tensor):
                adv_target = torch.tensor([self.target_class], dtype=target.dtype, device=target.device)
            else:
                adv_target = self.target_class
            return adv_img, adv_target

        return img, target

    def get_attack_metadata(self) -> Dict[str, Any]:
        """Return structured summary of backdoor training configuration."""
        return {
            "attack_type": "backdoor_trigger_injection",
            "trigger_geometry": "3x3_bottom_right_[24:27, 24:27]",
            "trigger_intensity_normalized": round(TRIGGER_VALUE_NORMALIZED, 4),
            "poison_fraction": self.poison_fraction,
            "target_class": self.target_class,
            "source_class": self.source_class,
            "seed": self.seed,
            "total_local_samples": self.total_samples,
            "original_normal_count": self.normal_count,
            "original_pneumonia_count": self.pneumonia_count,
            "poisoned_sample_count": self.poisoned_sample_count,
        }


class TriggeredPneumoniaTestDataset(Dataset):
    """
    Synthetic backdoor test evaluation dataset.
    Extracts all Pneumonia samples (class 1) from the official test dataset
    and applies the identical post-normalization trigger.
    Ground-truth target for ASR evaluation is class 0 (Normal).
    """

    def __init__(self, clean_test_dataset: Dataset) -> None:
        self.clean_test_dataset = clean_test_dataset
        self.pneumonia_indices: List[int] = []

        # Find all Pneumonia samples (class 1) in official test set
        if hasattr(clean_test_dataset, "labels"):
            labels = np.squeeze(clean_test_dataset.labels)
            self.pneumonia_indices = list(np.where(labels == 1)[0])
        else:
            for idx in range(len(clean_test_dataset)):
                _, lbl_raw = clean_test_dataset[idx]
                if int(np.squeeze(lbl_raw)) == 1:
                    self.pneumonia_indices.append(idx)

        assert len(self.pneumonia_indices) == 390, (
            f"Expected 390 pneumonia test samples, found {len(self.pneumonia_indices)}"
        )

    def __len__(self) -> int:
        return len(self.pneumonia_indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, np.ndarray]:
        parent_idx = self.pneumonia_indices[index]
        clean_img, _ = self.clean_test_dataset[parent_idx]

        # Apply identical trigger
        triggered_img = inject_trigger_to_tensor(clean_img)
        # Expected backdoor target class is 0 (Normal)
        target = np.array([0], dtype=np.int64)

        return triggered_img, target


def evaluate_backdoor_asr(
    model: nn.Module,
    triggered_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluate the Backdoor Attack Success Rate (ASR) strictly post-training.
    Measures the percentage of triggered pneumonia test images predicted as Normal (0).
    """
    model.eval()
    total_triggered = 0
    successful_attacks = 0

    with torch.no_grad():
        for images, targets in triggered_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Target class is 0 (Normal)
            successful_attacks += int((preds == 0).sum().item())
            total_triggered += images.size(0)

    asr = (successful_attacks / max(total_triggered, 1)) * 100.0

    return {
        "total_triggered_test_samples": total_triggered,
        "successful_backdoor_attacks": successful_attacks,
        "attack_success_rate_percent": round(asr, 2),
        "target_class": 0,
        "source_class": 1,
    }
