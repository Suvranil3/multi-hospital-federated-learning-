"""
Targeted Label-Flipping Attack Module (Phase 5).
Simulates a compromised hospital participant that systematically flips
Pneumonia labels (class 1) to Normal (class 0) during local training.

Preserves the underlying dataset and clean validation/test datasets without
in-place mutation.
"""

from typing import Dict, Any, List, Set, Union
import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class LabelFlippedSubset(Dataset):
    """
    Wraps a client dataset (typically a PyTorch Subset) and flips targeted labels.
    Guarantees strict non-mutation of the underlying dataset.
    """

    def __init__(
        self,
        base_dataset: Union[Dataset, Subset],
        source_class: int = 1,
        target_class: int = 0,
        flip_fraction: float = 1.0,
        seed: int = 42,
    ) -> None:
        """
        Args:
            base_dataset: The client's original local dataset partition.
            source_class: Ground truth class to attack (default 1: Pneumonia).
            target_class: Malicious target class (default 0: Normal).
            flip_fraction: Fraction of source_class samples to flip (0.0 to 1.0, default 1.0).
            seed: Deterministic random seed for selecting subset if flip_fraction < 1.0.
        """
        self.base_dataset = base_dataset
        self.source_class = int(source_class)
        self.target_class = int(target_class)
        self.flip_fraction = float(flip_fraction)
        self.seed = int(seed)

        assert 0.0 <= self.flip_fraction <= 1.0, f"flip_fraction must be in [0, 1], got {self.flip_fraction}"

        # Inspect local partition without mutating underlying dataset
        self.total_samples = len(base_dataset)
        self.source_indices: List[int] = []
        self.normal_count = 0
        self.pneumonia_count = 0

        # Retrieve labels directly if Subset over PneumoniaMNIST, or fallback to iteration
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

        # Determine which source_class samples to flip
        if self.flip_fraction >= 1.0:
            self.flipped_indices: Set[int] = set(self.source_indices)
        else:
            num_to_flip = int(round(self.flip_fraction * len(self.source_indices)))
            rng = np.random.RandomState(self.seed)
            shuffled = self.source_indices.copy()
            rng.shuffle(shuffled)
            self.flipped_indices = set(shuffled[:num_to_flip])

        self.poisoned_sample_count = len(self.flipped_indices)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> Any:
        img, target = self.base_dataset[index]

        if index in self.flipped_indices:
            # Create a new label object preserving the original container type
            if isinstance(target, np.ndarray):
                adv_target = np.array([self.target_class], dtype=target.dtype)
            elif isinstance(target, torch.Tensor):
                adv_target = torch.tensor([self.target_class], dtype=target.dtype, device=target.device)
            else:
                adv_target = self.target_class
            return img, adv_target

        return img, target

    def get_attack_metadata(self) -> Dict[str, Any]:
        """Return structured summary of the applied label-flipping attack."""
        return {
            "attack_type": "targeted_label_flipping",
            "source_class": self.source_class,
            "target_class": self.target_class,
            "flip_fraction": self.flip_fraction,
            "seed": self.seed,
            "total_local_samples": self.total_samples,
            "original_normal_count": self.normal_count,
            "original_pneumonia_count": self.pneumonia_count,
            "poisoned_sample_count": self.poisoned_sample_count,
        }
