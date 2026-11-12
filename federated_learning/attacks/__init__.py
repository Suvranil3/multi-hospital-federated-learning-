"""
Phase 5: Adversarial Attack Modules for Federated Learning.
Provides implementations of:
- Targeted label flipping (Pneumonia 1 -> Normal 0)
- Model poisoning (Update scaling and inversion strictly on trainable parameters)
- Backdoor trigger injection (Post-normalization synthetic watermark and triggered evaluation)
"""

from federated_learning.attacks.label_flipping import LabelFlippedSubset
from federated_learning.attacks.model_poisoning import (
    PoisonedHospitalClient,
    apply_adversarial_update_transformation,
)
from federated_learning.attacks.backdoor import (
    BackdoorSubset,
    TriggeredPneumoniaTestDataset,
    TRIGGER_VALUE_NORMALIZED,
    TRIGGER_COORDS,
)
from federated_learning.attacks.batchnorm_poisoning import (
    BatchNormPoisonedHospitalClient,
    apply_batchnorm_poisoning_transformation,
)

__all__ = [
    "LabelFlippedSubset",
    "PoisonedHospitalClient",
    "apply_adversarial_update_transformation",
    "BackdoorSubset",
    "TriggeredPneumoniaTestDataset",
    "TRIGGER_VALUE_NORMALIZED",
    "TRIGGER_COORDS",
    "BatchNormPoisonedHospitalClient",
    "apply_batchnorm_poisoning_transformation",
]

