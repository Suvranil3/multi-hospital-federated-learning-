"""
Central Configuration and Reproducibility Module.
Defines experiment constants, dataset metadata, filesystem paths,
and deterministic seeding across the project.
"""

from pathlib import Path
import random
import os
import numpy as np
import torch

# Base Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_LOGS_DIR = RESULTS_DIR / "logs"
RESULTS_MODELS_DIR = RESULTS_DIR / "models"

# Ensure all result and data directories exist
for directory in [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    RESULTS_FIGURES_DIR,
    RESULTS_TABLES_DIR,
    RESULTS_LOGS_DIR,
    RESULTS_MODELS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# Random Seed for Strict Reproducibility
SEED = 42

# CIFAR-10 Dataset Specifications
CIFAR10_NUM_CLASSES = 10
CIFAR10_TRAIN_SAMPLES = 50_000
CIFAR10_TEST_SAMPLES = 10_000
CIFAR10_TRAIN_SAMPLES_PER_CLASS = 5_000
CIFAR10_TEST_SAMPLES_PER_CLASS = 1_000
CIFAR10_IMAGE_SHAPE = (3, 32, 32)  # (C, H, W)
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Standard CIFAR-10 Normalization Constants (for future preprocessing)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Federated Learning Settings (Planned for later phases)
NUM_CLIENTS = 10
DIRICHLET_ALPHAS = [1.0, 0.5, 0.1]
DEFAULT_BATCH_SIZE = 32
DEFAULT_LOCAL_EPOCHS = 2
DEFAULT_FL_ROUNDS = 20
DEFAULT_LEARNING_RATE = 0.01

# Centralized Training Baseline Settings (Phase 2)
CENTRALIZED_EPOCHS = 15
CENTRALIZED_BATCH_SIZE = 64
CENTRALIZED_LEARNING_RATE = 0.001
CENTRALIZED_VAL_RATIO = 0.2  # 40,000 train (80%), 10,000 validation (20%)
CENTRALIZED_WEIGHT_DECAY = 1e-4



def set_seed(seed: int = SEED) -> None:
    """
    Set deterministic random seeds across Python, NumPy, and PyTorch.
    Ensures identical initialization and data shuffling across runs.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
