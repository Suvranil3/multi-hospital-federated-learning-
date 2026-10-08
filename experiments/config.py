"""
Central Configuration and Reproducibility Module.
Primary Medical Scenario: Multi-Hospital Chest X-Ray Classification (PneumoniaMNIST).
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
RESULTS_ARCHIVE_DIR = RESULTS_DIR / "archive" / "cifar10"

# Ensure all result and data directories exist
for directory in [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    RESULTS_FIGURES_DIR,
    RESULTS_TABLES_DIR,
    RESULTS_LOGS_DIR,
    RESULTS_MODELS_DIR,
    RESULTS_ARCHIVE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# Random Seed for Strict Reproducibility
SEED = 42

# =========================================================================
# PRIMARY MEDICAL DATASET SPECIFICATIONS: PneumoniaMNIST (MedMNIST v3)
# =========================================================================
DATASET_NAME = "PneumoniaMNIST"
NUM_CLASSES = 2
CLASSES = ("normal", "pneumonia")
CLASS_LABELS = {0: "normal", 1: "pneumonia"}
IMAGE_CHANNELS = 1
IMAGE_SIZE = (28, 28)
IMAGE_SHAPE = (1, 28, 28)  # (C, H, W)

# Official Dataset Sample Counts
TRAIN_SAMPLES = 4708
VAL_SAMPLES = 524
TEST_SAMPLES = 624
TOTAL_SAMPLES = 5856

# Official Class Distributions (Measured)
# Train: Normal (0) = 1214, Pneumonia (1) = 3494
# Val:   Normal (0) = 135,  Pneumonia (1) = 389
# Test:  Normal (0) = 234,  Pneumonia (1) = 390
TRAIN_CLASS_COUNTS = {0: 1214, 1: 3494}
VAL_CLASS_COUNTS = {0: 135, 1: 389}
TEST_CLASS_COUNTS = {0: 234, 1: 390}

# Grayscale Normalization Constants (Empirically computed on PneumoniaMNIST train set)
PNEUMONIA_MEAN = (0.5719,)
PNEUMONIA_STD = (0.1684,)

# Medical Task Metadata
TASK_TYPE = "binary-class"
URL_NPZ = "https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1"

# =========================================================================
# CENTRALIZED MEDICAL TRAINING BASELINE SETTINGS
# =========================================================================
CENTRALIZED_EPOCHS = 15
CENTRALIZED_BATCH_SIZE = 64
CENTRALIZED_LEARNING_RATE = 0.001
CENTRALIZED_WEIGHT_DECAY = 1e-4

# =========================================================================
# FEDERATED LEARNING SETTINGS (Planned for future hospital simulation)
# =========================================================================
NUM_HOSPITALS = 10
NUM_CLIENTS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_LOCAL_EPOCHS = 2
DEFAULT_FL_ROUNDS = 20
DEFAULT_LEARNING_RATE = 0.01

# =========================================================================
# ARCHIVED CIFAR-10 SPECIFICATIONS (Preserved Proof-of-Concept)
# =========================================================================
ARCHIVED_CIFAR10_NUM_CLASSES = 10
ARCHIVED_CIFAR10_TRAIN_SAMPLES = 50_000
ARCHIVED_CIFAR10_TEST_SAMPLES = 10_000
ARCHIVED_CIFAR10_IMAGE_SHAPE = (3, 32, 32)
ARCHIVED_CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
ARCHIVED_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
ARCHIVED_CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Legacy aliases for backward compatibility with archived code/tests
CIFAR10_NUM_CLASSES = ARCHIVED_CIFAR10_NUM_CLASSES
CIFAR10_TRAIN_SAMPLES = ARCHIVED_CIFAR10_TRAIN_SAMPLES
CIFAR10_TEST_SAMPLES = ARCHIVED_CIFAR10_TEST_SAMPLES
CIFAR10_IMAGE_SHAPE = ARCHIVED_CIFAR10_IMAGE_SHAPE
CIFAR10_CLASSES = ARCHIVED_CIFAR10_CLASSES
CIFAR10_MEAN = ARCHIVED_CIFAR10_MEAN
CIFAR10_STD = ARCHIVED_CIFAR10_STD


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
