# Experiment Log

This document records every empirical experiment conducted in the project.
No entries may contain simulated, guessed, or fabricated numbers. All metrics must strictly originate from recorded experiment execution logs.

---

## Log Format Schema
Each experiment must record:
- **Experiment ID**: Standardized identifier (e.g., `EXP-001-CENTRAL-CNN`, `EXP-002-FL-IID-FEDAVG`)
- **Date**: Execution timestamp
- **Dataset Configuration**: Dataset name, sample counts, splits
- **Number of Clients**: Total participating edge clients
- **IID / Non-IID Setting**: Data distribution type
- **Dirichlet Alpha**: Heterogeneity concentration parameter ($\alpha$) or N/A
- **Attack Type**: Injected attack or None
- **Malicious Clients**: Count and percentage of adversaries
- **Model Architecture**: Neural network definition
- **FL Algorithm**: Federated learning framework / protocol
- **Detector Version**: Detection algorithm version or None
- **Aggregation Method**: Model update aggregation rule
- **Random Seed**: Master seed used for run
- **Results**: Quantitative metrics (Accuracy, Loss, FPR, FNR, ASR, etc.)
- **Execution Status**: Succeeded or Failed
- **Notes**: Key observations and reproducibility pointers

---

## Experiment History

### [EXP-000-DATA] - CIFAR-10 Dataset Verification & Statistics
- **Date**: 2026-10-08
- **Dataset Configuration**: CIFAR-10 (raw download from torchvision official mirror to `data/raw/`)
- **Number of Clients**: N/A (Phase 1 Central Dataset Verification)
- **IID / Non-IID Setting**: N/A (Unpartitioned global dataset)
- **Dirichlet Alpha**: N/A (Partitioning scheduled for Phase 4)
- **Attack Type**: None
- **Malicious Clients**: 0 (0%)
- **Model Architecture**: None (Data verification only)
- **FL Algorithm**: None
- **Detector Version**: None
- **Aggregation Method**: None
- **Random Seed**: 42
- **Results**:
  - Training Samples: 50,000 (Exactly 5,000 per class across 10 classes)
  - Test Samples: 10,000 (Exactly 1,000 per class across 10 classes)
  - Total Samples: 60,000
  - Classes (10): `airplane` (0), `automobile` (1), `bird` (2), `cat` (3), `deer` (4), `dog` (5), `frog` (6), `horse` (7), `ship` (8), `truck` (9)
  - Raw Image Resolution: 32x32 pixels, 3 channels (RGB)
  - Tensor Shape: (3, 32, 32)
  - Class Distribution: Perfectly balanced (Train: 5,000/class, Test: 1,000/class)
  - Channel Means (R, G, B): [0.49140, 0.48216, 0.44653]
  - Channel Stds (R, G, B): [0.24703, 0.24349, 0.26159]
  - Visual Artifacts:
    - Sample Grid: `results/figures/cifar10_sample_grid.png`
    - Class Distribution: `results/figures/cifar10_class_distribution.png`
  - Report File: `results/logs/cifar10_verification_report.json`
- **Execution Status**: SUCCEEDED (Phase 1)
- **Notes**: Dataset verification pipeline script `data/verify_dataset.py` passed all assertions. Unit test suite `tests/test_data_loading.py` passed (3/3 tests). Environment verified with Python 3.14.4, PyTorch 2.14.1+cu130, Torchvision 0.29.1+cu130, NumPy 2.5.3, Matplotlib 3.11.2, Flower 1.39.0.

---

### [EXP-001-CENTRAL-CNN] - Centralized Small CNN Baseline Training on CIFAR-10
- **Date**: 2026-10-08
- **Dataset Configuration**: CIFAR-10
  - Training Split: 40,000 samples (4,000 per class, 80% stratified split)
  - Validation Split: 10,000 samples (1,000 per class, 20% stratified split)
  - Test Split: 10,000 samples (official untouched test set)
- **Number of Clients**: N/A (Centralized ML baseline)
- **IID / Non-IID Setting**: N/A (Centralized single-node training)
- **Dirichlet Alpha**: N/A (Partitioning scheduled for Phase 4)
- **Attack Type**: None
- **Malicious Clients**: 0 (0%)
- **Model Architecture**: SmallCNN (3 Conv-BN-ReLU-MaxPool blocks + Dropout + 2-layer FC head; 189,130 parameters)
- **FL Algorithm**: None (Centralized PyTorch benchmark)
- **Detector Version**: None
- **Aggregation Method**: None
- **Random Seed**: 42
- **Training Hyperparameters**:
  - Epochs: 15
  - Batch Size: 64
  - Optimizer: Adam
  - Learning Rate: 0.001
  - Weight Decay: 0.0001
  - Loss Criterion: CrossEntropyLoss
  - Augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip()
  - Normalization: Mean=(0.4914, 0.4822, 0.4465), Std=(0.2470, 0.2435, 0.2616)
- **Compute Device**: CUDA (NVIDIA GeForce RTX 3050 Laptop GPU)
- **Results**:
  - Total Training Time: 100.97 seconds (~6.7 seconds/epoch)
  - Final Training Loss: 0.7310
  - Final Training Accuracy: 74.50%
  - Best Validation Loss: 0.6400 (Epoch 15)
  - Best Validation Accuracy: 77.34% (Epoch 15)
  - Untouched Test Set Loss: 0.6578
  - Untouched Test Set Accuracy: 76.94% (7,694 / 10,000 correct)
  - Visual Artifacts:
    - Loss Curve: `results/figures/centralized_training_loss.png`
    - Accuracy Curve: `results/figures/centralized_training_accuracy.png`
    - Confusion Matrix: `results/figures/confusion_matrix.png`
  - Model Checkpoint: `results/models/best_centralized_cnn.pt`
  - Report File: `results/logs/centralized_cnn_results.json`
- **Execution Status**: SUCCEEDED (Phase 2)
- **Notes**: Established solid centralized baseline (~76.94% test accuracy) for subsequent FL benchmarking. Untouched test set evaluated strictly once after training completion. Unit tests in `tests/test_model.py` and `tests/test_training.py` all passed.


