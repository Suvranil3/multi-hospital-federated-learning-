# Experiment Log

This document records every empirical experiment conducted in the project.
No entries may contain simulated, guessed, or fabricated numbers. All metrics must strictly originate from recorded experiment execution logs.

---

## Log Format Schema
Each experiment must record:
- **Experiment ID**: Standardized identifier (e.g., `EXP-MED-000-DATA`, `EXP-MED-001-CENTRAL-CNN`)
- **Date**: Execution timestamp
- **Dataset Configuration**: Dataset name, sample counts, splits
- **Number of Clients**: Total participating edge/hospital clients
- **IID / Non-IID Setting**: Data distribution type
- **Dirichlet Alpha**: Heterogeneity concentration parameter ($\alpha$) or N/A
- **Attack Type**: Injected attack or None
- **Malicious Clients**: Count and percentage of adversaries
- **Model Architecture**: Neural network definition
- **FL Algorithm**: Federated learning framework / protocol
- **Detector Version**: Detection algorithm version or None
- **Aggregation Method**: Model update aggregation rule
- **Random Seed**: Master seed used for run
- **Results**: Quantitative metrics (Accuracy, Loss, Precision, Recall, F1, etc.)
- **Execution Status**: Succeeded or Failed
- **Notes**: Key observations and reproducibility pointers

---

## Active Medical Experiment History

### [EXP-MED-000-DATA] - PneumoniaMNIST Dataset Verification & Statistical Analysis
- **Date**: 2026-10-08
- **Dataset Configuration**: PneumoniaMNIST (MedMNIST v3, downloaded from Zenodo mirror to `data/raw/`)
- **Number of Clients / Hospitals**: N/A (Phase MEDICAL-01 Central Dataset Verification)
- **IID / Non-IID Setting**: N/A (Official unpartitioned dataset)
- **Dirichlet Alpha**: N/A (Multi-hospital partitioning scheduled for Phase 4)
- **Attack Type**: None
- **Malicious Clients**: 0 (0%)
- **Model Architecture**: None (Data verification only)
- **FL Algorithm**: None
- **Detector Version**: None
- **Aggregation Method**: None
- **Random Seed**: 42
- **Results**:
  - Training Samples: 4,708 (Normal: 1,214, Pneumonia: 3,494)
  - Validation Samples: 524 (Normal: 135, Pneumonia: 389)
  - Test Samples: 624 (Normal: 234, Pneumonia: 390)
  - Total Samples: 5,856
  - Classes (2): `0: normal`, `1: pneumonia`
  - Raw Image Resolution: 28x28 pixels, 1 channel (Grayscale)
  - Tensor Shape: (1, 28, 28)
  - Pixel Value Range: Min = 0, Max = 255
  - Grayscale Empirical Mean [0, 1]: 0.57192
  - Grayscale Empirical Std [0, 1]: 0.16835
  - Visual Artifacts:
    - Sample Grid: `results/figures/pneumonia_sample_grid.png`
    - Class Distribution: `results/figures/pneumonia_class_distribution.png`
  - Report File: `results/logs/pneumonia_mnist_verification_report.json`
- **Execution Status**: SUCCEEDED (Phase MEDICAL-01)
- **Notes**: All split counts and dimensions match official MedMNIST specification. Verified class imbalance: Pneumonia represents ~74.2% of train/val and ~62.5% of test, highlighting necessity of reporting sensitivity, specificity, and F1.

---

### [EXP-MED-001-CENTRAL-CNN] - Centralized Medical Small CNN Baseline Training on PneumoniaMNIST
- **Date**: 2026-10-08
- **Dataset Configuration**: PneumoniaMNIST
  - Training Split: 4,708 samples (official train split)
  - Validation Split: 524 samples (official val split, used for checkpointing)
  - Test Split: 624 samples (official untouched test split)
- **Number of Clients / Hospitals**: N/A (Centralized single-node medical baseline)
- **IID / Non-IID Setting**: N/A
- **Dirichlet Alpha**: N/A
- **Attack Type**: None
- **Malicious Clients**: 0 (0%)
- **Model Architecture**: `SmallCNN` (2 Conv-BN-ReLU-MaxPool blocks, Flatten, Dense 128, Dropout 0.2, 2-class output; **420,802 parameters**)
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
  - Augmentation: RandomHorizontalFlip, RandomRotation(10)
  - Normalization: Mean=(0.5719,), Std=(0.1684,)
- **Compute Device**: CUDA (NVIDIA GeForce RTX 3050 Laptop GPU)
- **Results**:
  - Total Training Time: 13.35 seconds (~0.89 seconds/epoch)
  - Final Training Loss: 0.0810 | Final Training Accuracy: 96.77%
  - Best Validation Loss: 0.0760 | Best Validation Accuracy: 97.90% (Epoch 14)
  - Untouched Test Set Loss: 0.9172
  - Untouched Test Set Accuracy: **81.25%** (507 / 624 correct)
  - Pneumonia Sensitivity / Recall: **99.74%** (389 / 390 pneumonia cases correctly detected)
  - Pneumonia Precision: **77.03%**
  - Pneumonia F1-Score: **0.8693**
  - Normal Specificity / Recall: **50.43%** (118 / 234 normal cases correctly identified)
  - Normal Precision: **99.16%** (118 / 119)
  - Normal F1-Score: **0.6686**
  - Macro F1-Score: **0.7689**
  - Confusion Matrix: TN=118, FP=116, FN=1, TP=389
  - Visual Artifacts:
    - Loss Curve: `results/figures/medical_training_loss.png`
    - Accuracy Curve: `results/figures/medical_training_accuracy.png`
    - Confusion Matrix: `results/figures/medical_confusion_matrix.png`
  - Model Checkpoint: `results/models/best_centralized_medical_cnn.pt`
  - Report File: `results/logs/medical_centralized_results.json`
- **Execution Status**: SUCCEEDED (Phase MEDICAL-02)
- **Notes**: Model demonstrates near-perfect pneumonia sensitivity (99.74%, missing only 1 single pneumonia case in test set), crucial for medical triage. High pneumonia recall results in some false positives (116 normal predicted as pneumonia), typical for models optimizing cross-entropy under imbalanced clinical data without class weighting. Established clear benchmark before introducing federated learning.

---

## Archived Proof-of-Concept History (CIFAR-10)

### [EXP-000-DATA] - CIFAR-10 Dataset Verification & Statistics *(Archived)*
- **Date**: 2026-10-08
- **Status**: ARCHIVED PROOF-OF-CONCEPT (Preserved in `results/archive/cifar10/`)
- **Key Results**: 60,000 images, 10 balanced classes (5,000 train/class, 1,000 test/class), 32x32 RGB. Report: `results/archive/cifar10/logs/cifar10_verification_report.json`.

### [EXP-001-CENTRAL-CNN] - Centralized Small CNN Baseline on CIFAR-10 *(Archived)*
- **Date**: 2026-10-08
- **Status**: ARCHIVED PROOF-OF-CONCEPT (Preserved in `results/archive/cifar10/`)
- **Key Results**: SmallCNN (189,130 params), 15 epochs, RTX 3050 GPU, 100.97s, 76.94% Test Accuracy (7,694/10,000). Report: `results/archive/cifar10/logs/cifar10_centralized_cnn_results.json`.
