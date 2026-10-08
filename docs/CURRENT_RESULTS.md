# Current Empirical Results

This document records the exact, measured empirical metrics currently established in the project.
All values are derived directly from reproducible execution logs and automated test runs (master seed 42). No values are simulated or fabricated.

---

## 1. Phase MEDICAL-01: PneumoniaMNIST Dataset Verification (`EXP-MED-000-DATA`)

- **Execution Script**: [`data/verify_dataset.py`](file:///home/suvranil/project/data/verify_dataset.py)
- **Log Report**: [`results/logs/pneumonia_mnist_verification_report.json`](file:///home/suvranil/project/results/logs/pneumonia_mnist_verification_report.json)
- **Status**: **PASSED (All Assertions Verified)**
- **Deterministic Seed**: `42`

### 1.1. Sample Counts and Official Benchmark Splits
| Split | Total Samples | Normal (Class 0) | Pneumonia (Class 1) | % Pneumonia |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 4,708 | 1,214 | 3,494 | 74.21% |
| **Validation** | 524 | 135 | 389 | 74.24% |
| **Test (Untouched)** | 624 | 234 | 390 | 62.50% |
| **Total Collection** | **5,856** | **1,583** | **4,273** | **72.95%** |

### 1.2. Pixel Intensity Statistics
- **Image Resolution**: $28 \times 28$ pixels, single grayscale channel (`shape: (1, 28, 28)`)
- **Raw Intensity Range**: $[0, 255]$ (Min: `0`, Max: `255`)
- **Empirical Normalized Mean $[0, 1]$**: `0.57192`
- **Empirical Normalized Std $[0, 1]$**: `0.16835`

### 1.3. Verification Artifacts
- **Representative Sample Grid**: [`results/figures/pneumonia_sample_grid.png`](file:///home/suvranil/project/results/figures/pneumonia_sample_grid.png)
- **Class Distribution Bar Chart**: [`results/figures/pneumonia_class_distribution.png`](file:///home/suvranil/project/results/figures/pneumonia_class_distribution.png)

---

## 2. Phase MEDICAL-02: Centralized Medical Small CNN Baseline (`EXP-MED-001-CENTRAL-CNN`)

- **Execution Script**: [`experiments/train_medical.py`](file:///home/suvranil/project/experiments/train_medical.py)
- **Log Report**: [`results/logs/medical_centralized_results.json`](file:///home/suvranil/project/results/logs/medical_centralized_results.json)
- **Model Checkpoint**: [`results/models/best_centralized_medical_cnn.pt`](file:///home/suvranil/project/results/models/best_centralized_medical_cnn.pt)
- **Compute Device**: **NVIDIA GeForce RTX 3050 Laptop GPU** (CUDA, PyTorch 2.14.1+cu130)
- **Training Time**: **13.35 seconds** (15 epochs, ~0.89s per epoch)
- **Hyperparameters**: Batch Size: 64, Optimizer: Adam (lr = 0.001, weight decay = 0.0001), Loss: CrossEntropyLoss

### 2.1. Epoch-by-Epoch Training Trajectory
| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Epoch Time | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.3236 | 87.60% | 0.1422 | 95.23% | 4.3s | Best |
| 2 | 0.1389 | 94.41% | 0.1136 | 95.04% | 0.6s | — |
| 3 | 0.1181 | 95.43% | 0.1083 | 95.80% | 0.6s | Best |
| 4 | 0.1149 | 95.60% | 0.1037 | 95.80% | 0.6s | — |
| 5 | 0.1079 | 95.71% | 0.1092 | 95.42% | 0.6s | — |
| 6 | 0.0996 | 95.92% | 0.0812 | 96.56% | 0.6s | Best |
| 7 | 0.0959 | 96.18% | 0.1019 | 95.80% | 0.6s | — |
| 8 | 0.0907 | 96.52% | 0.1052 | 95.80% | 0.6s | — |
| 9 | 0.0925 | 96.24% | 0.0731 | 97.52% | 0.6s | Best |
| 10 | 0.0947 | 96.33% | 0.0929 | 96.56% | 0.6s | — |
| 11 | 0.0958 | 96.54% | 0.0815 | 96.37% | 0.6s | — |
| 12 | 0.0879 | 96.52% | 0.0647 | 97.52% | 0.7s | — |
| 13 | 0.0830 | 96.79% | 0.0693 | 97.71% | 0.6s | Best |
| **14** | **0.0777** | **97.13%** | **0.0760** | **97.90%** | **0.6s** | **BEST CHECKPOINT** |
| 15 | 0.0810 | 96.77% | 0.0804 | 96.76% | 0.6s | — |

### 2.2. Evaluation on Untouched Test Split (624 Images)
- **Test Loss**: **0.9172**
- **Test Accuracy**: **81.25%** (507 / 624 correct)

#### Diagnostic Performance Scorecard
| Metric | Class 1 (Pneumonia) | Class 0 (Normal) | Macro Average |
| :--- | :---: | :---: | :---: |
| **Sensitivity / Recall** | **99.74%** (389 / 390) | **50.43%** (118 / 234) | **75.09%** |
| **Precision** | **77.03%** (389 / 505) | **99.16%** (118 / 119) | **88.09%** |
| **F1-Score** | **0.8693** | **0.6686** | **0.7689** |

#### Test Confusion Matrix
| True \ Predicted | Predicted Normal (0) | Predicted Pneumonia (1) | Total |
| :--- | :---: | :---: | :---: |
| **True Normal (0)** | **118** (True Negative) | **116** (False Positive) | 234 |
| **True Pneumonia (1)** | **1** (False Negative) | **389** (True Positive) | 390 |
| **Total Predicted** | 119 | 505 | 624 |

### 2.3. Clinical Interpretation of Results
- **Near-Zero False Negative Rate for Pneumonia**: The model missed only **1 out of 390 pneumonia cases** (99.74% sensitivity). In clinical screening, missing an active pneumonia infection can be life-threatening; high sensitivity is the most critical safety property.
- **Precision Trade-off**: The model produces 116 false positives on normal scans (77.03% precision), a standard trade-off in medical image triage without cost-weighted loss tuning.
- **Medical Disclaimer**: *These are research benchmark scores evaluated on $28 \times 28$ downsampled images for federated learning simulation; they do not represent clinical diagnostic deployment.*

### 2.4. Generated Baseline Figures
- **Training Loss Curve**: [`results/figures/medical_training_loss.png`](file:///home/suvranil/project/results/figures/medical_training_loss.png)
- **Training Accuracy Curve**: [`results/figures/medical_training_accuracy.png`](file:///home/suvranil/project/results/figures/medical_training_accuracy.png)
- **Medical Confusion Matrix**: [`results/figures/medical_confusion_matrix.png`](file:///home/suvranil/project/results/figures/medical_confusion_matrix.png)

---

## 3. Automated Quality & Regression Test Suite

- **Test Framework**: `pytest 9.1.1`
- **Total Tests**: **14 tests**
- **Test Results**: **14 passed (100%) in 3.20 seconds**
  - `tests/test_data_loading.py`: Splits exist, batch shapes (1x28x28), deterministic shuffling.
  - `tests/test_model.py`: Model architecture, parameters (420,802), output shapes `[1, 4, 16, 32, 2]`, forward validity, loss computation, gradient backpropagation step.
  - `tests/test_training.py`: Split separation, loader batch shapes, medical metrics functions.

---

## 4. Summary of Archived CIFAR-10 Results
The previous generic proof-of-concept remains archived in [`results/archive/cifar10/`](file:///home/suvranil/project/results/archive/cifar10/README.md):
- **Dataset**: CIFAR-10 (60,000 images, 10 classes, 32x32 RGB).
- **Model**: SmallCNN (189,130 parameters).
- **Test Accuracy**: 76.94% (7,694 / 10,000 correct).
- **Training Time**: 100.97 seconds on RTX 3050 GPU.
- **Status**: Archived; superseded by PneumoniaMNIST for the hospital project.
