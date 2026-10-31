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

## 3. Phase 3: 10-Hospital IID Federated Learning Baseline (`EXP-MED-002-FED-IID`)

- **Execution Script**: [`experiments/train_federated.py`](file:///home/suvranil/project/experiments/train_federated.py)
- **Log Report**: [`results/logs/federated_iid_results.json`](file:///home/suvranil/project/results/logs/federated_iid_results.json)
- **Model Checkpoints**:
  - **Best (validation-selected)**: [`results/models/best_federated_iid_model.pt`](file:///home/suvranil/project/results/models/best_federated_iid_model.pt) — Round 17 (first round to reach 96.18% val accuracy)
  - **Final (end-of-training)**: [`results/models/final_federated_iid_model.pt`](file:///home/suvranil/project/results/models/final_federated_iid_model.pt) — Round 20 weights (val_acc = 95.80%, test_acc = 86.38%)
- **Framework & Protocol**: Flower (`flwr` 1.39.0) Virtual Client Engine, Federated Averaging (`FedAvg`)
- **Compute Architecture**: CPU (AMD Ryzen 5 5600H, 3 concurrent Ray actor pool, 4 CPUs/client)
- **Training Time**: **108.29 seconds** across 20 communication rounds (~5.41s per round)
- **Configuration**: 10 simulated hospitals, 2 local epochs/round, batch size 32, Adam (lr = 0.001)

### 3.1. Round-by-Round Trajectory (Summary Sample)
| Round | Clients Train Loss | Clients Train Acc (%) | Global Val Loss | Global Val Acc (%) | Round Notes |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | N/A | N/A | 0.7122 | 25.95% | Initial parameters |
| 1 | 0.4744 | 81.07% | 0.4093 | 89.89% | First aggregation |
| 5 | 0.1589 | 93.77% | 0.1243 | 94.66% | Rapid stabilization |
| 10 | 0.1318 | 95.06% | 0.1024 | 95.23% | Steady convergence |
| **17** | **0.1100** | **96.06%** | **0.0833** | **96.18%** | **BEST CHECKPOINT** (first to reach 96.18%) |
| 19 | 0.1105 | 95.85% | 0.0801 | 96.18% | Same val accuracy, lower val loss; checkpoint not overwritten |
| 20 | 0.1108 | 95.62% | 0.0870 | 95.80% | Final round (end-of-training) |

### 3.2. Evaluation on Untouched Test Split (624 Images) — Round 17 Best Checkpoint
- **Test Loss**: **0.5306**
- **Test Accuracy**: **85.74%** (535 / 624 correct)

> **Note**: All test metrics in this section are evaluated from the **Round 17 best checkpoint** (`best_federated_iid_model.pt`). The Round 20 final checkpoint (`final_federated_iid_model.pt`) independently achieves **86.38%** test accuracy (539/624, TN=159, FP=75, FN=10, TP=380).

#### Diagnostic Performance Scorecard (10-Hospital FedAvg vs Centralized Baseline)
| Metric | Centralized Small CNN (Phase 2) | 10-Hospital FedAvg IID (Phase 3) | Delta (FL vs Centralized) |
| :--- | :---: | :---: | :---: |
| **Overall Test Accuracy** | 81.25% (507/624) | **85.74%** (535/624) | **+4.49%** |
| **Pneumonia Sensitivity / Recall** | **99.74%** (389/390) | **97.95%** (382/390) | -1.79% |
| **Pneumonia Precision** | 77.03% (389/505) | **82.51%** (382/463) | **+5.48%** |
| **Pneumonia F1-Score** | 0.8693 | **0.8957** | **+0.0264** |
| **Normal Specificity / Recall** | 50.43% (118/234) | **65.38%** (153/234) | **+14.95%** |
| **Normal Precision** | 99.16% (118/119) | 95.03% (153/161) | -4.13% |
| **Macro F1-Score** | 0.7689 | **0.8352** | **+0.0663** |

#### Test Confusion Matrix (Phase 3 FedAvg)
| True \ Predicted | Predicted Normal (0) | Predicted Pneumonia (1) | Total |
| :--- | :---: | :---: | :---: |
| **True Normal (0)** | **153** (True Negative) | **81** (False Positive) | 234 |
| **True Pneumonia (1)** | **8** (False Negative) | **382** (True Positive) | 390 |
| **Total Predicted** | 161 | 463 | 624 |

### 3.3. Key Scientific Takeaways
1. **Sample Exposure Confound**: The federated experiment used **2.67× more total sample exposures** than the centralized baseline (20 rounds × 2 local epochs × 4,708 = 188,320 vs 15 epochs × 4,708 = 70,620). Therefore, the observed performance improvement (+4.49% test accuracy, +14.95% normal specificity) **cannot be attributed solely to FedAvg's implicit regularization effect**. The additional training compute is a confounding variable that must be controlled in future ablations.
2. **Competitive Distributed Performance**: Despite the confound, the IID federated setup demonstrates that FedAvg training across 10 hospital clients produces a global model competitive with (and in this case exceeding) centralized training, reducing false positives on normal scans from 116 to 81 while maintaining clinically vital pneumonia sensitivity at 97.95%.
3. **Definitive IID Control Baseline**: Establishes a clean benchmark for the homogeneous case. Any degradation observed in Phase 4 (non-IID Dirichlet skew) can be attributed to data heterogeneity, not the distributed protocol itself.

### 3.4. Generated Phase 3 Figures
- **FL Training Loss Curve**: [`results/figures/federated_iid_training_loss.png`](file:///home/suvranil/project/results/figures/federated_iid_training_loss.png)
- **FL Training Accuracy Curve**: [`results/figures/federated_iid_training_accuracy.png`](file:///home/suvranil/project/results/figures/federated_iid_training_accuracy.png)
- **FL Confusion Matrix**: [`results/figures/federated_iid_confusion_matrix.png`](file:///home/suvranil/project/results/figures/federated_iid_confusion_matrix.png)
- **Centralized vs. Federated Comparison**: [`results/figures/centralized_vs_federated_comparison.png`](file:///home/suvranil/project/results/figures/centralized_vs_federated_comparison.png)

---

## 4. Automated Quality & Regression Test Suite

- **Test Framework**: `pytest 9.1.1`
- **Total Tests**: **21 tests**
- **Test Results**: **21 passed (100%) in 3.91 seconds**
  - `tests/test_data_loading.py` (3 tests): Splits exist, batch shapes (1x28x28), deterministic shuffling.
  - `tests/test_model.py` (8 tests): Model architecture, parameters (420,802), output shapes `[1, 4, 16, 32, 2]`, forward validity, loss computation, gradient backpropagation step.
  - `tests/test_training.py` (3 tests): Split separation, loader batch shapes, medical metrics functions.
  - `tests/test_federated.py` (7 tests): IID partition integrity, stratification balance, determinism, parameter serialization invariance, client isolation, fit execution, server validation evaluation.

---

## 4. Summary of Archived CIFAR-10 Results
The previous generic proof-of-concept remains archived in [`results/archive/cifar10/`](file:///home/suvranil/project/results/archive/cifar10/README.md):
- **Dataset**: CIFAR-10 (60,000 images, 10 classes, 32x32 RGB).
- **Model**: SmallCNN (189,130 parameters).
- **Test Accuracy**: 76.94% (7,694 / 10,000 correct).
- **Training Time**: 100.97 seconds on RTX 3050 GPU.
- **Status**: Archived; superseded by PneumoniaMNIST for the hospital project.
