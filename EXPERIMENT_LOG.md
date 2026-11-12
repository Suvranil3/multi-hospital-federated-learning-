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

### [EXP-MED-002-FED-IID] - 10-Hospital IID Federated Learning Baseline on PneumoniaMNIST
- **Date**: 2026-10-31
- **Dataset Configuration**: PneumoniaMNIST
  - Training Split: 4,708 samples partitioned across 10 hospital clients
  - Validation Split: 524 samples (official val split, server-side evaluation each round)
  - Test Split: 624 samples (official untouched test split, evaluated only after training)
- **Number of Clients / Hospitals**: 10 simulated hospital clients
  - Sample distribution: 8 hospitals with 471 samples, 2 hospitals with 470 samples
  - Stratified per-hospital class ratio: ~25.8% Normal (121-122 samples), ~74.2% Pneumonia (349-350 samples)
- **IID / Non-IID Setting**: IID (Stratified uniform random partitioning)
- **Dirichlet Alpha**: N/A (Homogeneous case-mix; scheduled for Phase 4)
- **Attack Type**: None
- **Malicious Clients**: 0 (0%)
- **Model Architecture**: `SmallCNN` (2 Conv-BN-ReLU-MaxPool blocks, Dense 128, Dropout 0.2, 2 classes; **420,802 parameters**)
- **FL Algorithm**: Federated Averaging (`FedAvg`, Flower simulation via Virtual Client Engine)
- **Detector Version**: None (Phase 3 baseline)
- **Aggregation Method**: Standard sample-weighted FedAvg
- **Random Seed**: 42
- **Training Hyperparameters**:
  - Communication Rounds: 20
  - Local Training Epochs: 2 epochs per round per client
  - Local Batch Size: 32
  - Optimizer: Adam (lr = 0.001, weight decay = 0.0001)
  - Loss Criterion: CrossEntropyLoss
  - Augmentation: RandomHorizontalFlip, RandomRotation(10)
  - Normalization: Mean=(0.5719,), Std=(0.1684,)
- **Compute Resources**: CPU (AMD Ryzen 5 5600H, Virtual Client Engine with 3 concurrent Ray actors)
- **Results**:
  - Total Simulation Time: **108.29 seconds** (~5.41 seconds / communication round)
  - Initial Global Val Accuracy (Round 0): 25.95% | Initial Val Loss: 0.7122
  - Final Training Loss (Aggregated): 0.1108 | Final Training Accuracy: 95.62%
  - Best Server Validation Accuracy: **96.18%** — first achieved at **Round 17** (val_loss = 0.0833, saved as best checkpoint); also matched at Round 19 (val_loss = 0.0801, lower loss but same accuracy; checkpoint not overwritten)
  - Test metrics below are evaluated from the **Round 17 best checkpoint**:
  - Untouched Test Set Loss: **0.5306**
  - Untouched Test Set Accuracy: **85.74%** (535 / 624 correct)
  - Pneumonia Sensitivity / Recall: **97.95%** (382 / 390 pneumonia cases correctly detected)
  - Pneumonia Precision: **82.51%** (382 / 463)
  - Pneumonia F1-Score: **0.8957**
  - Normal Specificity / Recall: **65.38%** (153 / 234 normal cases correctly identified)
  - Normal Precision: **95.03%** (153 / 161)
  - Normal F1-Score: **0.7747**
  - Macro F1-Score: **0.8352**
  - Confusion Matrix: TN=153, FP=81, FN=8, TP=382
  - Comparison vs Centralized Baseline (Phase MEDICAL-02):
    - Test Accuracy: **85.74% vs 81.25%** (+4.49 percentage points)
    - Normal Specificity: **65.38% vs 50.43%** (+14.95 percentage points)
    - Pneumonia Precision: **82.51% vs 77.03%** (+5.48 percentage points)
    - Macro F1-Score: **0.8352 vs 0.7689** (+0.0663)
    - Pneumonia Sensitivity: **97.95% vs 99.74%** (-1.79 percentage points, 8 missed vs 1 missed, with massively reduced false positives from 116 down to 81)
  - Visual Artifacts:
    - Loss Curve: `results/figures/federated_iid_training_loss.png`
    - Accuracy Curve: `results/figures/federated_iid_training_accuracy.png`
    - Confusion Matrix: `results/figures/federated_iid_confusion_matrix.png`
    - Centralized vs Federated Comparison: `results/figures/centralized_vs_federated_comparison.png`
  - Model Checkpoints:
    - Best Model (Round 17, validation-selected): `results/models/best_federated_iid_model.pt` — test metrics above are from this checkpoint
    - Final Model (Round 20): `results/models/final_federated_iid_model.pt` — end-of-training weights (val_acc = 95.80%, test_acc = 86.38%)
  - Report File: `results/logs/federated_iid_results.json`
- **Execution Status**: SUCCEEDED (Phase 3)
- **Notes**: The federated experiment used **2.67× more total sample exposures** than the centralized baseline (20 rounds × 2 local epochs × 4,708 = 188,320 vs 15 epochs × 4,708 = 70,620), so the observed performance improvement (+4.49% test accuracy, +14.95% normal specificity) **cannot be attributed solely to FedAvg's implicit regularization effect**. The additional training compute is a confounding factor. Nonetheless, the IID federated setup demonstrates that distributed FedAvg training over 10 hospital clients produces a competitive global model, and establishes a clean IID control baseline for upcoming Non-IID heterogeneity studies (Phase 4).

---

### [EXP-MED-003-FED-NONIID-A10] - 10-Hospital Controlled Non-IID Federated Learning (alpha = 1.0)
- **Date**: 2026-10-31
- **Dataset Configuration**: PneumoniaMNIST
  - Training Split: 4,708 samples partitioned across 10 hospital clients with controlled sample sizes (8 with 471, 2 with 470)
  - Validation Split: 524 samples (official val split, server-side evaluation each round)
  - Test Split: 624 samples (official untouched test split, evaluated post-training)
- **Number of Clients / Hospitals**: 10 simulated hospital clients
- **IID / Non-IID Setting**: Non-IID (Population-calibrated constrained Beta/Dirichlet case-mix allocation)
- **Dirichlet Concentration Parameter (alpha)**: 1.0 (Total concentration parameter $\alpha = 1.0$; parameter vector $\boldsymbol{\alpha} = [0.2579, 0.7421]$)
- **Heterogeneity Metrics**:
  - Rejection Acceptance: Attempt 21 (initial unconstrained discrepancy: -29.38 samples)
  - Hospital Normal Ratio Mean: 25.78% | Std: 26.62% | Variance: 0.0709
  - Hospital Capacities: min = 470, max = 471, sum = 4,708
  - Normal Sample Total: exactly 1,214 | Pneumonia Sample Total: exactly 3,494
  - Mean Absolute Deviation from Dirichlet Target: 0.63% (Max: 1.88%)
- **Model Architecture**: `SmallCNN` (420,802 parameters, identical to Phase 2 & 3)
- **FL Algorithm**: `FedAvg` (Flower Virtual Client Engine, Ray)
- **BatchNorm Handling**: BatchNorm running buffers (`running_mean`, `running_var`, `num_batches_tracked`) included in sample-weighted FedAvg aggregation
- **Seeds**: Partition Seed = 42, Initialization Seed = 42, Training Seed = 42
- **Initial Model State**: Loaded from `results/models/initial_federated_model.pt` (Verified SHA-256: `0c411b83c2f0acef58beb5b9a968030d877e079eb2356eb3748903f8ae9bbc83`)
- **Training Hyperparameters**: 20 rounds, 2 local epochs/round, batch size = 32, Adam (lr = 0.001, wd = 1e-4)
- **Results**:
  - Simulation Duration: 103.93 seconds
  - Best Validation Accuracy: **95.99%** (Achieved at **Round 15**)
  - Untouched Test Set Loss: **0.7390**
  - Untouched Test Set Accuracy: **79.81%** (498 / 624 correct)
  - Pneumonia Sensitivity / Recall: **98.97%** (386 / 390 cases detected)
  - Pneumonia Precision: **75.98%** (386 / 508)
  - Pneumonia F1-Score: **0.8597**
  - Normal Specificity / Recall: **47.86%** (112 / 234 cases identified)
  - Normal Precision: **96.55%** (112 / 116)
  - Normal F1-Score: **0.6400**
  - Macro F1-Score: **0.7498**
  - Confusion Matrix: TN=112, FP=122, FN=4, TP=386
- **Artifacts**:
  - Case-Mix Distribution: `results/figures/hospital_distribution_alpha10.png`
  - Loss Curve: `results/figures/federated_non_iid_alpha10_training_loss.png`
  - Accuracy Curve: `results/figures/federated_non_iid_alpha10_training_accuracy.png`
  - Confusion Matrix: `results/figures/federated_non_iid_alpha10_confusion_matrix.png`
  - Checkpoints: `results/models/best_federated_non_iid_alpha10_model.pt` & `final_federated_non_iid_alpha10_model.pt`
  - Report File: `results/logs/federated_non_iid_alpha10_results.json`
- **Execution Status**: SUCCEEDED (Phase 4)

---

### [EXP-MED-004-FED-NONIID-A05] - 10-Hospital Controlled Non-IID Federated Learning (alpha = 0.5)
- **Date**: 2026-10-31
- **Dataset Configuration**: PneumoniaMNIST
  - Training Split: 4,708 samples partitioned across 10 hospital clients with controlled sample sizes (8 with 471, 2 with 470)
  - Validation Split: 524 samples (official val split, server-side evaluation each round)
  - Test Split: 624 samples (official untouched test split, evaluated post-training)
- **Number of Clients / Hospitals**: 10 simulated hospital clients
- **IID / Non-IID Setting**: Non-IID (Population-calibrated constrained Beta/Dirichlet case-mix allocation)
- **Dirichlet Concentration Parameter (alpha)**: 0.5 (Total concentration parameter $\alpha = 0.5$; parameter vector $\boldsymbol{\alpha} = [0.1289, 0.3711]$)
- **Heterogeneity Metrics**:
  - Rejection Acceptance: Attempt 6 (initial unconstrained discrepancy: -37.77 samples)
  - Hospital Normal Ratio Mean: 25.78% | Std: 37.18% | Variance: 0.1382
  - Hospital Capacities: min = 470, max = 471, sum = 4,708
  - Normal Sample Total: exactly 1,214 | Pneumonia Sample Total: exactly 3,494
  - Mean Absolute Deviation from Dirichlet Target: 0.81% (Max: 3.17%)
- **Model Architecture**: `SmallCNN` (420,802 parameters, identical to Phase 2 & 3)
- **FL Algorithm**: `FedAvg` (Flower Virtual Client Engine, Ray)
- **BatchNorm Handling**: BatchNorm running buffers (`running_mean`, `running_var`, `num_batches_tracked`) included in sample-weighted FedAvg aggregation
- **Seeds**: Partition Seed = 42, Initialization Seed = 42, Training Seed = 42
- **Initial Model State**: Loaded from `results/models/initial_federated_model.pt` (Verified SHA-256: `0c411b83c2f0acef58beb5b9a968030d877e079eb2356eb3748903f8ae9bbc83`)
- **Training Hyperparameters**: 20 rounds, 2 local epochs/round, batch size = 32, Adam (lr = 0.001, wd = 1e-4)
- **Results**:
  - Simulation Duration: 97.61 seconds
  - Best Validation Accuracy: **95.99%** (Achieved at **Round 18**)
  - Untouched Test Set Loss: **0.6930**
  - Untouched Test Set Accuracy: **80.45%** (502 / 624 correct)
  - Pneumonia Sensitivity / Recall: **98.46%** (384 / 390 cases detected)
  - Pneumonia Precision: **76.80%** (384 / 500)
  - Pneumonia F1-Score: **0.8629**
  - Normal Specificity / Recall: **50.43%** (118 / 234 cases identified)
  - Normal Precision: **95.16%** (118 / 124)
  - Normal F1-Score: **0.6592**
  - Macro F1-Score: **0.7611**
  - Confusion Matrix: TN=118, FP=116, FN=6, TP=384
- **Artifacts**:
  - Case-Mix Distribution: `results/figures/hospital_distribution_alpha05.png`
  - Loss Curve: `results/figures/federated_non_iid_alpha05_training_loss.png`
  - Accuracy Curve: `results/figures/federated_non_iid_alpha05_training_accuracy.png`
  - Confusion Matrix: `results/figures/federated_non_iid_alpha05_confusion_matrix.png`
  - Checkpoints: `results/models/best_federated_non_iid_alpha05_model.pt` & `final_federated_non_iid_alpha05_model.pt`
  - Report File: `results/logs/federated_non_iid_alpha05_results.json`
- **Execution Status**: SUCCEEDED (Phase 4)

---

### [EXP-MED-005-FED-NONIID-A01] - 10-Hospital Controlled Non-IID Federated Learning (alpha = 0.1)
- **Date**: 2026-10-31
- **Dataset Configuration**: PneumoniaMNIST
  - Training Split: 4,708 samples partitioned across 10 hospital clients with controlled sample sizes (8 with 471, 2 with 470)
  - Validation Split: 524 samples (official val split, server-side evaluation each round)
  - Test Split: 624 samples (official untouched test split, evaluated post-training)
- **Number of Clients / Hospitals**: 10 simulated hospital clients
- **IID / Non-IID Setting**: Non-IID (Population-calibrated constrained Beta/Dirichlet case-mix allocation)
- **Dirichlet Concentration Parameter (alpha)**: 0.1 (Total concentration parameter $\alpha = 0.1$; parameter vector $\boldsymbol{\alpha} = [0.0258, 0.0742]$)
- **Heterogeneity Metrics**:
  - Rejection Acceptance: Attempt 132 (initial unconstrained discrepancy: +20.17 samples)
  - Hospital Normal Ratio Mean: 25.80% | Std: 38.49% | Variance: 0.1482
  - Hospital Capacities: min = 470, max = 471, sum = 4,708
  - Normal Sample Total: exactly 1,214 | Pneumonia Sample Total: exactly 3,494
  - Mean Absolute Deviation from Dirichlet Target: 0.43% (Max: 1.62%)
- **Model Architecture**: `SmallCNN` (420,802 parameters, identical to Phase 2 & 3)
- **FL Algorithm**: `FedAvg` (Flower Virtual Client Engine, Ray)
- **BatchNorm Handling**: BatchNorm running buffers (`running_mean`, `running_var`, `num_batches_tracked`) included in sample-weighted FedAvg aggregation
- **Seeds**: Partition Seed = 42, Initialization Seed = 42, Training Seed = 42
- **Initial Model State**: Loaded from `results/models/initial_federated_model.pt` (Verified SHA-256: `0c411b83c2f0acef58beb5b9a968030d877e079eb2356eb3748903f8ae9bbc83`)
- **Training Hyperparameters**: 20 rounds, 2 local epochs/round, batch size = 32, Adam (lr = 0.001, wd = 1e-4)
- **Results**:
  - Simulation Duration: 95.34 seconds
  - Best Validation Accuracy: **89.89%** (Achieved at **Round 12**; subsequent rounds exhibited oscillation and representation drift)
  - Untouched Test Set Loss: **0.6326**
  - Untouched Test Set Accuracy: **85.10%** (531 / 624 correct)
  - Pneumonia Sensitivity / Recall: **91.79%** (358 / 390 cases detected)
  - Pneumonia Precision: **85.44%** (358 / 419)
  - Pneumonia F1-Score: **0.8850**
  - Normal Specificity / Recall: **73.93%** (173 / 234 cases identified)
  - Normal Precision: **84.39%** (173 / 205)
  - Normal F1-Score: **0.7882**
  - Macro F1-Score: **0.8366**
  - Confusion Matrix: TN=173, FP=61, FN=32, TP=358
- **Artifacts**:
  - Case-Mix Distribution: `results/figures/hospital_distribution_alpha01.png`
  - Loss Curve: `results/figures/federated_non_iid_alpha01_training_loss.png`
  - Accuracy Curve: `results/figures/federated_non_iid_alpha01_training_accuracy.png`
  - Confusion Matrix: `results/figures/federated_non_iid_alpha01_confusion_matrix.png`
  - Multi-Condition Comparison: `results/figures/iid_vs_non_iid_heterogeneity_comparison.png`
  - Checkpoints: `results/models/best_federated_non_iid_alpha01_model.pt` & `final_federated_non_iid_alpha01_model.pt`
  - Report File: `results/logs/federated_non_iid_alpha01_results.json`
  - Comparative Summary Log: `results/logs/phase4_heterogeneity_comparison.json`
- **Execution Status**: SUCCEEDED (Phase 4)

---

## Phase 5: Adversarial Attacks Simulation & Matched Honest Controls

### [EXP-MED-006-CTRL-IID] - Phase 5 Honest IID Control
- **Date**: 2026-10-31
- **Setting**: IID (Uniform class distribution, 10 honest hospitals, ~471 samples/client)
- **Attack Type**: None (Honest Consortium Baseline)
- **Initial Model State**: Loaded from `results/models/initial_federated_model.pt` (Verified SHA-256: `0c411b83c2f0acef58beb5b9a968030d877e079eb2356eb3748903f8ae9bbc83`)
- **Partition Hash**: `520e8d401be069a34ca8395b752dc487b732cc7fdae9481e34159498bc38fa04`
- **Training Hyperparameters**: 20 rounds, 2 local epochs/round, batch size = 32, Adam (lr = 0.001, wd = 1e-4), seeds = 42
- **Results**:
  - Best Validation Accuracy: **96.37%** (Achieved at **Round 18**)
  - Untouched Test Set Accuracy: **84.62%** (528 / 624 correct)
  - Pneumonia Sensitivity: **98.21%** (383 / 390 detected)
  - Normal Specificity: **61.97%** (145 / 234 identified)
  - Confusion Matrix: TN=145, FP=89, FN=7, TP=383
  - Attack-Induced Pneumonia Misclassification: **1.79%** (7 / 390 missed)
  - Triggered Pneumonia Baseline ASR: **0.51%** (2 / 390 misclassified on synthetic triggered test set)
- **Artifacts**: Checkpoints: `results/models/best_exp_med_006_ctrl_iid_model.pt`, Curves: `results/figures/exp_med_006_ctrl_iid_training_curves.png`, Confusion Matrix: `results/figures/exp_med_006_ctrl_iid_confusion_matrix.png`, Report: `results/logs/exp_med_006_ctrl_iid_results.json`
- **Execution Status**: SUCCEEDED (Phase 5)

---

### [EXP-MED-007-CTRL-NONIID-A05] - Phase 5 Honest Non-IID Control (alpha = 0.5)
- **Date**: 2026-10-31
- **Setting**: Non-IID Dirichlet Case-Mix ($\alpha = 0.5$, 10 honest hospitals, variance 0.1382)
- **Attack Type**: None (Honest Consortium Baseline)
- **Initial Model State**: Verified SHA-256: `0c411b83c2f0acef58beb5b9a968030d877e079eb2356eb3748903f8ae9bbc83`
- **Partition Hash**: `805ae5812d637dd78b40bb3858055627237fa74a4ff49942a78faef670f55cfd`
- **Training Hyperparameters**: 20 rounds, 2 local epochs/round, batch size = 32, Adam (lr = 0.001, wd = 1e-4), seeds = 42
- **Results**:
  - Best Validation Accuracy: **95.61%** (Achieved at **Round 20**)
  - Untouched Test Set Accuracy: **78.37%** (489 / 624 correct)
  - Pneumonia Sensitivity: **98.97%** (386 / 390 detected)
  - Normal Specificity: **44.02%** (103 / 234 identified)
  - Confusion Matrix: TN=103, FP=131, FN=4, TP=386
  - Attack-Induced Pneumonia Misclassification: **1.03%** (4 / 390 missed)
  - Triggered Pneumonia Baseline ASR: **0.26%** (1 / 390 misclassified on synthetic triggered test set)
- **Artifacts**: Checkpoints: `results/models/best_exp_med_007_ctrl_noniid_a05_model.pt`, Curves: `results/figures/exp_med_007_ctrl_noniid_a05_training_curves.png`, Confusion Matrix: `results/figures/exp_med_007_ctrl_noniid_a05_confusion_matrix.png`, Report: `results/logs/exp_med_007_ctrl_noniid_a05_results.json`
- **Execution Status**: SUCCEEDED (Phase 5)

---

### [EXP-MED-008-ATK-LF-IID-M1] - Targeted Label Flipping (IID, M=1, Hospital 0)
- **Date**: 2026-10-31
- **Setting**: IID (Partition Hash: `520e8d401be069a34ca8395b752dc487b732cc7fdae9481e34159498bc38fa04`)
- **Malicious Participant**: Hospital `0` (10% consortium compromised, 349 pneumonia labels flipped to 0)
- **Matched Control**: `EXP-MED-006-CTRL-IID`
- **Results**:
  - Best Validation Accuracy: **94.66%** (Round 14)
  - Clean Test Set Accuracy: **88.78%** (554 / 624 correct)
  - Pneumonia Sensitivity: **96.41%** (376 / 390 detected; decreased from 98.21% control)
  - Normal Specificity: **76.07%** (178 / 234 identified)
  - Confusion Matrix: TN=178, FP=56, FN=14, TP=376
  - Attack-Induced Pneumonia Misclassification: **3.59%** (14 / 390 missed)
  - **Matched Control Comparison**: False Negatives increased from 7 to 14 ($\Delta \text{FN} = \mathbf{+7}$, **100% increase** in missed diagnoses)
- **Artifacts**: Checkpoints: `results/models/best_exp_med_008_atk_lf_iid_m1_model.pt`, Report: `results/logs/exp_med_008_atk_lf_iid_m1_results.json`
- **Execution Status**: SUCCEEDED (Phase 5)

---

### [EXP-MED-009-ATK-LF-IID-M2] - Targeted Label Flipping (IID, M=2, Hospitals 0 & 1)
- **Date**: 2026-10-31
- **Setting**: IID (Partition Hash: `520e8d401be069a34ca8395b752dc487b732cc7fdae9481e34159498bc38fa04`)
- **Malicious Participants**: Hospitals `0, 1` (20% consortium compromised, 698 pneumonia labels flipped to 0)
- **Matched Control**: `EXP-MED-006-CTRL-IID`
- **Results**:
  - Best Validation Accuracy: **86.64%** (Round 2; subsequent training degraded significantly)
  - Clean Test Set Accuracy: **86.54%** (540 / 624 correct)
  - Pneumonia Sensitivity: **84.62%** (330 / 390 detected; severe drop from 98.21% control)
  - Normal Specificity: **89.74%** (210 / 234 identified)
  - Confusion Matrix: TN=210, FP=24, FN=60, TP=330
  - Attack-Induced Pneumonia Misclassification: **15.38%** (60 / 390 missed)
  - **Matched Control Comparison**: False Negatives surged from 7 to 60 ($\Delta \text{FN} = \mathbf{+53}$, **+757% increase** / 8.57x jump in missed diagnoses)
- **Artifacts**: Checkpoints: `results/models/best_exp_med_009_atk_lf_iid_m2_model.pt`, Report: `results/logs/exp_med_009_atk_lf_iid_m2_results.json`
- **Execution Status**: SUCCEEDED (Phase 5)

---

### [EXP-MED-010-ATK-LF-NONIID-M1] - Targeted Label Flipping (Non-IID alpha=0.5, M=1, Hospital 0)
- **Date**: 2026-10-31
- **Setting**: Non-IID ($\alpha = 0.5$, Partition Hash: `805ae5812d637dd78b40bb3858055627237fa74a4ff49942a78faef670f55cfd`)
- **Malicious Participant**: Hospital `0` (10% consortium compromised, 428 pneumonia labels flipped to 0)
- **Matched Control**: `EXP-MED-007-CTRL-NONIID-A05`
- **Results**:
  - Best Validation Accuracy: **95.99%** (Round 20)
  - Clean Test Set Accuracy: **85.90%** (536 / 624 correct)
  - Pneumonia Sensitivity: **97.18%** (379 / 390 detected; decreased from 98.97% control)
  - Normal Specificity: **67.09%** (157 / 234 identified)
  - Confusion Matrix: TN=157, FP=77, FN=11, TP=379
  - Attack-Induced Pneumonia Misclassification: **2.82%** (11 / 390 missed)
  - **Matched Control Comparison**: False Negatives nearly tripled from 4 to 11 ($\Delta \text{FN} = \mathbf{+7}$, **+175% increase** in missed diagnoses)
- **Artifacts**: Checkpoints: `results/models/best_exp_med_010_atk_lf_noniid_m1_model.pt`, Report: `results/logs/exp_med_010_atk_lf_noniid_m1_results.json`
- **Execution Status**: SUCCEEDED (Phase 5)

---

### [EXP-MED-011-ATK-MP-INV-IID-M1] - Model Update Inversion (IID, M=1, gamma = -1.0)
- **Date**: 2026-10-31
- **Setting**: IID (Partition Hash: `520e8d401be069a34ca8395b752dc487b732cc7fdae9481e34159498bc38fa04`)
- **Malicious Participant**: Hospital `0` (10% consortium, $\gamma = -1.0$ applied strictly to 12 trainable parameter tensors)
- **Matched Control**: `EXP-MED-006-CTRL-IID`
- **Results**:
  - Best Validation Accuracy: **95.80%** (Round 18)
  - Clean Test Set Accuracy: **84.94%** (530 / 624 correct)
  - Pneumonia Sensitivity: **97.95%** (382 / 390 detected)
  - Normal Specificity: **63.25%** (148 / 234 identified)
  - Confusion Matrix: TN=148, FP=86, FN=8, TP=382
  - **Matched Control Comparison**: Control FN: 7 $\to$ Attack FN: 8 ($\Delta \text{FN} = \mathbf{+1}$). Demonstrates that 9 honest clients (weight 0.9) partially dampens a single inverted client (weight 0.1).
- **Artifacts**: Checkpoints: `results/models/best_exp_med_011_atk_mp_inv_iid_m1_model.pt`, Report: `results/logs/exp_med_011_atk_mp_inv_iid_m1_results.json`
- **Execution Status**: SUCCEEDED (Phase 5)

---

### [EXP-MED-012-ATK-MP-SCALE-IID-M1] - Model Update Scaling (IID, M=1, gamma = +10.0)
- **Date**: 2026-10-31
- **Setting**: IID (Partition Hash: `520e8d401be069a34ca8395b752dc487b732cc7fdae9481e34159498bc38fa04`)
- **Malicious Participant**: Hospital `0` (10% consortium, $\gamma = +10.0$ applied strictly to 12 trainable parameter tensors)
- **Matched Control**: `EXP-MED-006-CTRL-IID`
- **Results**:
  - Best Validation Accuracy: **96.56%** (Round 13)
  - Clean Test Set Accuracy: **86.70%** (541 / 624 correct)
  - Pneumonia Sensitivity: **97.69%** (381 / 390 detected)
  - Normal Specificity: **68.38%** (160 / 234 identified)
  - Confusion Matrix: TN=160, FP=74, FN=9, TP=381
  - **Matched Control Comparison**: Control FN: 7 $\to$ Attack FN: 9 ($\Delta \text{FN} = \mathbf{+2}$)
- **Artifacts**: Checkpoints: `results/models/best_exp_med_012_atk_mp_scale_iid_m1_model.pt`, Report: `results/logs/exp_med_012_atk_mp_scale_iid_m1_results.json`
- **Execution Status**: SUCCEEDED (Phase 5)

---

### [EXP-MED-013-ATK-BD-IID-M1] - Backdoor Trigger Injection (IID, M=1, rho = 0.50)
- **Date**: 2026-10-31
- **Setting**: IID (Partition Hash: `520e8d401be069a34ca8395b752dc487b732cc7fdae9481e34159498bc38fa04`)
- **Malicious Participant**: Hospital `0` (10% consortium, 174 pneumonia scans stamped with $3\times3$ white trigger, normalized intensity $+2.5422$)
- **Matched Control**: `EXP-MED-006-CTRL-IID`
- **Results**:
  - Clean Test Set Accuracy (CDA): **86.06%** (537 / 624 correct; clean performance pristine)
  - Clean Pneumonia Sensitivity: **97.44%** (380 / 390 detected)
  - Normal Specificity: **67.09%** (157 / 234 identified)
  - Confusion Matrix: TN=157, FP=77, FN=10, TP=380
  - Matched Control Baseline ASR: **0.51%** (2 / 390 triggered scans classified Normal)
  - Backdoor Triggered ASR: **7.44%** (29 / 390 triggered scans classified Normal)
  - **Attack-Induced Delta ASR Increase**: $\mathbf{+6.93\%}$ (**14.6x jump** in backdoor activation rate)
- **Artifacts**: Checkpoints: `results/models/best_exp_med_013_atk_bd_iid_m1_model.pt`, Report: `results/logs/exp_med_013_atk_bd_iid_m1_results.json`
- **Execution Status**: SUCCEEDED (Phase 5)

---

### [EXP-MED-014-ATK-BD-NONIID-M1] - Backdoor Trigger Injection (Non-IID alpha=0.5, M=1, rho = 0.50)
- **Date**: 2026-10-31
- **Setting**: Non-IID ($\alpha = 0.5$, Partition Hash: `805ae5812d637dd78b40bb3858055627237fa74a4ff49942a78faef670f55cfd`)
- **Malicious Participant**: Hospital `0` (10% consortium, 214 pneumonia scans stamped with $3\times3$ white trigger)
- **Matched Control**: `EXP-MED-007-CTRL-NONIID-A05`
- **Results**:
  - Clean Test Set Accuracy (CDA): **84.46%** (527 / 624 correct; clean performance pristine)
  - Clean Pneumonia Sensitivity: **97.95%** (382 / 390 detected)
  - Normal Specificity: **61.97%** (145 / 234 identified)
  - Confusion Matrix: TN=145, FP=89, FN=8, TP=382
  - Matched Control Baseline ASR: **0.26%** (1 / 390 triggered scans classified Normal)
  - Backdoor Triggered ASR: **8.46%** (33 / 390 triggered scans classified Normal)
  - **Attack-Induced Delta ASR Increase**: $\mathbf{+8.20\%}$ (**32.5x jump** over baseline control; Non-IID heterogeneity amplifies backdoor vulnerability)
- **Artifacts**: Checkpoints: `results/models/best_exp_med_014_atk_bd_noniid_m1_model.pt`, Report: `results/logs/exp_med_014_atk_bd_noniid_m1_results.json`
- **Comparative Phase 5 Summary**: Figure: `results/figures/phase5_attack_comparison.png`, JSON Log: `results/logs/phase5_attack_comparison.json`
---

## Phase 6: Multi-Factor Non-IID-Aware Malicious Hospital Detection

### [EXP-MED-015-DETECTOR-CALIB] - Quantile-Based Calibration on Honest Baseline Runs
- **Date**: 2026-10-31
- **Objective**: Calibrate empirical quantile thresholds across five feature groups using honest client distributions (no arbitrary offsets).
- **Calibration Cohort**: 400 client-round observations from `EXP-MED-006-CTRL-IID` (200 records) and `EXP-MED-007-CTRL-NONIID-A05` (200 records).
- **Target Quantiles**: Malicious $\text{FPR} \le 1.0\%$ ($Q_{0.99}$), Suspicious $\text{FPR} \le 5.0\%$ ($Q_{0.95}$); Bonferroni layer correction ($Q_{0.9975}$ and $Q_{0.9875}$).
- **Derived Empirical Thresholds**:
  - Global Modified Z-Score: Suspicious = $4.38$, Malicious = $9.10$
  - Max Layer Modified Z-Score: Suspicious = $9.10$, Malicious = $12.41$
  - Directional Cosine Similarity ($\cos_{\text{ref}}$): Suspicious = $0.1465$, Catastrophic Inversion = $-0.75$, Corroborated Negative Boundary = $0.0000$
  - Uncompensated Validation Loss Impact ($\Delta \mathcal{L}_{\text{total}}$): Suspicious = $0.0170$, Malicious = $0.0274$
  - Historical Trajectory EMA Accumulator: Suspicious = $0.4768$, Malicious = $0.8023$
- **Artifacts**: Config: `results/phase6_detector/calibration/`

---

### [EXP-MED-016-DETECTOR-BENCHMARK-DUAL] - Dual Reporting: Calibration-Excluded Out-of-Sample vs In-Sample Benchmark
- **Date**: 2026-10-31
- **Objective**: Eliminate calibration-data leakage by evaluating strictly on held-out runs, reporting dual metrics for transparent comparison.
- **Evaluation Cohorts**:
  - **Calibration-Excluded (Out-of-Sample, 9 Simulations)**: 1,800 client-round observations (1,640 honest, 160 malicious).
  - **In-Sample Reference (11 Simulations)**: 2,200 client-round observations (2,040 honest, 160 malicious).
- **Dual Results Scorecard**:
  - **Continuous Discrimination**:
    - Out-of-Sample: **AUROC = 0.9814**, **AUPRC = 0.8801**
    - In-Sample Reference: **AUROC = 0.9761**, **AUPRC = 0.8026**
  - **Binary Malicious Decision**:
    - Out-of-Sample: **TPR = 13.13%** (21 / 160), **FPR = 0.00%** (0 / 1,640), **Precision = 100.00%**, **F1 = 23.20%**
    - In-Sample Reference: **TPR = 13.13%** (21 / 160), **FPR = 0.00%** (0 / 2,040), **Precision = 100.00%**, **F1 = 23.20%**
  - **Tri-State Flagged Decision (SUSPICIOUS or MALICIOUS)**:
    - Out-of-Sample: **Total Flagged Recall = 96.25%** (154 / 160), **Flagged FPR = 10.00%** (164 / 1,640), **Precision = 48.43%**, **F1 = 64.44%**
    - In-Sample Reference: **Total Flagged Recall = 96.25%** (154 / 160), **Flagged FPR = 9.95%** (203 / 2,040)
  - **Tri-State Confusion Matrix (Out-of-Sample)**:
    - Honest ($N=1,640$): NORMAL = 1,476 (90.00%), SUSPICIOUS = 164 (10.00%), MALICIOUS = 0 (0.00%)
    - Malicious ($N=160$): NORMAL = 6 (3.75%), SUSPICIOUS = 133 (83.13%), MALICIOUS = 21 (13.13%)
  - **Stratified False Positive Rates (Out-of-Sample)**:
    - Honest IID ($N=880$): **False Malicious = 0.00%** (0 / 880), Suspicious = 0.11% (1 / 880), Normal = 99.89% (879 / 880)
    - Honest Non-IID $\alpha=1.0$ ($N=200$): **False Malicious = 0.00%** (0 / 200), Suspicious = 19.00% (38 / 200), Normal = 81.00% (162 / 200)
    - Honest Non-IID $\alpha=0.5$ ($N=360$): **False Malicious = 0.00%** (0 / 360), Suspicious = 21.94% (79 / 360), Normal = 78.06% (281 / 360)
    - Honest Non-IID $\alpha=0.1$ ($N=200$): **False Malicious = 0.00%** (0 / 200), Suspicious = 23.00% (46 / 200), Normal = 77.00% (154 / 200)
  - **Per-Attack Detection & Latency**:
    - Label Flipping IID $M=1$: 0% Malicious, 100% Suspicious (**100.0% Flagged**; Latency: Round 1)
    - Label Flipping IID $M=2$: 0% Malicious, 95% Suspicious (**95.0% Flagged**; Latency: Round 2)
    - Label Flipping Non-IID $M=1$: 0% Malicious, 100% Suspicious (**100.0% Flagged**; Latency: Round 1)
    - Model Poisoning Invert $\gamma=-1.0$: 5% Malicious, 95% Suspicious (**100.0% Flagged**; Latency: Round 1)
    - Model Poisoning Scale $\gamma=+10.0$: **100.0% Malicious** (**100.0% Flagged**; Latency: Round 1)
    - Backdoor Trigger IID: 0% Malicious, 95% Suspicious (**95.0% Flagged**; Latency: Round 2)
    - Backdoor Trigger Non-IID: 0% Malicious, 85% Suspicious (**85.0% Flagged**; Latency: Round 2)
- **Artifacts**: Evaluation JSON: `results/phase6_detector/evaluation/detection_results.json`, Figure: `results/phase6_detector/figures/phase6_detector_performance.png`
- **Execution Status**: SUCCEEDED (Phase 6 Complete)

---

### [EXP-MED-017-INDEPENDENT-SEEDS] - Multi-Seed Independent Honest Benchmark Suite (Seeds 43, 44, 45)
- **Date**: 2026-10-31
- **Objective**: Establish true out-of-sample honest generalization across 12 independent simulations (Seeds 43, 44, 45 across IID and Non-IID $\alpha \in \{1.0, 0.5, 0.1\}$), avoiding seed-42 bias while preserving model initialization hash (`0c411b83c2f0acef58beb5b9a968030d877e079eb2356eb3748903f8ae9bbc83`).
- **Total Independent Honest Observations**: 2,400 client-round records (600 records per heterogeneity condition).
- **Stratified Independent Generalization Results**:
  - Honest IID ($N=600$): **False Malicious = 0.00%** (0 / 600), Suspicious = 0.50% (3 / 600), Normal = 99.50% (597 / 600)
  - Honest Non-IID $\alpha=1.0$ ($N=600$): **False Malicious = 0.00%** (0 / 600), Suspicious = 18.83% (113 / 600), Normal = 81.17% (487 / 600)
  - Honest Non-IID $\alpha=0.5$ ($N=600$): **False Malicious = 0.00%** (0 / 600), Suspicious = 28.00% (168 / 600), Normal = 72.00% (432 / 600)
  - Honest Non-IID $\alpha=0.1$ ($N=600$): **False Malicious = 0.17%** (1 / 600), Suspicious = 36.33% (218 / 600), Normal = 63.50% (381 / 600)
  - **Aggregate Independent Multi-Seed False Malicious Rate**: **0.04%** (1 / 2,400)
- **Artifacts**: Summary: `results/phase6_detector/independent_honest_suite_summary.json`
- **Execution Status**: SUCCEEDED

---

### [EXP-MED-018-FORENSIC-AUDIT] - Forensic Audit of Honest False Positives & Negative Cosine Dispersion
- **Date**: 2026-10-31
- **Objective**: Audit all honest observations classified as MALICIOUS or SUSPICIOUS across 16 honest simulations (3,200 observations), specifically investigating negative cosine similarity among Non-IID hospitals.
- **Key Findings**:
  - **Natural Occurrence of Negative Cosine in Non-IID**: Across 3,200 honest observations, 199 events (6.22%) exhibited negative cosine ($\cos_{\text{ref}} < 0.0$), ranging from $-0.8411$ to $-0.0002$ (mean: $-0.3103$). Occurrence scales directly with heterogeneity: IID = 2.43%, $\alpha=1.0$ = 5.50%, $\alpha=0.5$ = 8.50%, $\alpha=0.1$ = 13.50%.
  - **Root Cause of Extreme Negative Cosine**: In $\alpha=0.1$, minority hospitals (e.g. 467 Normal / 3 Pneumonia) compute gradient updates opposing the majority Pneumonia cohort (e.g. 7 hospitals with 99% Pneumonia cases). Their updates dramatically improve Normal validation loss ($\Delta \mathcal{L}_{\text{normal}} = -6.1091$ to $-8.4208$).
  - **Symmetrical Clinical Trade-Off Efficacy**: Protected **199 / 199 (100.0%)** of honest negative-cosine events from uncompensated loss penalties.
  - **Decision Breakdown for Negative Cosine**: 199 / 199 classified as `SUSPICIOUS` (monitored for Non-IID variance), **0 classified as `MALICIOUS`**.
  - **Statistical Artifact in Low-MAD Cohorts**: In $\alpha=0.1$, 7 clients had near-identical norms ($\approx 3.4$), shrinking the cohort MAD to $\approx 0.05$. Minority clients with norm $6.4$ ($1.87\times$ median) produced Modified Z-scores of $35+$. Coupling $Z > 9.10$ with `norm_ratio_global > 5.0` completely eliminated this artifact while retaining 100% recall on genuine scaling attacks ($\gamma=10.0$, ratio $\approx 9.9$).
  - **Single Residual False Malicious Event**: `EXP-MED-HONEST-A01-S45` Hospital 6 at Round 10: a minority hospital (469 Normal / 2 Pneumonia) with layer $Z=67.75$ during a transient round where both classes saw small loss increases ($\Delta \mathcal{L}_{\text{total}} = 0.0507$).
- **Artifacts**: Report: `results/phase6_detector/evaluation/honest_false_positive_audit.json`
- **Execution Status**: SUCCEEDED

---

### [EXP-MED-019-INSTRUMENTATION-PARITY] - Direct Mathematical Equivalence Test & Concurrency Analysis
- **Date**: 2026-10-31
- **Objective**: Formally audit replay numerical reproducibility vs mathematical aggregation equivalence.
- **Direct Parity Unit Test**: [`tests/test_instrumentation_parity.py`](file:///home/suvranil/project/tests/test_instrumentation_parity.py).
- **Mathematical Equivalence Result**: Bit-for-bit exact parameter identity (**0.000 maximum absolute parameter difference**) across all 12 trainable parameter tensors and all 6 BatchNorm buffers when supplied with identical client fit result objects.
- **Concurrency & Replay Root Cause**: The parameter deviations reported in earlier replays ($0.642 - 2.590$) are driven by **Ray virtual client actor pool concurrency**: Flower's `VirtualClientEngineActorPool` executes client trainings across 12 CPU cores asynchronously. Floating-point non-associativity across multi-threaded PyTorch CPU gradient sums causes minute perturbations ($\sim 10^{-7}$) that compound over 20 rounds of nonlinear SGD.
- **Three-Tier Reproducibility Clarification**:
  1. *Aggregation Equivalence*: Mathematically exact ($0.000$ deviation).
  2. *Training Replay Reproducibility*: Structurally bounded by asynchronous actor scheduling concurrency.
  3. *Checkpoint Identity*: Historical Phase 4/5 checkpoints preserved untouched; instrumentation checkpoints stored separately in `results/phase6_detector/checkpoints/`.
- **Execution Status**: SUCCEEDED (51 / 51 tests passing)

---

### [EXP-MED-020-BLIND-EVALUATION] - Phase 6 Blind Evaluation Suite (Fresh Untouched Seeds 111, 112, 113)
- **Date**: 2026-11-01
- **Objective**: Execute a rigorous, cryptographically verifiable blind evaluation of the frozen tri-state detector (`federated_learning/detection/frozen_v1/`) across a pristine, untouched seed cohort (`111, 112, 113`). Prior seed-101 runs were segregated into `results/phase6_preflight_exploratory_seed101/` to ensure absolute seed purity. All pre-execution pipeline checks used engineering seed `9999` with zero test set access.
- **Evaluation Standard**: Zero threshold re-calibration, zero code changes between seeds, read-only frozen detector (`chmod 444`), official test split (N=624) evaluated strictly post-training at the best validation checkpoint.
- **Total Simulations**: 33 full simulations (11 configurations $\times$ 3 seeds).
- **Total Observations**: 6,600 client-round records (6,120 honest, 480 malicious).
- **Two-Tier Scorecard**:
  - **Hard MALICIOUS Level (Direct Exclusion)**:
    - **False Positive Rate (FPR)**: **0.00%** (0 / 6,120 honest observations falsely banned across all IID and Non-IID regimes).
    - **Recall (TPR)**: **12.92%** (62 / 480 malicious observations).
    - **Precision**: **100.00%** (62 / 62).
    - **F1-Score**: 0.2288.
  - **Overall Flagged Level (Supervisory Audit Gate)**:
    - **Threat Recall (TPR)**: **93.75%** (450 / 480 malicious observations flagged as SUSPICIOUS or MALICIOUS).
    - **Honest Flagged Rate (FPR)**: **13.15%** (805 / 6,120 honest observations routed to supervisory review).
    - **Precision**: 35.86% (450 / 1,255).
    - **F1-Score**: 0.5188.
- **Tri-State Confusion Matrix ($N=6,600$)**:
  - Predicted NORMAL: Honest = 5,315 (86.85%), Malicious = 30 (6.25%)
  - Predicted SUSPICIOUS: Honest = 805 (13.15%), Malicious = 388 (80.83%)
  - Predicted MALICIOUS: Honest = 0 (0.00%), Malicious = 62 (12.92%)
- **Attack-Specific Detection**:
  - Model Scaling ($\gamma=+10.0$): **100.0% Hard Malicious** (60/60) at Round 1 across all 3 seeds.
  - Model Inversion ($\gamma=-1.0$): **100.0% Flagged** (60/60; 2 hard malicious, 58 suspicious).
  - Label Flipping IID $M=1$: **98.33% Flagged** (59/60).
  - Label Flipping IID $M=2$: **100.00% Flagged** (120/120).
  - Label Flipping Non-IID $\alpha=0.5$ $M=1$: **90.00% Flagged** (54/60).
  - Backdoor IID $M=1$: **91.67% Flagged** (55/60); Mean Test ASR: 7.01%.
  - Backdoor Non-IID $\alpha=0.5$ $M=1$: **70.00% Flagged** (42/60); Mean Test ASR: 4.87%.
- **Stratified Honest Flagged Rates**:
  - Honest IID ($N=3,240$): False Malicious = 0.00% (0), Suspicious = 0.43% (14), Normal = 99.57% (3,226)
  - Non-IID $\alpha=1.0$ ($N=600$): False Malicious = 0.00% (0), Suspicious = 17.17% (103), Normal = 82.83% (497)
  - Non-IID $\alpha=0.5$ ($N=1,680$): False Malicious = 0.00% (0), Suspicious = 27.80% (467), Normal = 72.20% (1,213)
  - Non-IID $\alpha=0.1$ ($N=600$): False Malicious = 0.00% (0), Suspicious = 36.83% (221), Normal = 63.17% (379)
- **Multi-Seed Stability (Seeds 111, 112, 113)**:
  - Hard Malicious Recall: Seed 111 = 13.12%, Seed 112 = 13.12%, Seed 113 = 12.50% (Std Dev: 0.36%).
  - Hard Malicious FPR: Exactly 0.00% across all 3 seeds.
  - Flagged Recall: Seed 111 = 91.88%, Seed 112 = 95.62%, Seed 113 = 93.75% (Std Dev: 1.87%).
- **Cryptographic Hashes**:
  - Frozen Detector (`detector.py`): `6576468da037927fda3283a1e2d8402f3b2739d6c45b6fe8ad517900e94a7f07`
  - Frozen Thresholds (`calibrated_thresholds.json`): `737c4819df70a8cadc93d9ff9a2db82bc36040403e11ee1240031c9b0cf34507`
  - Evaluation Script (`run_phase6_blind_evaluation.py`): `de5fd065f384a8f6de4feac9232d95a3e85fec1dafa7d10fd10293c93b07f1f6`
- **Artifacts**:
  - Evaluation Summary: `results/phase6_blind_evaluation/blind_suite_summary.json`
  - Evaluation Results: `results/phase6_blind_evaluation/evaluation/blind_evaluation_results.json`
  - Pre-Execution Manifest: `results/phase6_blind_evaluation/PRE_EXECUTION_REPRODUCIBILITY_MANIFEST.json`
  - Standalone Report: `docs/PHASE6_BLIND_EVALUATION_REPORT.md`
- **Execution Status**: SUCCEEDED (Phase 6 Fully Complete)

---

### [EXP-MED-021-BLOCKCHAIN-AUDIT] - Permissioned Blockchain Audit Layer Implementation and Cryptographic Verification
- **Date**: 2026-11-02
- **Objective**: Implement a lightweight, tamper-evident permissioned audit ledger component (`blockchain_audit/`) tailored for the 8 GB RAM profile to anchor federated training events, detector decisions, exclusions, and result artifact hashes.
- **Ledger Path**: `results/blockchain_audit/audit_ledger.jsonl`
- **Key Modules**:
  - `blockchain_audit/block.py`: Canonical JSON serialization, deterministic SHA-256 block hashing, and genesis definition (`previous_hash = "0"*64`).
  - `blockchain_audit/blockchain.py`: Disk-backed append-only ledger with `fsync`, $O(1)$-memory streaming verification, and interrupted write recovery.
  - `blockchain_audit/audit_logger.py`: Federated lifecycle logging (experiment init, batched communication rounds, final completions with artifact SHA-256).
  - `blockchain_audit/verification.py`: Standalone audit verifier for genesis, hash validity, backwards link continuity, duplicate protection, and disk artifact verification.
- **Privacy Policy**: Zero patient records, zero raw images, zero private client datasets, zero model weights/tensors stored.
- **Unit Test Scorecard**: **10 / 10 tests passed (100%)** in 0.04s (`tests/test_blockchain_audit.py`).
- **Regression Suite**: **83 / 83 tests passed (100%)** in 7.92s with zero regressions.
- **Execution Status**: SUCCEEDED

---

### [EXP-MED-022-PHASE7-MEMORY-SAFE] - Phase 7 Resumable Sequential Execution Architecture & Baseline Validation
- **Date**: 2026-11-02
- **Objective**: Resolve system memory exhaustion from previous Ray actor pool OOM crashes by deploying a strictly sequential in-process execution engine (`experiments/run_phase7_memory_safe.py`) under the 8 GB RAM profile.
- **Hardware Profile**: 8 GB RAM (6,749 MB usable), 12 CPU cores, CPU device, 5.2–5.4 GB base memory load.
- **Architecture Innovations**:
  - Direct in-process sequential client fitting (1 client at a time; zero Ray processes, zero Ray object store).
  - Explicit tensor and client instance destruction with per-round garbage collection (`gc.collect()`).
  - Real-time memory monitoring via `/proc/meminfo` with automatic pause circuit breaker at $<250\text{ MB}$ available RAM.
  - Persistent execution registry (`results/phase7_robust_aggregation/execution_registry.json`) for automatic skipping and seamless resumption.
  - Integrated blockchain audit ledger anchoring 100% of lifecycle events and artifact SHA-256 checksums.
- **Initial Baseline Execution Scorecard (Seed 211, Honest IID)**:
  - `EXP-P7-FEDAVG-HONEST-IID-S211`: Runtime 87.34s | Peak RSS 907.8 MB | Test Acc 85.74% | Sens 96.41% | Spec 67.95% | Macro F1 83.78% | Blockchain Block #43.
  - `EXP-P7-COORDINATE_MEDIAN-HONEST-IID-S211`: Runtime 89.45s | Peak RSS 907.7 MB | Test Acc 86.54% | Sens 96.67% | Spec 69.66% | Macro F1 84.74% | Blockchain Block #65.
  - `EXP-P7-TRIMMED_MEAN_BETA1-HONEST-IID-S211`: Runtime 87.06s | Peak RSS 1097.0 MB | Test Acc 86.22% | Sens 96.67% | Spec 68.80% | Macro F1 84.34% | Blockchain Block #87.
- **Memory Stability**: System RAM usage remained flat and stable throughout ($5,225 - 5,359\text{ MB}$ used, $>1.4\text{ GB}$ free). Zero memory leaks.
- **Registry & Ledger Audit**: Resumption check verified (completed experiments skipped in 0.001s); all 88 committed blocks validated with zero errors.
- **Execution Status**: SUCCEEDED (Phase 7 Memory-Safe Foundation Fully Operational)

---

### [EXP-MED-023-PHASE7-BATCH1] - Phase 7 Resumable Sequential Execution: Next Five Simulations & Parity Verification
- **Date**: 2026-11-02
- **Objective**: Execute the next five simulations of the 351-experiment Phase 7 matrix using the memory-safe runner (`experiments/run_phase7_memory_safe.py`), verify parity between sequential execution and Flower aggregation, and commit full lifecycle events to the permissioned audit ledger.
- **Hardware & Memory Profile**: 8 GB host RAM (6.7 GB usable, 12 CPU cores). Peak Process RSS: **1,049.2 MB**. Minimum Available RAM: **1,373.4 MB** (comfortably above the 250 MB circuit breaker threshold).
- **Execution Model**: Strictly sequential (one simulation at a time, one hospital client at a time, zero Ray processes, zero background workers).
- **Reproducibility & Parity Audit**:
  - Implemented `tests/test_phase7_sequential_parity.py` (6 tests passing).
  - Validated initial model checkpoint hash parity (`0c411b83c2f0acef58beb5b9a968030d877e079eb2356eb3748903f8ae9bbc83`).
  - Validated MockClientProxy + FitRes numerical and deterministic hashing parity across aggregation rules (`fedavg`, `coordinate_median`, `trimmed_mean_beta1`, `multi_krum_f1`).
  - Validated BatchNorm floating-point running buffer weighted linear combinations and integer scalar step counter advancements.
  - Full project test suite: **89 passed, 0 failed** in 11.14s.
- **Executed Simulations (Seed 211, Honest IID)**:
  1. `EXP-P7-TRIMMED_MEAN_BETA2-HONEST-IID-S211`:
     - Runtime: 97.99s | Peak RSS: 856.2 MB | Available RAM: 1,541.2 MB
     - Test Accuracy: **86.70%** | Sensitivity: **95.90%** | Specificity: **71.37%** | Macro F1: **85.05%** | Test Loss: 0.3804
     - Confusion Matrix: TP=374, FN=16, TN=167, FP=67
     - Blockchain Completion Block: **#109** (Lifecycle Blocks #88–109)
  2. `EXP-P7-MULTI_KRUM_F1-HONEST-IID-S211`:
     - Runtime: 98.57s | Peak RSS: 1,041.1 MB | Available RAM: 1,375.7 MB
     - Test Accuracy: **85.58%** | Sensitivity: **97.44%** | Specificity: **65.81%** | Macro F1: **83.40%** | Test Loss: 0.3957
     - Confusion Matrix: TP=380, FN=10, TN=154, FP=80
     - Blockchain Completion Block: **#131** (Lifecycle Blocks #110–131)
  3. `EXP-P7-MULTI_KRUM_F2-HONEST-IID-S211`:
     - Runtime: 97.62s | Peak RSS: 1,045.8 MB | Available RAM: 1,373.4 MB
     - Test Accuracy: **85.74%** | Sensitivity: **96.15%** | Specificity: **68.38%** | Macro F1: **83.82%** | Test Loss: 0.4072
     - Confusion Matrix: TP=375, FN=15, TN=160, FP=74
     - Blockchain Completion Block: **#153** (Lifecycle Blocks #132–153)
  4. `EXP-P7-TRI_STATE_MONITOR-HONEST-IID-S211`:
     - Runtime: 96.75s | Peak RSS: 1,049.2 MB | Available RAM: 1,402.5 MB
     - Test Accuracy: **85.74%** | Sensitivity: **96.41%** | Specificity: **67.95%** | Macro F1: **83.78%** | Test Loss: 0.4299
     - Confusion Matrix: TP=376, FN=14, TN=159, FP=75
     - Blockchain Completion Block: **#175** (Lifecycle Blocks #154–175)
  5. `EXP-P7-TRI_STATE_DISCOUNT_050-HONEST-IID-S211`:
     - Runtime: 97.70s | Peak RSS: 1,049.2 MB | Available RAM: 1,444.8 MB
     - Test Accuracy: **85.74%** | Sensitivity: **97.44%** | Specificity: **66.24%** | Macro F1: **83.61%** | Test Loss: 0.4285
     - Confusion Matrix: TP=380, FN=10, TN=155, FP=79
     - Blockchain Completion Block: **#197** (Lifecycle Blocks #176–197)
- **Cumulative Matrix Progress**: **8 / 351 simulations completed (2.28%)**. 343 remaining.
- **Audit Ledger Verification**:
  - Total verified blocks: **198** (Blocks #0 through #197).
  - Status: **VALID (0 errors, 0 warnings)**.
  - All disk artifact SHA-256 hashes (`.json` evaluation, `.pt` checkpoints, `.jsonl` features) match ledger records bit-for-bit.
- **Execution Status**: SUCCEEDED

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
