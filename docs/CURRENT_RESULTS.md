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

## 4. Phase 4: Non-IID Hospital Data Heterogeneity Study (`EXP-MED-003` to `EXP-MED-005`)

- **Execution Script**: [`experiments/train_federated_non_iid.py`](file:///home/suvranil/project/experiments/train_federated_non_iid.py)
- **Log Reports**:
  - Alpha = 1.0: [`results/logs/federated_non_iid_alpha10_results.json`](file:///home/suvranil/project/results/logs/federated_non_iid_alpha10_results.json)
  - Alpha = 0.5: [`results/logs/federated_non_iid_alpha05_results.json`](file:///home/suvranil/project/results/logs/federated_non_iid_alpha05_results.json)
  - Alpha = 0.1: [`results/logs/federated_non_iid_alpha01_results.json`](file:///home/suvranil/project/results/logs/federated_non_iid_alpha01_results.json)
  - Comparative Summary: [`results/logs/phase4_heterogeneity_comparison.json`](file:///home/suvranil/project/results/logs/phase4_heterogeneity_comparison.json)
- **Experimental Protocol**: Controlled-capacity population-calibrated Beta/Dirichlet case-mix partitioning across 10 hospitals (8 with 471, 2 with 470 samples; sum = 4,708). Exact parity with Phase 3: Adam ($\text{lr}=0.001$, $\text{wd}=10^{-4}$), 2 local epochs/round, batch size 32, 20 rounds, 100% client participation, sample-weighted FedAvg including BatchNorm buffers. Identical initial parameter state (`0c411b83c2f0acef58beb5b9a968030d877e079eb2356eb3748903f8ae9bbc83`) verified by SHA-256 hash.

### 4.1. Heterogeneity Quantification & Client Distributions
| Condition | Total Concentration ($\alpha$) | Parameter Vector $\boldsymbol{\alpha} = \alpha \cdot \bar{\mathbf{p}}$ | Rejection Attempt | Normal Ratio Mean | Normal Ratio Std | Normal Ratio Variance | Mean Abs Target Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **IID Baseline** | $\infty$ | N/A | 0 | 25.79% | 0.05% | 0.0000 | 0.00% |
| **Alpha = 1.0** | 1.0 | $[0.2579, 0.7421]$ | 21 | 25.78% | 26.62% | 0.0709 | 0.63% |
| **Alpha = 0.5** | 0.5 | $[0.1289, 0.3711]$ | 6 | 25.78% | 37.18% | 0.1382 | 0.81% |
| **Alpha = 0.1** | 0.1 | $[0.0258, 0.0742]$ | 132 | 25.80% | 38.49% | 0.1482 | 0.43% |

### 4.1. Heterogeneity Quantification & Client Distributions
> **Important Methodological Distinction**: The 10-seed characterization below is strictly an empirical evaluation of the **partitioner's mathematical dispersion properties** across random seeds. The subsequent 20-round federated training runs (`EXP-MED-003`, `EXP-MED-004`, `EXP-MED-005`) are **single-seed (seed 42) training simulations** evaluated on the official test set, not 10-seed averaged model training evaluations.

| Condition | Total Concentration ($\alpha$) | Parameter Vector $\boldsymbol{\alpha} = \alpha \cdot \bar{\mathbf{p}}$ | Rejection Attempt | Normal Ratio Mean | Normal Ratio Std | Normal Ratio Variance | Mean Abs Target Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **IID Baseline** | $\infty$ | N/A | 0 | 25.79% | 0.05% | 0.0000 | 0.00% |
| **Alpha = 1.0** | 1.0 | $[0.2579, 0.7421]$ | 21 | 25.78% | 26.62% | 0.0709 | 0.63% |
| **Alpha = 0.5** | 0.5 | $[0.1289, 0.3711]$ | 6 | 25.78% | 37.18% | 0.1382 | 0.81% |
| **Alpha = 0.1** | 0.1 | $[0.0258, 0.0742]$ | 132 | 25.80% | 38.49% | 0.1482 | 0.43% |

### 4.2. Diagnostic Performance Comparison (Untouched Test Split, N=624)
| Metric | Centralized Baseline | FedAvg IID (Phase 3) | Non-IID $\alpha = 1.0$ | Non-IID $\alpha = 0.5$ | Non-IID $\alpha = 0.1$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Experiment ID** | `EXP-MED-001` | `EXP-MED-002` | `EXP-MED-003` | `EXP-MED-004` | `EXP-MED-005` |
| **Selection Criterion** | Val Loss (Ep 14) | Val Acc (Rd 17) | Val Acc (Rd 15) | Val Acc (Rd 18) | Val Acc (Rd 12) |
| **Best Val Acc (%)** | 97.90% | 96.18% | 95.99% | 95.99% | 89.89% |
| **Test Loss** | 0.9172 | 0.5306 | 0.7390 | 0.6930 | 0.6326 |
| **Test Accuracy (%)** | 81.25% (507/624) | **85.74%** (535/624) | 79.81% (498/624) | 80.45% (502/624) | 85.10% (531/624) |
| **Pneumonia Sensitivity** | **99.74%** (389/390) | 97.95% (382/390) | 98.97% (386/390) | 98.46% (384/390) | 91.79% (358/390) |
| **Pneumonia Precision** | 77.03% (389/505) | 82.51% (382/463) | 75.98% (386/508) | 76.80% (384/500) | **85.44%** (358/419) |
| **Pneumonia F1-Score** | 0.8693 | **0.8957** | 0.8597 | 0.8629 | 0.8850 |
| **Normal Specificity** | 50.43% (118/234) | 65.38% (153/234) | 47.86% (112/234) | 50.43% (118/234) | **73.93%** (173/234) |
| **Normal Precision** | **99.16%** (118/119) | 95.03% (153/161) | 96.55% (112/116) | 95.16% (118/124) | 84.39% (173/205) |
| **Normal F1-Score** | 0.6686 | 0.7747 | 0.6400 | 0.6592 | **0.7882** |
| **Macro F1-Score** | 0.7689 | **0.8352** | 0.7498 | 0.7611 | **0.8366** |

### 4.3. Test Confusion Matrices
| Experimental Condition | True Normal (TN) | False Positive (FP) | False Negative (FN) | True Positive (TP) | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized Baseline** | 118 | 116 | 1 | 389 | 624 |
| **FedAvg IID (Phase 3)** | 153 | 81 | 8 | 382 | 624 |
| **Non-IID $\alpha = 1.0$** | 112 | 122 | 4 | 386 | 624 |
| **Non-IID $\alpha = 0.5$** | 118 | 116 | 6 | 384 | 624 |
| **Non-IID $\alpha = 0.1$** | **173** | **61** | 32 | 358 | 624 |

### 4.4. Scientific Observations on Non-IID Dynamics
1. **Absence of Simple Monotonic Degradation**:
   - The empirical results do **not** exhibit monotonic performance degradation with decreasing $\alpha$. Test accuracy moves from 85.74% (IID) $\to$ 79.81% ($\alpha=1.0$) $\to$ 80.45% ($\alpha=0.5$) $\to$ 85.10% ($\alpha=0.1$).
   - Rather than simple scalar degradation, varying $\alpha$ induces **distinct clinical trade-offs and optimization instability**:
     * At moderate skew ($\alpha=1.0, 0.5$), FedAvg exhibits high sensitivity (~98.5–99.0%) but lower specificity (~48–50%), reflecting majority-class dominance.
     * At extreme skew ($\alpha=0.1$), the optimization landscape experiences acute representation oscillations. The Round 12 validation-selected checkpoint captured a balanced decision boundary that achieved the highest specificity in the project (73.93%, TN=173, reducing false positives to 61), but at the cost of lowered sensitivity (91.79%, 32 missed pneumonia cases).
2. **Client Drift & Representation Collapse Under Extreme Skew ($\alpha = 0.1$)**:
   - In $\alpha = 0.1$, five hospitals have 0 normal scans (100% pneumonia), while Hospital 9 has 96.17% normal scans.
   - This acute gradient disparity caused global validation accuracy to oscillate heavily across rounds (alternating between 74.24% majority-class collapse, 27.67% over-correction, and 89.89% balanced peaks).
   - While the validation-based checkpointing mechanism successfully isolated the Round 12 peak (89.89% val acc), the subsequent rounds (Rounds 13–20) collapsed entirely into predicting 100% pneumonia (val acc = 74.24%), empirically demonstrating the severe vulnerability of unregularized FedAvg to representation collapse under acute case-mix divergence.
3. **Implications for Malicious Attack Detection (Phase 5 & 6)**:
   - Because benign case-mix skew at $\alpha = 0.1$ naturally produces large local gradient divergence, a Byzantine detector that relies solely on $L_2$ distance or naive update deviation would mistakenly flag honest specialized hospitals (like Hospital 9 or Hospital 6) as adversaries.
   - This proves the critical architectural need for the multi-factor, heterogeneity-aware detector planned for Phase 6.

### 4.5. Generated Phase 4 Visual Artifacts
- **Case-Mix Distributions**:
  - [`results/figures/hospital_distribution_alpha10.png`](file:///home/suvranil/project/results/figures/hospital_distribution_alpha10.png)
  - [`results/figures/hospital_distribution_alpha05.png`](file:///home/suvranil/project/results/figures/hospital_distribution_alpha05.png)
  - [`results/figures/hospital_distribution_alpha01.png`](file:///home/suvranil/project/results/figures/hospital_distribution_alpha01.png)
- **Training Curves**:
  - [`results/figures/federated_non_iid_alpha10_training_loss.png`](file:///home/suvranil/project/results/figures/federated_non_iid_alpha10_training_loss.png) & [`training_accuracy.png`](file:///home/suvranil/project/results/figures/federated_non_iid_alpha10_training_accuracy.png)
  - [`results/figures/federated_non_iid_alpha05_training_loss.png`](file:///home/suvranil/project/results/figures/federated_non_iid_alpha05_training_loss.png) & [`training_accuracy.png`](file:///home/suvranil/project/results/figures/federated_non_iid_alpha05_training_accuracy.png)
  - [`results/figures/federated_non_iid_alpha01_training_loss.png`](file:///home/suvranil/project/results/figures/federated_non_iid_alpha01_training_loss.png) & [`training_accuracy.png`](file:///home/suvranil/project/results/figures/federated_non_iid_alpha01_training_accuracy.png)
- **Confusion Matrices**:
  - [`results/figures/federated_non_iid_alpha10_confusion_matrix.png`](file:///home/suvranil/project/results/figures/federated_non_iid_alpha10_confusion_matrix.png)
  - [`results/figures/federated_non_iid_alpha05_confusion_matrix.png`](file:///home/suvranil/project/results/figures/federated_non_iid_alpha05_confusion_matrix.png)
  - [`results/figures/federated_non_iid_alpha01_confusion_matrix.png`](file:///home/suvranil/project/results/figures/federated_non_iid_alpha01_confusion_matrix.png)
- **Comprehensive Multi-Condition Comparison**:
  - [`results/figures/iid_vs_non_iid_heterogeneity_comparison.png`](file:///home/suvranil/project/results/figures/iid_vs_non_iid_heterogeneity_comparison.png)

---

---

## 5. Phase 5: Adversarial Attacks Simulation & Matched Controls (`EXP-MED-006` to `EXP-MED-014`)

- **Execution Runner**: [`experiments/train_federated_attack.py`](file:///home/suvranil/project/experiments/train_federated_attack.py) & [`experiments/run_phase5_suite.py`](file:///home/suvranil/project/experiments/run_phase5_suite.py)
- **Summary Log**: [`results/logs/phase5_attack_comparison.json`](file:///home/suvranil/project/results/logs/phase5_attack_comparison.json)
- **Summary Figure**: [`results/figures/phase5_attack_comparison.png`](file:///home/suvranil/project/results/figures/phase5_attack_comparison.png)

### 5.1. Comprehensive Phase 5 Empirical Results (Untouched Test Split N=624)
| Experiment ID | Scenario | Attack Type | Malicious Nodes | Best Val Acc (%) | Clean Test Acc (%) | Sensitivity | Specificity | False Negatives (FN) | Attack-Induced Pneu Misclass (%) | Triggered ASR (%) | Delta ASR Increase (%) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`EXP-MED-006-CTRL-IID`** | IID | None (Honest) | $M=0$ | 96.37% (Rd 18) | 84.62% | 98.21% | 61.97% | 7 | 1.79% | 0.51% (2/390) | Baseline Control |
| **`EXP-MED-007-CTRL-NONIID-A05`** | Non-IID | None (Honest) | $M=0$ | 95.61% (Rd 20) | 78.37% | 98.97% | 44.02% | 4 | 1.03% | 0.26% (1/390) | Baseline Control |
| **`EXP-MED-008-ATK-LF-IID-M1`** | IID | Label Flipping | $M=1$ (H0) | 94.66% (Rd 14) | 88.78% | 96.41% | 76.07% | 14 | 3.59% | N/A | N/A ($\Delta \text{FN} = \mathbf{+7}$) |
| **`EXP-MED-009-ATK-LF-IID-M2`** | IID | Label Flipping | $M=2$ (H0,H1) | 86.64% (Rd 2) | 86.54% | 84.62% | 89.74% | 60 | 15.38% | N/A | N/A ($\Delta \text{FN} = \mathbf{+53}$) |
| **`EXP-MED-010-ATK-LF-NONIID-M1`** | Non-IID | Label Flipping | $M=1$ (H0) | 95.99% (Rd 20) | 85.90% | 97.18% | 67.09% | 11 | 2.82% | N/A | N/A ($\Delta \text{FN} = \mathbf{+7}$) |
| **`EXP-MED-011-ATK-MP-INV-IID-M1`** | IID | Update Invert ($\gamma=-1$) | $M=1$ (H0) | 95.80% (Rd 18) | 84.94% | 97.95% | 63.25% | 8 | 2.05% | N/A | N/A ($\Delta \text{FN} = \mathbf{+1}$) |
| **`EXP-MED-012-ATK-MP-SCALE-IID-M1`**| IID | Update Scale ($\gamma=+10$)| $M=1$ (H0) | 96.56% (Rd 13) | 86.70% | 97.69% | 68.38% | 9 | 2.31% | N/A | N/A ($\Delta \text{FN} = \mathbf{+2}$) |
| **`EXP-MED-013-ATK-BD-IID-M1`** | IID | Backdoor ($\rho=0.50$) | $M=1$ (H0) | 95.99% (Rd 16) | 86.06% | 97.44% | 67.09% | 10 | 2.56% | 7.44% (29/390) | $\mathbf{+6.93\%}$ (14.6x jump) |
| **`EXP-MED-014-ATK-BD-NONIID-M1`** | Non-IID | Backdoor ($\rho=0.50$) | $M=1$ (H0) | 95.99% (Rd 13) | 84.46% | 97.95% | 61.97% | 8 | 2.05% | 8.46% (33/390) | $\mathbf{+8.20\%}$ (32.5x jump) |

### 5.2. Test Confusion Matrices (Untouched Test Split N=624)
| Experiment ID | True Normal (TN) | False Positive (FP) | False Negative (FN) | True Positive (TP) | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`EXP-MED-006-CTRL-IID`** | 145 | 89 | 7 | 383 | 624 |
| **`EXP-MED-007-CTRL-NONIID-A05`** | 103 | 131 | 4 | 386 | 624 |
| **`EXP-MED-008-ATK-LF-IID-M1`** | 178 | 56 | 14 | 376 | 624 |
| **`EXP-MED-009-ATK-LF-IID-M2`** | 210 | 24 | 60 | 330 | 624 |
| **`EXP-MED-010-ATK-LF-NONIID-M1`** | 157 | 77 | 11 | 379 | 624 |
| **`EXP-MED-011-ATK-MP-INV-IID-M1`** | 148 | 86 | 8 | 382 | 624 |
| **`EXP-MED-012-ATK-MP-SCALE-IID-M1`** | 160 | 74 | 9 | 381 | 624 |
| **`EXP-MED-013-ATK-BD-IID-M1`** | 157 | 77 | 10 | 380 | 624 |
| **`EXP-MED-014-ATK-BD-NONIID-M1`** | 145 | 89 | 8 | 382 | 624 |

---

## 6. Automated Quality & Regression Test Suite

- **Test Framework**: `pytest 9.1.1`
- **Total Tests**: **50 tests**
- **Test Results**: **50 passed (100%) in 4.53 seconds**
  - `tests/test_detection.py` (5 tests): Verified SmallCNN parameter layer accounting (Conv1: 384, Conv2: 18,624, Dense: 401,536, Head: 258, Total: 420,802), Modified Z-score resistance, symmetrical clinical trade-off protection, tri-state decision consistency, and stratified FPR calculations.
  - `tests/test_attacks.py` (11 tests): Label-flipping selectivity, base dataset immutability, update inversion and scaling math, BatchNorm buffer invariance, PoisonedHospitalClient execution, trigger geometry, BackdoorSubset labeling, triggered test evaluation, and initial model hash verification.
  - `tests/test_data_loading.py` (3 tests): Splits exist, batch shapes (1x28x28), deterministic shuffling.
  - `tests/test_model.py` (8 tests): Model architecture, parameters (420,802), output shapes `[1, 4, 16, 32, 2]`, forward validity, loss computation, gradient backpropagation step.
  - `tests/test_training.py` (3 tests): Split separation, loader batch shapes, medical metrics functions.
  - `tests/test_federated.py` (7 tests): IID partition integrity, stratification balance, determinism, parameter serialization invariance, client isolation, fit execution, server validation evaluation.
  - `tests/test_non_iid.py` (13 tests): Controlled client capacity constraints ($470 \le |S_i| \le 471$), full coverage and mutual exclusivity, exact class conservation, determinism, structural dataset isolation, single-class client fit execution, and 10-seed aggregate statistical characterization.

---

## 7. Phase 6: Multi-Factor Non-IID-Aware Malicious Hospital Detection

- **Master Evaluation Script**: [`experiments/run_phase6_evaluation.py`](file:///home/suvranil/project/experiments/run_phase6_evaluation.py)
- **Instrumentation Runner**: [`experiments/run_phase6_instrumentation.py`](file:///home/suvranil/project/experiments/run_phase6_instrumentation.py)
- **Replay Verification Report**: [`results/phase6_detector/verification/verification_report.json`](file:///home/suvranil/project/results/phase6_detector/verification/verification_report.json)
- **Evaluation Summary Report**: [`results/phase6_detector/evaluation/detection_results.json`](file:///home/suvranil/project/results/phase6_detector/evaluation/detection_results.json)
- **Publication Figure**: [`results/phase6_detector/figures/phase6_detector_performance.png`](file:///home/suvranil/project/results/phase6_detector/figures/phase6_detector_performance.png)

### 7.1. Benchmark Evaluation Metrics: Dual Reporting (Calibration-Excluded vs. In-Sample Reference)

Thresholds were calibrated using two held-out controls (`EXP-MED-006-CTRL-IID` and `EXP-MED-007-CTRL-NONIID-A05`, 400 client-round records). We report both the strict out-of-sample calibration-excluded evaluation (9 simulations, 1,800 records: 1,640 honest, 160 malicious) and the in-sample reference (11 simulations, 2,200 records: 2,040 honest, 160 malicious).

| Evaluation Metric | Calibration-Excluded (Out-of-Sample, N=1,800) | | In-Sample Reference (Full Cohort, N=2,200) | |
| :--- | :---: | :---: | :---: | :---: |
| | **Binary Malicious** | **Flagged (Suspicious/Malicious)** | **Binary Malicious** | **Flagged (Suspicious/Malicious)** |
| **True Positive Rate (Recall)** | **13.13%** (21 / 160) | **96.25%** (154 / 160) | **13.13%** (21 / 160) | **96.25%** (154 / 160) |
| **False Positive Rate (FPR)** | **0.00%** (0 / 1,640) | **10.00%** (164 / 1,640) | **0.00%** (0 / 2,040) | **9.95%** (203 / 2,040) |
| **Precision** | **100.00%** (21 / 21) | **48.43%** (154 / 318) | **100.00%** (21 / 21) | **43.14%** (154 / 357) |
| **F1-Score** | **23.20%** | **64.44%** | **23.20%** | **59.57%** |
| **AUROC** | — | **0.9814** | — | **0.9761** |
| **AUPRC** | — | **0.8801** | — | **0.8026** |

#### Tri-State Confusion Matrices (Actual vs. Predicted)

**Calibration-Excluded (N=1,800)**:
- Predicted `NORMAL`: Honest = 1,476 (90.00%), Malicious = 6 (3.75%)
- Predicted `SUSPICIOUS`: Honest = 164 (10.00%), Malicious = 133 (83.12%)
- Predicted `MALICIOUS`: Honest = 0 (**0.00%**), Malicious = 21 (13.13%)

**In-Sample Reference (N=2,200)**:
- Predicted `NORMAL`: Honest = 1,837 (90.05%), Malicious = 6 (3.75%)
- Predicted `SUSPICIOUS`: Honest = 203 (9.95%), Malicious = 133 (83.12%)
- Predicted `MALICIOUS`: Honest = 0 (**0.00%**), Malicious = 21 (13.13%)

### 7.2. Stratified False-Positive Rate Across Heterogeneity (Calibration-Excluded)

| Heterogeneity Level | Honest Client-Rounds | False Malicious Count | False Malicious Rate (%) | Suspicious Monitored Count | Suspicious Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Honest IID** | 880 | **0** | **0.00%** | 1 | 0.11% |
| **Honest Non-IID $\alpha = 1.0$** | 200 | **0** | **0.00%** | 38 | 19.00% |
| **Honest Non-IID $\alpha = 0.5$** | 360 | **0** | **0.00%** | 79 | 21.94% |
| **Honest Non-IID $\alpha = 0.1$** | 200 | **0** | **0.00%** | 46 | 23.00% |

### 7.3. Independent-Seed Honest Evaluation Suite (Seeds 43, 44, 45; 2,400 Observations)

Conducted 12 independent honest simulations across three distinct partition/training seeds (43, 44, 45) while holding the initial model parameter hash constant (`0c411b83c2f0acef58beb5b9a968030d877e079eb2356eb3748903f8ae9bbc83`):

| Heterogeneity Condition | Seeds Evaluated | Honest Client-Rounds | False Malicious Count | False Malicious Rate (%) | Suspicious Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Honest IID** | 43, 44, 45 | 600 | **0** | **0.00%** | 0.50% (3 / 600) |
| **Honest Non-IID $\alpha = 1.0$** | 43, 44, 45 | 600 | **0** | **0.00%** | 18.83% (113 / 600) |
| **Honest Non-IID $\alpha = 0.5$** | 43, 44, 45 | 600 | **0** | **0.00%** | 28.00% (168 / 600) |
| **Honest Non-IID $\alpha = 0.1$** | 43, 44, 45 | 600 | **1** | **0.17%** | 36.33% (218 / 600) |
| **Aggregate Multi-Seed Total** | — | **2,400** | **1** | **0.04%** | **20.92%** (502 / 2,400) |

### 7.4. Forensic Audit of Honest Non-IID Observations (3,200 Observations Audited)

Across all 16 honest simulations (3,200 client-round observations):
- **False Malicious Rate**: **0.03%** (1 / 3,200 observations).
- **Negative Cosine Events**: 199 events occurred among honest participants (6.22% of honest observations).
  - Frequency scaled with clinical heterogeneity: IID = 2.43% (34/1400), $\alpha=1.0$ = 5.50% (33/600), $\alpha=0.5$ = 8.50% (51/600), $\alpha=0.1$ = 13.50% (81/600).
  - Negative cosine in honest Non-IID is driven by legitimate clinical specialization: minority hospitals (e.g., 467 Normal / 3 Pneumonia) compute gradient vectors opposing the majority Pneumonia cohort (7 hospitals with 99% Pneumonia cases). Cosines drop as low as $-0.8411$, yet these hospitals reduce Normal validation loss by $\Delta \mathcal{L}_{\text{normal}} = -6.1091$ to $-8.4208$.
  - **Symmetrical Clinical Trade-Off Protection**: **199 / 199 (100.0%)** of negative cosine events were protected from uncompensated loss penalties.
  - Zero honest negative cosine events were classified as MALICIOUS; all 199 were assigned to `SUSPICIOUS` (monitored with reduced aggregation weight).
- **Sole False Malicious Event**:
  - `EXP-MED-HONEST-A01-S45`, Hospital 6, Round 10 ($\alpha=0.1$, class ratio: 469 Normal / 2 Pneumonia).
  - Driven by a statistical low-MAD artifact in the dense projection layer ($Z = 67.75$) where 7 other clients had near-identical update norms ($\approx 3.4$, causing cohort MAD $\approx 0.05$). The client's actual norm was $6.4$ ($1.87\times$ median), which co-occurred with transient positive validation loss ($\Delta \mathcal{L}_{\text{total}} = +0.0507$).

### 7.5. Per-Attack Detection Breakdown & Latency (Calibration-Excluded)

| Attack Experiment | Malicious Rounds | Flagged Malicious | Flagged Suspicious | Total Flagged | Detection Recall (%) | Earliest Detection Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Label Flipping IID ($M=1$)** | 20 | 0 | 20 | 20 | **100.0%** | Round 1 (Suspicious) |
| **Label Flipping IID ($M=2$)** | 40 | 0 | 38 | 38 | **95.0%** | Round 2 (Suspicious) |
| **Label Flipping Non-IID ($M=1$)** | 20 | 0 | 20 | 20 | **100.0%** | Round 1 (Suspicious) |
| **Model Inversion ($\gamma=-1.0$)** | 20 | 1 | 19 | 20 | **100.0%** | Round 1 (Suspicious), Round 10 (Malicious) |
| **Model Scaling ($\gamma=+10.0$)** | 20 | 20 | 0 | 20 | **100.0%** | Round 1 (Malicious) |
| **Backdoor Trigger IID ($\rho=0.50$)** | 20 | 0 | 19 | 19 | **95.0%** | Round 2 (Suspicious) |
| **Backdoor Trigger Non-IID ($\rho=0.50$)** | 20 | 0 | 17 | 17 | **85.0%** | Round 2 (Suspicious) |

### 7.6. Instrumentation Parity & Replay Reproducibility Audit

1. **Aggregation Mathematical Equivalence**:
   - `InstrumentedFedAvg` was tested against standard Flower `FedAvg` on identical client `FitRes` objects across all 12 trainable parameter tensors and 6 BatchNorm buffers ([`tests/test_instrumentation_parity.py`](file:///home/suvranil/project/tests/test_instrumentation_parity.py)).
   - **Result**: Exactly **$0.000$ maximum absolute difference** across all 18 parameter arrays. Mathematical equivalence is $100\%$ verified.
2. **Training Replay Numerical Divergence Under Ray**:
   - Parameter deviations observed between historical Phase 5 checkpoints and replayed runs ($0.642 - 2.590$) are caused by Flower's `VirtualClientEngineActorPool` multi-core Ray actor scheduling.
   - Multithreaded CPU floating-point operations in PyTorch are non-associative; variations in thread dispatch order across 12 CPU cores introduce $\sim 10^{-7}$ floating-point rounding differences per batch, which compound across 20 rounds of nonlinear SGD.
   - Replay logs serve as self-consistent, fully-instrumented execution streams rather than bit-for-bit historical mirrors.

### 7.7. Phase 6 Frozen Blind Evaluation Suite (Fresh Untouched Seeds 111, 112, 113)

Following the seed integrity correction (segregating prior seed-101 runs to `results/phase6_preflight_exploratory_seed101/`), a full blind evaluation was executed across 33 fresh simulations (6,600 observations) using seeds `111, 112, 113`. The detector was locked as read-only (`federated_learning/detection/frozen_v1/`) with zero re-calibration.

- **Two-Tier Scorecard ($N=6,600$ Observations)**:
  - **Hard Malicious Level**: Recall = **12.92%** (62 / 480), **FPR = 0.00%** (**0 / 6,120**), Precision = **100.00%**, F1 = 0.2288.
  - **Overall Flagged Level**: Threat Recall = **93.75%** (450 / 480), Honest Flagged Rate = **13.15%** (805 / 6,120), Precision = 35.86%, F1 = 0.5188.
- **Tri-State Matrix**:
  - NORMAL: Honest = 5,315 (86.85%), Malicious = 30 (6.25%)
  - SUSPICIOUS: Honest = 805 (13.15%), Malicious = 388 (80.83%)
  - MALICIOUS: Honest = 0 (0.00%), Malicious = 62 (12.92%)
- **Honest Flagged Rate Across Heterogeneity Regimes**:
  - Honest IID ($N=3,240$): Hard False Malicious = **0.00%** (0), Suspicious = **0.43%** (14).
  - Honest Non-IID $\alpha=1.0$ ($N=600$): Hard False Malicious = **0.00%** (0), Suspicious = **17.17%** (103).
  - Honest Non-IID $\alpha=0.5$ ($N=1,680$): Hard False Malicious = **0.00%** (0), Suspicious = **27.80%** (467).
  - Honest Non-IID $\alpha=0.1$ ($N=600$): Hard False Malicious = **0.00%** (0), Suspicious = **36.83%** (221).
- **Per-Attack Detection Recall Across Seeds 111, 112, 113**:
  - Model Scaling ($\gamma=+10.0$): **100.00% Hard Malicious** (60/60) at Round 1.
  - Model Inversion ($\gamma=-1.0$): **100.00% Flagged** (60/60; 2 hard malicious, 58 suspicious).
  - Label Flipping IID $M=1$: **98.33% Flagged** (59/60).
  - Label Flipping IID $M=2$: **100.00% Flagged** (120/120).
  - Label Flipping Non-IID $\alpha=0.5$ $M=1$: **90.00% Flagged** (54/60).
  - Backdoor IID $M=1$: **91.67% Flagged** (55/60); Mean Test ASR: 7.01%.
  - Backdoor Non-IID $\alpha=0.5$ $M=1$: **70.00% Flagged** (42/60); Mean Test ASR: 4.87%.
- Full Report: [`docs/PHASE6_BLIND_EVALUATION_REPORT.md`](file:///home/suvranil/project/docs/PHASE6_BLIND_EVALUATION_REPORT.md).

---

## 8. Phase 7: Byzantine-Robust Aggregation & Blockchain Audit Foundation

### 8.1. Memory-Safe Sequential Execution Baseline Measurements (Seed 211, Honest IID)
- **Runner Script**: [`experiments/run_phase7_memory_safe.py`](file:///home/suvranil/project/experiments/run_phase7_memory_safe.py)
- **Execution Strategy**: Strictly sequential in-process execution ($O(1)$ memory, zero Ray multi-processing overhead, per-round tensor and client destruction with `gc.collect()`).
- **Memory Profile**: System total RAM: 6,749 MB, base usage: 5.2–5.4 GB, peak process RSS: 907–1097 MB, available system memory maintained at $>1.4\text{ GB}$ throughout all 20 rounds.
- **Circuit Breaker**: Enforced automatic pause if available system RAM falls below 250 MB.

#### Diagnostic Performance on Untouched Test Split ($N=624$ Chest X-Rays, Seed 211)
| Experiment Identifier | Strategy / Aggregator | Test Accuracy (%) | Pneumonia Sensitivity (%) | Normal Specificity (%) | Macro F1 (%) | Test Loss | Confusion Matrix `[TN, FP, FN, TP]` | Blockchain Block | Runtime (s) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `EXP-P7-FEDAVG-HONEST-IID-S211` | Standard FedAvg | 85.74% | 96.41% (376/390) | 67.95% (159/234) | 83.78% | 0.4299 | `[159, 75, 14, 376]` | #43 | 87.3s |
| `EXP-P7-COORDINATE_MEDIAN-HONEST-IID-S211` | Coordinate Median | 86.54% | 96.67% (377/390) | 69.66% (163/234) | 84.74% | 0.3815 | `[163, 71, 13, 377]` | #65 | 89.5s |
| `EXP-P7-TRIMMED_MEAN_BETA1-HONEST-IID-S211` | Trimmed Mean ($\beta=1$) | 86.22% | 96.67% (377/390) | 68.80% (161/234) | 84.34% | 0.4486 | `[161, 73, 13, 377]` | #87 | 87.1s |

### 8.2. Blockchain Audit Ledger Integrity & Artifact Fingerprints
- **Ledger Path**: [`results/blockchain_audit/audit_ledger.jsonl`](file:///home/suvranil/project/results/blockchain_audit/audit_ledger.jsonl)
- **Verifier Report**: Standalone verification via `blockchain_audit.verification` confirms:
  - **Total Blocks Verified**: 88 blocks (1 Genesis, 3 Experiment Inits, 60 Communication Rounds, 3 Final Completions, 21 Lifecycle Events).
  - **Genesis Block Integrity**: Verified index 0 and predefined previous hash (`0000...0000`).
  - **Backwards Hash Linkage**: 100% continuous backwards SHA-256 hash chaining ($\mathcal{H}_i = \text{SHA-256}(\text{CanonicalJSON}(\text{Block}_i))$).
  - **Disk Artifact Hashes**: Verified against physical `.pt` checkpoints, `.jsonl` telemetry, and `.json` metrics on disk with zero checksum mismatches.
  - **Ledger Verification Status**: **VALID (100% Cryptographic Match)**.

---

## 9. Summary of Archived CIFAR-10 Results
The previous generic proof-of-concept remains archived in [`results/archive/cifar10/`](file:///home/suvranil/project/results/archive/cifar10/README.md):
- **Dataset**: CIFAR-10 (60,000 images, 10 classes, 32x32 RGB).
- **Model**: SmallCNN (189,130 parameters).
- **Test Accuracy**: 76.94% (7,694 / 10,000 correct).
- **Training Time**: 100.97 seconds on RTX 3050 GPU.
- **Status**: Archived; superseded by PneumoniaMNIST for the hospital project.



