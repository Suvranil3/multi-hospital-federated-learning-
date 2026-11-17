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

## 8. Phase 7: Byzantine-Robust Aggregation & Consensus Under Non-IID Skew — COMPLETED & AUDITED

- **Status**: **COMPLETED & AUDITED (November 17, 2026)**
- **Scope & Balancing**: Evaluated across a 72-cell primary research matrix (4 aggregation strategies × 9 experimental regimes × 2 seeds: 211, 212) plus 24 preserved supplementary historical experiments (96 unique simulations total; 97 physical launches including 1 documented crash recovery in Batch 2).
- **Execution Methodology**: Strictly sequential, in-process CPU execution ([`experiments/run_phase7_memory_safe.py`](file:///home/suvranil/project/experiments/run_phase7_memory_safe.py)) with per-round tensor/client garbage collection, memory-safe execution, and a 250 MB RAM circuit breaker.
- **Untouched Benchmark Test Split**: $N = 624$ images (234 Normal [Class 0, 37.5%], 390 Pneumonia [Class 1, 62.5%]).
- **Statistical Disclosure**: All primary scorecard figures report exploratory benchmark means across $N=2$ distinct seeds (Seeds 211 and 212). They are presented as descriptive empirical trends and are not claimed to be hypothesis-tested statistical significance.
- **Experimental Setup**: Evaluated on simulated client nodes partitioned using synthetic Dirichlet distributions ($\alpha \in \{1.0, 0.5, 0.1\}$) over PneumoniaMNIST ($28 \times 28$ grayscale images); this setup serves as an algorithmic stress-test and does not claim to represent real clinical hospital imaging or multi-center institutional heterogeneity.

### 8.1. Artifact Inventory & Cryptographic Ledger Verification
- **Primary Manifest**: [`results/phase7_robust_aggregation/PHASE7_REDUCED_EXPERIMENT_MATRIX_96.json`](file:///home/suvranil/project/results/phase7_robust_aggregation/PHASE7_REDUCED_EXPERIMENT_MATRIX_96.json) (72 primary experiment specifications).
- **Primary Execution Registry**: [`results/phase7_robust_aggregation/execution_registry_reduced_96.json`](file:///home/suvranil/project/results/phase7_robust_aggregation/execution_registry_reduced_96.json) (72 completed records, bit-for-bit verified against evaluation JSONs).
- **Historical Execution Registry (Preserved)**: [`results/phase7_robust_aggregation/execution_registry.json`](file:///home/suvranil/project/results/phase7_robust_aggregation/execution_registry.json) (43 records, preserving 24 supplementary experiments: 5 Coordinate Median, 5 Trimmed Mean $\beta=1$, 5 Trimmed Mean $\beta=2$, 5 Multi-Krum $f=2$, 4 Tri-State Discount $\lambda=0.25$).
- **Checkpoints**: Exactly 144 `.pt` model files (72 `best_*.pt`, 72 `final_*.pt`, 1.7 MB each) in [`results/phase7_robust_aggregation/checkpoints/`](file:///home/suvranil/project/results/phase7_robust_aggregation/checkpoints/).
- **Evaluation Reports**: Exactly 72 `*_results.json` files in [`results/phase7_robust_aggregation/evaluation/`](file:///home/suvranil/project/results/phase7_robust_aggregation/evaluation/).
- **Telemetry Logs**: Exactly 72 `*_features.jsonl` files in [`results/phase7_robust_aggregation/instrumentation/`](file:///home/suvranil/project/results/phase7_robust_aggregation/instrumentation/).
- **Batch 2 Recovery Audit**: [`docs/PHASE7_BATCH2_RECOVERY_AUDIT.md`](file:///home/suvranil/project/docs/PHASE7_BATCH2_RECOVERY_AUDIT.md).
- **Blockchain Audit Trail**: [`results/blockchain_audit/audit_ledger.jsonl`](file:///home/suvranil/project/results/blockchain_audit/audit_ledger.jsonl):
  - **Genesis Block**: Block #0 verified with null parent hash.
  - **Final Tip Block**: Block #2133 (Total: 2,134 blocks).
  - **Cryptographic Verification**: `blockchain_audit.verification.verify_ledger()` confirms `is_valid = True`, 0 broken backwards SHA-256 hash links, 0 genesis discrepancies, and 100% cryptographic match against disk artifact hashes.

### 8.2. Strategy-Wise and Condition-Wise Summary Matrix (N=2 Seeds: 211, 212)
| Experimental Regime | Aggregation Strategy | Seed 211 Acc | Seed 212 Acc | Mean Acc (%) | Mean Spec (%) | Mean Sens (%) | Mean Macro F1 (%) | Mean Backdoor ASR (%) | Primary Outcome / State |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| HONEST-IID | FedAvg Baseline | 85.74% | 85.42% | **85.58%** | 67.31% | 96.54% | 83.56% | — | Normal Convergence |
| HONEST-IID | Multi-Krum (f=1) | 85.58% | 86.38% | **85.98%** | 67.09% | 97.31% | 83.93% | — | Normal Convergence |
| HONEST-IID | Tri-State Monitor | 85.74% | 85.42% | **85.58%** | 67.31% | 96.54% | 83.56% | — | Normal Convergence |
| HONEST-IID | Tri-State Discount (0.50) | 85.74% | 87.02% | **86.38%** | 68.59% | 97.06% | 84.47% | — | Normal Convergence |
| HONEST-A10 | FedAvg Baseline | 82.21% | 82.37% | **82.29%** | 55.55% | 98.33% | 78.80% | — | Normal Convergence |
| HONEST-A10 | Multi-Krum (f=1) | 83.97% | 85.10% | **84.53%** | 63.03% | 97.44% | 82.03% | — | Normal Convergence |
| HONEST-A10 | Tri-State Monitor | 82.21% | 82.37% | **82.29%** | 55.55% | 98.33% | 78.80% | — | Normal Convergence |
| HONEST-A10 | Tri-State Discount (0.50) | 80.45% | 84.29% | **82.37%** | 56.20% | 98.08% | 78.87% | — | Normal Convergence |
| HONEST-A05 | FedAvg Baseline | 84.94% | 76.44% | **80.69%** | 54.48% | 96.41% | 76.38% | — | Normal Convergence |
| HONEST-A05 | Multi-Krum (f=1) | 71.96% | 70.35% | **71.16%** | 23.08% | 100.00% | 59.35% | — | Severe Specificity Collapse (23.08%) |
| HONEST-A05 | Tri-State Monitor | 84.94% | 76.44% | **80.69%** | 54.48% | 96.41% | 76.38% | — | Normal Convergence |
| HONEST-A05 | Tri-State Discount (0.50) | 84.29% | 80.77% | **82.53%** | 55.98% | 98.46% | 79.00% | — | Normal Convergence |
| HONEST-A01 | FedAvg Baseline | 81.41% | 76.12% | **78.77%** | 70.72% | 83.59% | 76.93% | — | Optimal Extreme-Skew Resilience |
| HONEST-A01 | Multi-Krum (f=1) | 82.05% | 62.50% | **72.28%** | 37.60% | 93.08% | 59.62% | — | Normal Convergence |
| HONEST-A01 | Tri-State Monitor | 81.41% | 76.12% | **78.77%** | 70.72% | 83.59% | 76.93% | — | Normal Convergence |
| HONEST-A01 | Tri-State Discount (0.50) | 62.50% | 62.50% | **62.50%** | 0.00% | 100.00% | 38.46% | — | Degenerate All-Pneumonia (Starvation) |
| ATK-LF-IID-M1 | FedAvg Baseline | 86.22% | 86.38% | **86.30%** | 71.58% | 95.13% | 84.66% | — | Normal Convergence |
| ATK-LF-IID-M1 | Multi-Krum (f=1) | 84.29% | 86.38% | **85.34%** | 65.17% | 97.44% | 83.07% | — | Normal Convergence |
| ATK-LF-IID-M1 | Tri-State Monitor | 86.22% | 86.38% | **86.30%** | 71.58% | 95.13% | 84.66% | — | Normal Convergence |
| ATK-LF-IID-M1 | Tri-State Discount (0.50) | 85.74% | 87.34% | **86.54%** | 69.66% | 96.66% | 84.73% | — | Normal Convergence |
| ATK-LF-NONIID-M1 | FedAvg Baseline | 74.36% | 73.56% | **73.96%** | 30.77% | 99.87% | 64.85% | — | Normal Convergence |
| ATK-LF-NONIID-M1 | Multi-Krum (f=1) | 77.24% | 66.19% | **71.72%** | 25.00% | 99.75% | 59.52% | — | Normal Convergence |
| ATK-LF-NONIID-M1 | Tri-State Monitor | 74.36% | 73.56% | **73.96%** | 30.77% | 99.87% | 64.85% | — | Normal Convergence |
| ATK-LF-NONIID-M1 | Tri-State Discount (0.50) | 83.81% | 82.37% | **83.09%** | 58.98% | 97.56% | 80.04% | — | Robust Specificity Rescue (+28.21%) |
| ATK-BD-IID-M1 | FedAvg Baseline | 85.58% | 86.70% | **86.14%** | 69.44% | 96.15% | 84.31% | 8.46% | Normal Convergence |
| ATK-BD-IID-M1 | Multi-Krum (f=1) | 84.29% | 86.54% | **85.42%** | 65.81% | 97.18% | 83.22% | 2.43% | Optimal Backdoor Defense (ASR 2.43%) |
| ATK-BD-IID-M1 | Tri-State Monitor | 85.58% | 86.70% | **86.14%** | 69.44% | 96.15% | 84.31% | 8.46% | Normal Convergence |
| ATK-BD-IID-M1 | Tri-State Discount (0.50) | 86.38% | 86.54% | **86.46%** | 69.66% | 96.54% | 84.66% | 5.00% | Normal Convergence |
| ATK-BN-IID-M1 | FedAvg Baseline | 37.50% | 37.50% | **37.50%** | 100.00% | 0.00% | 27.27% | — | Degenerate All-Normal Collapse |
| ATK-BN-IID-M1 | Multi-Krum (f=1) | 37.50% | 37.50% | **37.50%** | 100.00% | 0.00% | 27.27% | — | Degenerate All-Normal Collapse |
| ATK-BN-IID-M1 | Tri-State Monitor | 37.50% | 37.50% | **37.50%** | 100.00% | 0.00% | 27.27% | — | Degenerate All-Normal Collapse |
| ATK-BN-IID-M1 | Tri-State Discount (0.50) | 37.50% | 37.50% | **37.50%** | 100.00% | 0.00% | 27.27% | — | Degenerate All-Normal Collapse |
| ATK-BN-NONIID-M1 | FedAvg Baseline | 37.50% | 37.50% | **37.50%** | 100.00% | 0.00% | 27.27% | — | Degenerate All-Normal Collapse |
| ATK-BN-NONIID-M1 | Multi-Krum (f=1) | 62.50% | 62.50% | **62.50%** | 0.00% | 100.00% | 38.46% | — | Degenerate All-Pneumonia Collapse |
| ATK-BN-NONIID-M1 | Tri-State Monitor | 37.50% | 37.50% | **37.50%** | 100.00% | 0.00% | 27.27% | — | Degenerate All-Normal Collapse |
| ATK-BN-NONIID-M1 | Tri-State Discount (0.50) | 62.50% | 37.50% | **50.00%** | 50.00% | 50.00% | 32.87% | — | Divergent Degenerate Collapse |

### 8.3. Behavioral Mechanics & Scientific Findings

#### 8.3.1. Non-IID Label-Flipping Defense: Tri-State Discounting Superiority
Under moderate Dirichlet non-IID partitioning ($\alpha=0.5$) with a label-flipping adversary ($M=1$, flipping Pneumonia $\to$ Normal), standard FedAvg drops to 73.96% accuracy and 30.77% specificity. Multi-Krum ($f=1$) suffers severe degradation, dropping to 71.72% accuracy and 25.00% specificity. In contrast, **Tri-State Discounting ($\lambda=0.50$) achieves 83.09% mean accuracy and 58.98% specificity**, outperforming FedAvg by **+9.13% accuracy / +28.21% specificity** and Multi-Krum by **+11.37% accuracy / +33.98% specificity**.
- **Verified Telemetry Audit**: In Seed 211, Client 0 (attacker) was flagged `SUSPICIOUS` in 16 of 20 rounds (80%), reducing its cumulative weight from 2.0008 to 1.4422 (a net cumulative attenuation of 27.92%). In Seed 212, Client 0 was flagged `SUSPICIOUS` in 18 of 20 rounds (90%), reducing its cumulative weight from 2.0008 to 1.2805 (a net cumulative attenuation of 36.00%). In rounds where Client 0 was discounted, per-round weight attenuation reached 37.5%–50.0% relative to honest peers, suppressing adversarial bias while preserving honest minority updates.

#### 8.3.2. Multi-Krum Dual Nature: IID Backdoor Defense vs. Non-IID Specificity Collapse
Multi-Krum ($f=1, m=9$) exhibits a stark dichotomy between IID and non-IID regimes:
- **IID Backdoor Suppression**: Under IID conditions with a $3\times3$ watermark trigger, Multi-Krum effectively isolates the triggered client, reducing Backdoor Attack Success Rate (ASR) to **2.43%** (vs. FedAvg's 8.46% and Tri-State Discount's 5.00%).
- **Non-IID Geometric Rejection**: Under honest non-IID skew ($\alpha=0.5$), Multi-Krum's test specificity collapses catastrophically to **23.08%** (25.21% in Seed 211, 20.94% in Seed 212), while Pneumonia sensitivity reaches 100.00%.
- **Verified Telemetry Audit**: Feature telemetry reveals why this collapse occurs: Multi-Krum uses pairwise Euclidean parameter distance to discard $f=1$ client each round. In Seed 211, Client 2 holds 97.9% Normal cases (461 Normal, 10 Pneumonia) and was rejected in **18 of 20 rounds (90%)**, receiving only 0.2222 cumulative effective weight (vs. 2.2222 for peers). In Seed 212, Client 3 holds 98.3% Normal cases (463 Normal, 8 Pneumonia) and was rejected in **19 of 20 rounds (95%)**, receiving only 0.1111 cumulative effective weight. Multi-Krum systematically purged the primary Normal-specialized client node from global aggregation, starving the model of Normal feature gradients.

#### 8.3.3. Extreme Skew ($\alpha=0.1$) Consensus Collapse and Class Starvation
Under severe Dirichlet partitioning ($\alpha=0.1$), the majority of clients hold nearly homogeneous Pneumonia partitions, causing honest clients with high Normal proportions to diverge sharply from the collective median:
- The Phase 6 detector's directional cosine metric misidentifies divergent honest clients as `SUSPICIOUS`, applying the $\lambda=0.50$ discount.
- This creates a feedback loop of **class starvation**: discounting minority Normal updates progressively reduces the global model's capacity to recognize Normal images.
- Consequently, Tri-State Discounting collapses to a degenerate **All-Pneumonia model** across both seeds (Accuracy = 62.50%, Specificity = 0.00%, Sensitivity = 100.00%, Confusion Matrix `[0, 234, 0, 390]`).
- **FedAvg Baseline Resilience**: Simple linear unweighted FedAvg was the most robust aggregator under extreme skew, maintaining **78.77% mean accuracy and 70.72% specificity** (Seed 211: 81.41% Acc / 85.04% Spec; Seed 212: 76.12% Acc / 55.98% Spec).

#### 8.3.4. BatchNorm Buffer Poisoning Blind Spot: Universal Collapse Across All Strategies
In all 16 simulations involving Batch Normalization buffer poisoning ($M=1$, injecting $10^6$ into `running_mean` and `running_var`), **every aggregation strategy collapsed completely** into a single-class predictor:
- **Root Cause (Detector & Distance Blind Spot)**: The Phase 6 detector and Multi-Krum distance metrics evaluate strictly trainable parameter tensors (`TRAINABLE_INDICES`). Because the attacker leaves trainable weights untouched and poisons only floating-point buffers (`BN_FLOAT_INDICES`), the detector classified Client 0 as `NORMAL` in **100% of rounds (20/20 rounds across all runs)**.
- **Linear Buffer Averaging**: Aggregators linearly averaged the poisoned buffer ($10^6$) across 10 clients, setting global running statistics to $\approx 1.0\times10^5$. During test evaluation in `.eval()` mode, the corrupted running statistics completely destroyed intermediate feature maps.
- **Collapse States (Disclosed & Distinguished)**:
  - **All-Normal Collapse**: Confusion matrix `[234, 0, 390, 0]`, Test Accuracy = 37.50% (234/624), Sensitivity = 0.00% (0/390), Specificity = 100.00% (234/234), Macro F1 = 27.27%. Observed in 100% of IID runs across all strategies, and in all FedAvg / Tri-State Monitor Non-IID runs.
  - **All-Pneumonia Collapse**: Confusion matrix `[0, 234, 0, 390]`, Test Accuracy = 62.50% (390/624), Sensitivity = 100.00% (390/390), Specificity = 0.00% (0/234), Macro F1 = 38.46%. Observed in Multi-Krum Non-IID runs (Seeds 211 and 212) and Tri-State Discount Seed 211.
  - **Divergent Collapse**: In Tri-State Discount Non-IID, Seed 211 collapsed to All-Pneumonia (62.50% Acc) while Seed 212 collapsed to All-Normal (37.50% Acc), yielding a mathematical average of 50.00% Acc / 50.00% Spec / 50.00% Sens.

### 8.4. Complete 72-Cell Granular Scorecard Table
Exact metrics for all 72 primary Phase 7 simulations, evaluated on the untouched benchmark test split ($N=624$ images: 234 Normal, 390 Pneumonia):

| # | Experiment Identifier | Condition | Variant | Seed | Acc (%) | Sens (%) | Spec (%) | Macro F1 (%) | Loss | Confusion Matrix [TN, FP, FN, TP] | Backdoor ASR | Blockchain Block |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `EXP-P7-FEDAVG-HONEST-IID-S211` | HONEST-IID | fedavg | 211 | 85.74% | 96.41% | 67.95% | 83.78% | 0.4299 | [159, 75, 14, 376] | — | #43 |
| 2 | `EXP-P7-MULTI_KRUM_F1-HONEST-IID-S211` | HONEST-IID | multi_krum_f1 | 211 | 85.58% | 97.44% | 65.81% | 83.40% | 0.5170 | [154, 80, 10, 380] | — | #131 |
| 3 | `EXP-P7-TRI_STATE_MONITOR-HONEST-IID-S211` | HONEST-IID | tri_state_monitor | 211 | 85.74% | 96.41% | 67.95% | 83.78% | 0.4299 | [159, 75, 14, 376] | — | #175 |
| 4 | `EXP-P7-TRI_STATE_DISCOUNT_050-HONEST-IID-S211` | HONEST-IID | tri_state_discount_050 | 211 | 85.74% | 97.44% | 66.24% | 83.61% | 0.5017 | [155, 79, 10, 380] | — | #197 |
| 5 | `EXP-P7-FEDAVG-HONEST-IID-S212` | HONEST-IID | fedavg | 212 | 85.42% | 96.67% | 66.67% | 83.33% | 0.4423 | [156, 78, 13, 377] | — | #1363 |
| 6 | `EXP-P7-MULTI_KRUM_F1-HONEST-IID-S212` | HONEST-IID | multi_krum_f1 | 212 | 86.38% | 97.18% | 68.38% | 84.46% | 0.4668 | [160, 74, 11, 379] | — | #1385 |
| 7 | `EXP-P7-TRI_STATE_MONITOR-HONEST-IID-S212` | HONEST-IID | tri_state_monitor | 212 | 85.42% | 96.67% | 66.67% | 83.33% | 0.4423 | [156, 78, 13, 377] | — | #1407 |
| 8 | `EXP-P7-TRI_STATE_DISCOUNT_050-HONEST-IID-S212` | HONEST-IID | tri_state_discount_050 | 212 | 87.02% | 96.67% | 70.94% | 85.34% | 0.4449 | [166, 68, 13, 377] | — | #1429 |
| 9 | `EXP-P7-FEDAVG-HONEST-A10-S211` | HONEST-A10 | fedavg | 211 | 82.21% | 98.46% | 55.13% | 78.65% | 0.6669 | [129, 105, 6, 384] | — | #241 |
| 10 | `EXP-P7-MULTI_KRUM_F1-HONEST-A10-S211` | HONEST-A10 | multi_krum_f1 | 211 | 83.97% | 97.95% | 60.68% | 81.19% | 0.5059 | [142, 92, 8, 382] | — | #329 |
| 11 | `EXP-P7-TRI_STATE_MONITOR-HONEST-A10-S211` | HONEST-A10 | tri_state_monitor | 211 | 82.21% | 98.46% | 55.13% | 78.65% | 0.6669 | [129, 105, 6, 384] | — | #373 |
| 12 | `EXP-P7-TRI_STATE_DISCOUNT_050-HONEST-A10-S211` | HONEST-A10 | tri_state_discount_050 | 211 | 80.45% | 98.97% | 49.57% | 75.95% | 0.6948 | [116, 118, 4, 386] | — | #395 |
| 13 | `EXP-P7-FEDAVG-HONEST-A10-S212` | HONEST-A10 | fedavg | 212 | 82.37% | 98.21% | 55.98% | 78.94% | 0.6273 | [131, 103, 7, 383] | — | #1451 |
| 14 | `EXP-P7-MULTI_KRUM_F1-HONEST-A10-S212` | HONEST-A10 | multi_krum_f1 | 212 | 85.10% | 96.92% | 65.38% | 82.87% | 0.4877 | [153, 81, 12, 378] | — | #1473 |
| 15 | `EXP-P7-TRI_STATE_MONITOR-HONEST-A10-S212` | HONEST-A10 | tri_state_monitor | 212 | 82.37% | 98.21% | 55.98% | 78.94% | 0.6273 | [131, 103, 7, 383] | — | #1495 |
| 16 | `EXP-P7-TRI_STATE_DISCOUNT_050-HONEST-A10-S212` | HONEST-A10 | tri_state_discount_050 | 212 | 84.29% | 97.18% | 62.82% | 81.78% | 0.5358 | [147, 87, 11, 379] | — | #1517 |
| 17 | `EXP-P7-FEDAVG-HONEST-A05-S211` | HONEST-A05 | fedavg | 211 | 84.94% | 93.59% | 70.51% | 83.21% | 0.4207 | [165, 69, 25, 365] | — | #439 |
| 18 | `EXP-P7-MULTI_KRUM_F1-HONEST-A05-S211` | HONEST-A05 | multi_krum_f1 | 211 | 71.96% | 100.00% | 25.21% | 60.97% | 0.4769 | [59, 175, 0, 390] | — | #527 |
| 19 | `EXP-P7-TRI_STATE_MONITOR-HONEST-A05-S211` | HONEST-A05 | tri_state_monitor | 211 | 84.94% | 93.59% | 70.51% | 83.21% | 0.4207 | [165, 69, 25, 365] | — | #571 |
| 20 | `EXP-P7-TRI_STATE_DISCOUNT_050-HONEST-A05-S211` | HONEST-A05 | tri_state_discount_050 | 211 | 84.29% | 97.69% | 61.97% | 81.67% | 0.3864 | [145, 89, 9, 381] | — | #593 |
| 21 | `EXP-P7-FEDAVG-HONEST-A05-S212` | HONEST-A05 | fedavg | 212 | 76.44% | 99.23% | 38.46% | 69.54% | 0.4431 | [90, 144, 3, 387] | — | #1539 |
| 22 | `EXP-P7-MULTI_KRUM_F1-HONEST-A05-S212` | HONEST-A05 | multi_krum_f1 | 212 | 70.35% | 100.00% | 20.94% | 57.73% | 0.5228 | [49, 185, 0, 390] | — | #1561 |
| 23 | `EXP-P7-TRI_STATE_MONITOR-HONEST-A05-S212` | HONEST-A05 | tri_state_monitor | 212 | 76.44% | 99.23% | 38.46% | 69.54% | 0.4431 | [90, 144, 3, 387] | — | #1583 |
| 24 | `EXP-P7-TRI_STATE_DISCOUNT_050-HONEST-A05-S212` | HONEST-A05 | tri_state_discount_050 | 212 | 80.77% | 99.23% | 50.00% | 76.34% | 0.5679 | [117, 117, 3, 387] | — | #1605 |
| 25 | `EXP-P7-FEDAVG-HONEST-A01-S211` | HONEST-A01 | fedavg | 211 | 81.41% | 79.23% | 85.04% | 80.81% | 0.6299 | [199, 35, 81, 309] | — | #637 |
| 26 | `EXP-P7-MULTI_KRUM_F1-HONEST-A01-S211` | HONEST-A01 | multi_krum_f1 | 211 | 82.05% | 86.15% | 75.21% | 80.79% | 0.6127 | [176, 58, 54, 336] | — | #725 |
| 27 | `EXP-P7-TRI_STATE_MONITOR-HONEST-A01-S211` | HONEST-A01 | tri_state_monitor | 211 | 81.41% | 79.23% | 85.04% | 80.81% | 0.6299 | [199, 35, 81, 309] | — | #769 |
| 28 | `EXP-P7-TRI_STATE_DISCOUNT_050-HONEST-A01-S211` | HONEST-A01 | tri_state_discount_050 | 211 | 62.50% | 100.00% | 0.00% | 38.46% | 5.0072 | [0, 234, 0, 390] | — | #791 |
| 29 | `EXP-P7-FEDAVG-HONEST-A01-S212` | HONEST-A01 | fedavg | 212 | 76.12% | 87.95% | 56.41% | 73.04% | 0.5539 | [132, 102, 47, 343] | — | #1627 |
| 30 | `EXP-P7-MULTI_KRUM_F1-HONEST-A01-S212` | HONEST-A01 | multi_krum_f1 | 212 | 62.50% | 100.00% | 0.00% | 38.46% | 5.8900 | [0, 234, 0, 390] | — | #1649 |
| 31 | `EXP-P7-TRI_STATE_MONITOR-HONEST-A01-S212` | HONEST-A01 | tri_state_monitor | 212 | 76.12% | 87.95% | 56.41% | 73.04% | 0.5539 | [132, 102, 47, 343] | — | #1671 |
| 32 | `EXP-P7-TRI_STATE_DISCOUNT_050-HONEST-A01-S212` | HONEST-A01 | tri_state_discount_050 | 212 | 62.50% | 100.00% | 0.00% | 38.46% | 6.7606 | [0, 234, 0, 390] | — | #1693 |
| 33 | `EXP-P7-FEDAVG-ATK-LF-IID-M1-S211` | ATK-LF-IID-M1 | fedavg | 211 | 86.22% | 93.59% | 73.93% | 84.78% | 0.3347 | [173, 61, 25, 365] | — | #835 |
| 34 | `EXP-P7-MULTI_KRUM_F1-ATK-LF-IID-M1-S211` | ATK-LF-IID-M1 | multi_krum_f1 | 211 | 84.29% | 97.44% | 62.39% | 81.72% | 0.4978 | [146, 88, 10, 380] | — | #923 |
| 35 | `EXP-P7-TRI_STATE_MONITOR-ATK-LF-IID-M1-S211` | ATK-LF-IID-M1 | tri_state_monitor | 211 | 86.22% | 93.59% | 73.93% | 84.78% | 0.3347 | [173, 61, 25, 365] | — | #967 |
| 36 | `EXP-P7-TRI_STATE_DISCOUNT_050-ATK-LF-IID-M1-S211` | ATK-LF-IID-M1 | tri_state_discount_050 | 211 | 85.74% | 96.92% | 67.09% | 83.69% | 0.4444 | [157, 77, 12, 378] | — | #989 |
| 37 | `EXP-P7-FEDAVG-ATK-LF-IID-M1-S212` | ATK-LF-IID-M1 | fedavg | 212 | 86.38% | 96.67% | 69.23% | 84.54% | 0.4259 | [162, 72, 13, 377] | — | #1715 |
| 38 | `EXP-P7-MULTI_KRUM_F1-ATK-LF-IID-M1-S212` | ATK-LF-IID-M1 | multi_krum_f1 | 212 | 86.38% | 97.44% | 67.95% | 84.42% | 0.4643 | [159, 75, 10, 380] | — | #1737 |
| 39 | `EXP-P7-TRI_STATE_MONITOR-ATK-LF-IID-M1-S212` | ATK-LF-IID-M1 | tri_state_monitor | 212 | 86.38% | 96.67% | 69.23% | 84.54% | 0.4259 | [162, 72, 13, 377] | — | #1759 |
| 40 | `EXP-P7-TRI_STATE_DISCOUNT_050-ATK-LF-IID-M1-S212` | ATK-LF-IID-M1 | tri_state_discount_050 | 212 | 87.34% | 96.41% | 72.22% | 85.77% | 0.4261 | [169, 65, 14, 376] | — | #1781 |
| 41 | `EXP-P7-FEDAVG-ATK-LF-NONIID-M1-S211` | ATK-LF-NONIID-M1 | fedavg | 211 | 74.36% | 99.74% | 32.05% | 65.66% | 0.8595 | [75, 159, 1, 389] | — | #1011 |
| 42 | `EXP-P7-MULTI_KRUM_F1-ATK-LF-NONIID-M1-S211` | ATK-LF-NONIID-M1 | multi_krum_f1 | 211 | 77.24% | 99.49% | 40.17% | 70.75% | 0.4471 | [94, 140, 2, 388] | — | #1033 |
| 43 | `EXP-P7-TRI_STATE_MONITOR-ATK-LF-NONIID-M1-S211` | ATK-LF-NONIID-M1 | tri_state_monitor | 211 | 74.36% | 99.74% | 32.05% | 65.66% | 0.8595 | [75, 159, 1, 389] | — | #1055 |
| 44 | `EXP-P7-TRI_STATE_DISCOUNT_050-ATK-LF-NONIID-M1-S211` | ATK-LF-NONIID-M1 | tri_state_discount_050 | 211 | 83.81% | 96.15% | 63.25% | 81.35% | 0.4061 | [148, 86, 15, 375] | — | #1077 |
| 45 | `EXP-P7-FEDAVG-ATK-LF-NONIID-M1-S212` | ATK-LF-NONIID-M1 | fedavg | 212 | 73.56% | 100.00% | 29.49% | 64.04% | 0.8828 | [69, 165, 0, 390] | — | #1803 |
| 46 | `EXP-P7-MULTI_KRUM_F1-ATK-LF-NONIID-M1-S212` | ATK-LF-NONIID-M1 | multi_krum_f1 | 212 | 66.19% | 100.00% | 9.83% | 48.30% | 1.2925 | [23, 211, 0, 390] | — | #1825 |
| 47 | `EXP-P7-TRI_STATE_MONITOR-ATK-LF-NONIID-M1-S212` | ATK-LF-NONIID-M1 | tri_state_monitor | 212 | 73.56% | 100.00% | 29.49% | 64.04% | 0.8828 | [69, 165, 0, 390] | — | #1847 |
| 48 | `EXP-P7-TRI_STATE_DISCOUNT_050-ATK-LF-NONIID-M1-S212` | ATK-LF-NONIID-M1 | tri_state_discount_050 | 212 | 82.37% | 98.97% | 54.70% | 78.74% | 0.4533 | [128, 106, 4, 386] | — | #1869 |
| 49 | `EXP-P7-FEDAVG-ATK-BD-IID-M1-S211` | ATK-BD-IID-M1 | fedavg | 211 | 85.58% | 96.15% | 67.95% | 83.61% | 0.3813 | [159, 75, 15, 375] | 7.69% | #1099 |
| 50 | `EXP-P7-MULTI_KRUM_F1-ATK-BD-IID-M1-S211` | ATK-BD-IID-M1 | multi_krum_f1 | 211 | 84.29% | 97.44% | 62.39% | 81.72% | 0.4978 | [146, 88, 10, 380] | 2.05% | #1121 |
| 51 | `EXP-P7-TRI_STATE_MONITOR-ATK-BD-IID-M1-S211` | ATK-BD-IID-M1 | tri_state_monitor | 211 | 85.58% | 96.15% | 67.95% | 83.61% | 0.3813 | [159, 75, 15, 375] | 7.69% | #1143 |
| 52 | `EXP-P7-TRI_STATE_DISCOUNT_050-ATK-BD-IID-M1-S211` | ATK-BD-IID-M1 | tri_state_discount_050 | 211 | 86.38% | 96.41% | 69.66% | 84.58% | 0.4015 | [163, 71, 14, 376] | 4.87% | #1165 |
| 53 | `EXP-P7-FEDAVG-ATK-BD-IID-M1-S212` | ATK-BD-IID-M1 | fedavg | 212 | 86.70% | 96.15% | 70.94% | 85.02% | 0.3953 | [166, 68, 15, 375] | 9.23% | #1891 |
| 54 | `EXP-P7-MULTI_KRUM_F1-ATK-BD-IID-M1-S212` | ATK-BD-IID-M1 | multi_krum_f1 | 212 | 86.54% | 96.92% | 69.23% | 84.71% | 0.4571 | [162, 72, 12, 378] | 2.82% | #1913 |
| 55 | `EXP-P7-TRI_STATE_MONITOR-ATK-BD-IID-M1-S212` | ATK-BD-IID-M1 | tri_state_monitor | 212 | 86.70% | 96.15% | 70.94% | 85.02% | 0.3953 | [166, 68, 15, 375] | 9.23% | #1935 |
| 56 | `EXP-P7-TRI_STATE_DISCOUNT_050-ATK-BD-IID-M1-S212` | ATK-BD-IID-M1 | tri_state_discount_050 | 212 | 86.54% | 96.67% | 69.66% | 84.74% | 0.4266 | [163, 71, 13, 377] | 5.13% | #1957 |
| 57 | `EXP-P7-FEDAVG-ATK-BN-IID-M1-S211` | ATK-BN-IID-M1 | fedavg | 211 | 37.50% | 0.00% | 100.00% | 27.27% | 0.6953 | [234, 0, 390, 0] | — | #1187 |
| 58 | `EXP-P7-MULTI_KRUM_F1-ATK-BN-IID-M1-S211` | ATK-BN-IID-M1 | multi_krum_f1 | 211 | 37.50% | 0.00% | 100.00% | 27.27% | 0.6953 | [234, 0, 390, 0] | — | #1209 |
| 59 | `EXP-P7-TRI_STATE_MONITOR-ATK-BN-IID-M1-S211` | ATK-BN-IID-M1 | tri_state_monitor | 211 | 37.50% | 0.00% | 100.00% | 27.27% | 0.6953 | [234, 0, 390, 0] | — | #1231 |
| 60 | `EXP-P7-TRI_STATE_DISCOUNT_050-ATK-BN-IID-M1-S211` | ATK-BN-IID-M1 | tri_state_discount_050 | 211 | 37.50% | 0.00% | 100.00% | 27.27% | 0.6953 | [234, 0, 390, 0] | — | #1253 |
| 61 | `EXP-P7-FEDAVG-ATK-BN-IID-M1-S212` | ATK-BN-IID-M1 | fedavg | 212 | 37.50% | 0.00% | 100.00% | 27.27% | 0.6954 | [234, 0, 390, 0] | — | #1979 |
| 62 | `EXP-P7-MULTI_KRUM_F1-ATK-BN-IID-M1-S212` | ATK-BN-IID-M1 | multi_krum_f1 | 212 | 37.50% | 0.00% | 100.00% | 27.27% | 0.6954 | [234, 0, 390, 0] | — | #2001 |
| 63 | `EXP-P7-TRI_STATE_MONITOR-ATK-BN-IID-M1-S212` | ATK-BN-IID-M1 | tri_state_monitor | 212 | 37.50% | 0.00% | 100.00% | 27.27% | 0.6954 | [234, 0, 390, 0] | — | #2023 |
| 64 | `EXP-P7-TRI_STATE_DISCOUNT_050-ATK-BN-IID-M1-S212` | ATK-BN-IID-M1 | tri_state_discount_050 | 212 | 37.50% | 0.00% | 100.00% | 27.27% | 0.6954 | [234, 0, 390, 0] | — | #2045 |
| 65 | `EXP-P7-FEDAVG-ATK-BN-NONIID-M1-S211` | ATK-BN-NONIID-M1 | fedavg | 211 | 37.50% | 0.00% | 100.00% | 27.27% | 0.6947 | [234, 0, 390, 0] | — | #1275 |
| 66 | `EXP-P7-MULTI_KRUM_F1-ATK-BN-NONIID-M1-S211` | ATK-BN-NONIID-M1 | multi_krum_f1 | 211 | 62.50% | 100.00% | 0.00% | 38.46% | 0.6930 | [0, 234, 0, 390] | — | #1297 |
| 67 | `EXP-P7-TRI_STATE_MONITOR-ATK-BN-NONIID-M1-S211` | ATK-BN-NONIID-M1 | tri_state_monitor | 211 | 37.50% | 0.00% | 100.00% | 27.27% | 0.6947 | [234, 0, 390, 0] | — | #1319 |
| 68 | `EXP-P7-TRI_STATE_DISCOUNT_050-ATK-BN-NONIID-M1-S211` | ATK-BN-NONIID-M1 | tri_state_discount_050 | 211 | 62.50% | 100.00% | 0.00% | 38.46% | 0.6931 | [0, 234, 0, 390] | — | #1341 |
| 69 | `EXP-P7-FEDAVG-ATK-BN-NONIID-M1-S212` | ATK-BN-NONIID-M1 | fedavg | 212 | 37.50% | 0.00% | 100.00% | 27.27% | 0.6947 | [234, 0, 390, 0] | — | #2067 |
| 70 | `EXP-P7-MULTI_KRUM_F1-ATK-BN-NONIID-M1-S212` | ATK-BN-NONIID-M1 | multi_krum_f1 | 212 | 62.50% | 100.00% | 0.00% | 38.46% | 0.6928 | [0, 234, 0, 390] | — | #2089 |
| 71 | `EXP-P7-TRI_STATE_MONITOR-ATK-BN-NONIID-M1-S212` | ATK-BN-NONIID-M1 | tri_state_monitor | 212 | 37.50% | 0.00% | 100.00% | 27.27% | 0.6947 | [234, 0, 390, 0] | — | #2111 |
| 72 | `EXP-P7-TRI_STATE_DISCOUNT_050-ATK-BN-NONIID-M1-S212` | ATK-BN-NONIID-M1 | tri_state_discount_050 | 212 | 37.50% | 0.00% | 100.00% | 27.27% | 0.6943 | [234, 0, 390, 0] | — | #2133 |

### 8.5. Preserved Supplementary Historical Experiments (N=24)
In addition to the 72 balanced primary matrix cells, 24 supplementary historical experiments from initial Phase 7 explorations are permanently preserved in [`results/phase7_robust_aggregation/execution_registry.json`](file:///home/suvranil/project/results/phase7_robust_aggregation/execution_registry.json):

| Strategy Variant | Experiments Preserved | Regimes Covered | Primary Findings |
|---|:---:|---|---|
| **Coordinate Median** | 5 | Honest IID, A10, A05, A01; ATK-LF-IID | Strong IID baseline (86.54% Acc / 69.66% Spec); degraded under A01 (74.84% Acc / 50.00% Spec). |
| **Trimmed Mean ($\beta=1$)** | 5 | Honest IID, A10, A05, A01; ATK-LF-IID | Solid IID baseline (86.22% Acc / 68.80% Spec); dropped under A01 (78.37% Acc / 67.52% Spec). |
| **Trimmed Mean ($\beta=2$)** | 5 | Honest IID, A10, A05, A01; ATK-LF-IID | Trimming 2 highest/lowest values caused over-trimming under non-IID partitions. |
| **Multi-Krum ($f=2$)** | 5 | Honest IID, A10, A05, A01; ATK-LF-IID | Discarding $f=2$ candidates worsened non-IID specificity collapse compared to $f=1$. |
| **Tri-State Discount ($\lambda=0.25$)** | 4 | Honest IID, A10, A05, A01 | More aggressive discount factor (0.25) accelerated class starvation under non-IID skew. |


---

## 9. Summary of Archived CIFAR-10 Results
The previous generic proof-of-concept remains archived in [`results/archive/cifar10/`](file:///home/suvranil/project/results/archive/cifar10/README.md):
- **Dataset**: CIFAR-10 (60,000 images, 10 classes, 32x32 RGB).
- **Model**: SmallCNN (189,130 parameters).
- **Test Accuracy**: 76.94% (7,694 / 10,000 correct).
- **Training Time**: 100.97 seconds on RTX 3050 GPU.
- **Status**: Archived; superseded by PneumoniaMNIST for the hospital project.



