# Project Status

## Project Title
**Multi-Hospital Federated Learning for Chest X-Ray Classification with Malicious Participant Detection**

---

## Current Status Overview
- **Active Primary Dataset**: **PneumoniaMNIST** (Pediatric Chest X-Rays from MedMNIST v3)
- **Problem Scenario**: **Hospital-based simulated Federated Learning** across 10 hospital clients
- **Current Model**: **SmallCNN** (2 Conv blocks, BatchNorm, MaxPool, Dense 128, 2-class output; **420,802 parameters**)
- **Current Milestone**: **Phase 7 Sequential Execution Active** (Implemented sequential O(1)-memory execution engine, persistent execution registry, and lightweight permissioned blockchain audit layer in `blockchain_audit/`; verified across 16 dedicated unit and parity tests, 89/89 full project suite tests passing; 8 / 351 matrix simulations completed)
- **Immediate Next Step**: **Phase 7 Sequential Matrix Batch Execution** (Remaining 343 simulations across 13 conditions, 9 aggregation variants, seeds 211, 212, 213, with tamper-evident audit logging)
- **Status Date**: November 2, 2026
- **Lead Developer**: AI Research Pair Programmer
- **Strict Scope Boundary**: Frozen detector v1 untouched; Phase 7 matrix (351 simulations) preserved intact; blockchain audit component operates as an experimental permissioned audit layer without heavy mining infrastructure.


---

## 1. Documentation Index

The project documentation has been fully organized under [`docs/`](file:///home/suvranil/project/docs):
- [`docs/PROJECT_CONTEXT.md`](file:///home/suvranil/project/docs/PROJECT_CONTEXT.md) — Clinical motivation, healthcare privacy requirements, and 4-pillar architecture.
- [`docs/WHY_CNN_AND_DATASET.md`](file:///home/suvranil/project/docs/WHY_CNN_AND_DATASET.md) — Comprehensive technical justification for PneumoniaMNIST and SmallCNN.
- [`docs/HOSPITAL_SCENARIO.md`](file:///home/suvranil/project/docs/HOSPITAL_SCENARIO.md) — 10-hospital federated training protocol, non-IID clinical skew vs. attack dilemma, and blockchain auditability.
- [`docs/RESEARCH_ROADMAP.md`](file:///home/suvranil/project/docs/RESEARCH_ROADMAP.md) — Phased development plan with Phase 3 (IID FL) as the next immediate step and Phase 4 (Non-IID) as later work.
- [`docs/RESEARCH_GAP_HOSPITAL.md`](file:///home/suvranil/project/docs/RESEARCH_GAP_HOSPITAL.md) — Failure modes of classical Byzantine aggregators in healthcare.
- [`docs/CURRENT_RESULTS.md`](file:///home/suvranil/project/docs/CURRENT_RESULTS.md) — Complete scorecard of measured empirical results.

---

## 2. Completed Milestones

### Migration to Hospital Scenario
- [x] Preserved pre-migration Git checkpoint on branch `checkpoint/pre-medical-migration` (`3f2e0ab`).
- [x] Designated **PneumoniaMNIST** (Pediatric Chest X-Rays) as the **Primary Medical Project Dataset**.
- [x] Safely archived all previous CIFAR-10 proof-of-concept work in [`results/archive/cifar10/`](file:///home/suvranil/project/results/archive/cifar10/README.md) (figures, logs, models, documentation) without mixing with medical artifacts.
- [x] Installed `medmnist>=3.0.2` and `scikit-learn>=1.4.0` in `.venv` and recorded in [`requirements.txt`](file:///home/suvranil/project/requirements.txt).

### Phase MEDICAL-01: Medical Dataset Verification
- [x] Verified automatic download and integrity in [`data/verify_dataset.py`](file:///home/suvranil/project/data/verify_dataset.py).
- [x] Confirmed exact official split sizes: Train = 4,708, Val = 524, Test = 624 (Total = 5,856).
- [x] Confirmed image format: $1 \times 28 \times 28$ single-channel grayscale.
- [x] Verified binary diagnosis labels: `0 = Normal` (1,214 train, 135 val, 234 test), `1 = Pneumonia` (3,494 train, 389 val, 390 test).
- [x] Measured empirical pixel statistics: Min = 0, Max = 255, Mean = 0.57192, Std = 0.16835.
- [x] Generated visual figures:
  - [`results/figures/pneumonia_sample_grid.png`](file:///home/suvranil/project/results/figures/pneumonia_sample_grid.png)
  - [`results/figures/pneumonia_class_distribution.png`](file:///home/suvranil/project/results/figures/pneumonia_class_distribution.png)
- [x] Emitted structured JSON verification log: [`results/logs/pneumonia_mnist_verification_report.json`](file:///home/suvranil/project/results/logs/pneumonia_mnist_verification_report.json).

### Phase MEDICAL-02: Centralized Medical Small CNN Baseline
- [x] Adapted [`SmallCNN`](file:///home/suvranil/project/federated_learning/model.py#L21-L71) in [`federated_learning/model.py`](file:///home/suvranil/project/federated_learning/model.py) for $1 \times 28 \times 28$ grayscale input and 2 output classes.
- [x] Documented exact parameter count: **420,802 total and trainable parameters** (~1.61 MB).
- [x] Implemented centralized training pipeline in [`experiments/train_medical.py`](file:///home/suvranil/project/experiments/train_medical.py) using PyTorch, Adam (lr=0.001), CrossEntropyLoss, and master seed 42.
- [x] Preserved official MedMNIST train (4,708), val (524), and untouched test (624) splits without artificial re-splitting.
- [x] Trained model on NVIDIA GeForce RTX 3050 Laptop GPU (15 epochs, 13.35 seconds):
  - Best Validation Accuracy: **97.90%** (Epoch 14, Val Loss: 0.0760)
  - Untouched Test Accuracy: **81.25%** (507 / 624 correct)
  - Final Test Loss: **0.9172**
  - Pneumonia Sensitivity / Recall: **99.74%** (389 / 390 pneumonia cases correctly identified, only 1 missed)
  - Pneumonia Precision: **77.03%**
  - Pneumonia F1-Score: **0.8693**
  - Normal Specificity / Recall: **50.43%** (118 / 234 normal cases correctly identified)
  - Normal Precision: **99.16%**
  - Normal F1-Score: **0.6686**
  - Macro F1-Score: **0.7689**
  - Confusion Matrix: TN=118, FP=116, FN=1, TP=389
- [x] Emitted structured JSON experiment logs: [`results/logs/medical_centralized_results.json`](file:///home/suvranil/project/results/logs/medical_centralized_results.json).
- [x] Generated visual figures:
  - [`results/figures/medical_training_loss.png`](file:///home/suvranil/project/results/figures/medical_training_loss.png)
  - [`results/figures/medical_training_accuracy.png`](file:///home/suvranil/project/results/figures/medical_training_accuracy.png)
  - [`results/figures/medical_confusion_matrix.png`](file:///home/suvranil/project/results/figures/medical_confusion_matrix.png)
- [x] Saved best model checkpoint: [`results/models/best_centralized_medical_cnn.pt`](file:///home/suvranil/project/results/models/best_centralized_medical_cnn.pt).

### Phase 3: 10-Hospital IID Federated Learning Baseline
- [x] Implemented reproducible stratified IID partitioning across 10 hospital clients in [`federated_learning/dataset.py`](file:///home/suvranil/project/federated_learning/dataset.py) (4,708 samples: 8 hospitals with 471, 2 hospitals with 470).
- [x] Preserved empirical class ratios across all clients (~25.8% Normal, ~74.2% Pneumonia).
- [x] Implemented simulated hospital client node in [`federated_learning/client.py`](file:///home/suvranil/project/federated_learning/client.py) with independent model instances and local training.
- [x] Configured Flower FedAvg strategy and server-side validation evaluation hook in [`federated_learning/strategy.py`](file:///home/suvranil/project/federated_learning/strategy.py).
- [x] Implemented end-to-end orchestration pipeline in [`experiments/train_federated.py`](file:///home/suvranil/project/experiments/train_federated.py).
- [x] Executed 20-round simulation with 2 local epochs/round on CPU (Ray VCE pool) in 108.29 seconds:
  - Best Global Validation Accuracy: **96.18%** (Round 17, Val Loss: 0.0833)
  - Untouched Test Set Accuracy: **85.74%** (535 / 624 correct; +4.49% over Centralized Baseline)
  - Pneumonia Sensitivity / Recall: **97.95%** (382 / 390 cases identified)
  - Pneumonia Precision: **82.51%** (382 / 463)
  - Pneumonia F1-Score: **0.8957**
  - Normal Specificity / Recall: **65.38%** (153 / 234 normal cases identified; +14.95% over Centralized Baseline)
  - Macro F1-Score: **0.8352**
  - Confusion Matrix: TN=153, FP=81, FN=8, TP=382
- [x] Saved global checkpoints:
  - Best Checkpoint: [`results/models/best_federated_iid_model.pt`](file:///home/suvranil/project/results/models/best_federated_iid_model.pt)
  - Final Checkpoint: [`results/models/final_federated_iid_model.pt`](file:///home/suvranil/project/results/models/final_federated_iid_model.pt)
- [x] Generated visual figures:
  - [`results/figures/federated_iid_training_loss.png`](file:///home/suvranil/project/results/figures/federated_iid_training_loss.png)
  - [`results/figures/federated_iid_training_accuracy.png`](file:///home/suvranil/project/results/figures/federated_iid_training_accuracy.png)
  - [`results/figures/federated_iid_confusion_matrix.png`](file:///home/suvranil/project/results/figures/federated_iid_confusion_matrix.png)
  - [`results/figures/centralized_vs_federated_comparison.png`](file:///home/suvranil/project/results/figures/centralized_vs_federated_comparison.png)
- [x] Emitted structured JSON experiment log: [`results/logs/federated_iid_results.json`](file:///home/suvranil/project/results/logs/federated_iid_results.json).

### Phase 4: Non-IID Hospital Data Heterogeneity Study
- [x] Implemented population-calibrated constrained Beta/Dirichlet case-mix partitioning in [`federated_learning/dataset.py`](file:///home/suvranil/project/federated_learning/dataset.py).
- [x] Controlled client sample sizes across 10 hospitals to ~471 samples (8 with 471, 2 with 470, sum = 4,708), isolating label-distribution heterogeneity.
- [x] Verified strict conservation of global classes (1,214 Normal, 3,494 Pneumonia) and mutual exclusivity across all clients.
- [x] Enforced identical model initialization across all runs via [`results/models/initial_federated_model.pt`](file:///home/suvranil/project/results/models/initial_federated_model.pt) (verified SHA-256: `0c411b83c2f0acef58beb5b9a968030d877e079eb2356eb3748903f8ae9bbc83`).
- [x] Preserved BatchNorm running buffers aggregation in FedAvg matching Phase 3.
- [x] Executed 20-round simulations for $\alpha = 1.0, 0.5, 0.1$:
  - $\alpha = 1.0$: 79.81% Test Acc, 98.97% Sensitivity, 47.86% Specificity, 0.7498 Macro F1 (Best Round 15: 95.99% Val Acc).
  - $\alpha = 0.5$: 80.45% Test Acc, 98.46% Sensitivity, 50.43% Specificity, 0.7611 Macro F1 (Best Round 18: 95.99% Val Acc).
  - $\alpha = 0.1$: 85.10% Test Acc, 91.79% Sensitivity, 73.93% Specificity, 0.8366 Macro F1 (Best Round 12: 89.89% Val Acc; severe subsequent oscillation).
- [x] Saved all model checkpoints (`best_federated_non_iid_alpha*.pt`, `final_federated_non_iid_alpha*.pt`).
- [x] Emitted structured JSON experiment logs and generated comparison plots ([`results/figures/iid_vs_non_iid_heterogeneity_comparison.png`](file:///home/suvranil/project/results/figures/iid_vs_non_iid_heterogeneity_comparison.png)).

### Phase 5: Adversarial Attacks Simulation & Matched Controls
- [x] Implemented targeted label flipping (`federated_learning/attacks/label_flipping.py`).
- [x] Implemented model update poisoning strictly on 12 trainable tensors (`federated_learning/attacks/model_poisoning.py`) while preserving BatchNorm buffers.
- [x] Implemented backdoor trigger injection (`federated_learning/attacks/backdoor.py`) with post-normalization normalized white watermark (`+2.5422`) and synthetic triggered evaluation set ($N=390$).
- [x] Implemented parameterized simulation runner (`experiments/train_federated_attack.py`) enforcing clean validation-only checkpoint selection.
- [x] Verified 100% parameter hash parity (`0c411b83...`) and partition reproducibility hashes across all runs.
- [x] Executed all 9 Phase 5 simulations sequentially (`EXP-MED-006` through `EXP-MED-014`):
  - Fresh Honest Controls: IID (84.62% test acc, FN=7, ASR=0.51%) and Non-IID $\alpha=0.5$ (78.37% test acc, FN=4, ASR=0.26%).
  - Label Flipping: $M=1$ IID (FN: $7 \to 14$), $M=2$ IID (FN: $7 \to 60$, $+757\%$), $M=1$ Non-IID (FN: $4 \to 11$, $+175\%$).
  - Model Poisoning: Inversion $\gamma=-1.0$ (FN: 8) and Scaling $\gamma=+10.0$ (FN: 9).
  - Backdoor Injection: IID $M=1$ ($\Delta \text{ASR} = +6.93\%$, 14.6x jump) and Non-IID $M=1$ ($\Delta \text{ASR} = +8.20\%$, 32.5x jump; Non-IID heterogeneity amplifies backdoor vulnerability).
- [x] Generated multi-panel comparison figure ([`results/figures/phase5_attack_comparison.png`](file:///home/suvranil/project/results/figures/phase5_attack_comparison.png)) and structured summary log ([`results/logs/phase5_attack_comparison.json`](file:///home/suvranil/project/results/logs/phase5_attack_comparison.json)).

### Phase 6: Multi-Factor Non-IID-Aware Malicious Hospital Detection & Empirical Validation
- [x] Implemented multi-factor feature extraction (`federated_learning/detection/feature_extractor.py`) tracking 5 feature families: global/layer magnitudes, directional cosines, class-disentangled proxy validation loss impacts, and historical trajectory velocity/EMA accumulator.
- [x] Implemented layer-wise parameter architecture targeting exact layers: Conv Block 1 (384), Conv Block 2 (18,624), Dense Projection (401,536), Classification Head (258); Total $D = 420,802$ parameters.
- [x] Implemented calibrated tri-state detector policy (`federated_learning/detection/detector.py`) with `NORMAL`, `SUSPICIOUS`, and `MALICIOUS` decisions.
- [x] Calibrated empirical quantiles on 400 honest baseline observations (`EXP-MED-006` and `EXP-MED-007`) with Bonferroni multiple testing correction.
- [x] Removed calibration-data leakage by evaluating strictly on held-out runs (9 simulations, 1,800 observations: 1,640 honest, 160 malicious).
- [x] Established rigorous dual reporting:
  - **Out-of-Sample Calibration-Excluded (9 Runs, 1,800 Obs)**: AUROC = **0.9814**, AUPRC = **0.8801**, Total Flagged Recall = **96.25%**, False Malicious Rate = **0.00%**, Precision = **100.00%**.
  - **In-Sample Reference (11 Runs, 2,200 Obs)**: AUROC = **0.9761**, AUPRC = **0.8026**, Total Flagged Recall = **96.25%**, False Malicious Rate = **0.00%**.
- [x] Conducted 12 independent multi-seed honest simulations (`EXP-MED-HONEST-*`) across Seeds 43, 44, 45 (2,400 observations): False Malicious Rate = **0.04%** (1 / 2,400).
- [x] Conducted forensic audit across 16 honest simulations (3,200 observations):
  - Investigated negative cosine occurrences (199 events, 6.22% prevalence, down to $-0.8411$).
  - Proved negative cosine is an expected consequence of class specialization in extreme Non-IID ($\alpha=0.1$).
  - Verified that Symmetrical Clinical Trade-Off Protection successfully shielded **199 / 199 (100.0%)** of negative cosine events from wrongful loss penalties.
- [x] Conducted instrumentation parity audit:
  - Verified exact mathematical equivalence between standard `FedAvg` and `InstrumentedFedAvg` (**0.000 max parameter diff** on identical inputs across all 18 state dict elements in [`tests/test_instrumentation_parity.py`](file:///home/suvranil/project/tests/test_instrumentation_parity.py)).
- [x] Generated publication figures ([`results/phase6_detector/figures/phase6_detector_performance.png`](file:///home/suvranil/project/results/phase6_detector/figures/phase6_detector_performance.png)) and structured JSON benchmarks.
- [x] **Conducted Phase 6 Frozen Blind Evaluation with Seed Integrity Correction (`EXP-MED-020`)**:
  - Segregated prior seed-101 runs into [`results/phase6_preflight_exploratory_seed101/`](file:///home/suvranil/project/results/phase6_preflight_exploratory_seed101) with explicit PREFLIGHT labels.
  - Executed 33 fresh simulations (6,600 observations) across untouched seeds `111, 112, 113` using read-only frozen detector (`frozen_v1/`).
  - Achieved **0.00% Hard Malicious False Positive Rate** (0 / 6,120) across all honest IID and Non-IID hospitals.
  - Achieved **93.75% Threat Flagged Recall** (450 / 480) and 100% elimination of model scaling attacks.
  - Published standalone report in [`docs/PHASE6_BLIND_EVALUATION_REPORT.md`](file:///home/suvranil/project/docs/PHASE6_BLIND_EVALUATION_REPORT.md).

### Quality, Tests & Dashboard
- [x] Automated test suite in `tests/`: **51 / 51 PyTest tests passing (100%)** in 8.97 seconds.
  - Added 5 unit tests in [`tests/test_detection.py`](file:///home/suvranil/project/tests/test_detection.py) verifying layer accounting, Modified Z-score MAD robustness, clinical trade-off protection, tri-state decision consistency, and stratified FPR.
  - Added direct parity test in [`tests/test_instrumentation_parity.py`](file:///home/suvranil/project/tests/test_instrumentation_parity.py) proving 0.000 bit-level aggregation equivalence.
- [x] Interactive teacher demonstration dashboard ([`dashboard/app.py`](file:///home/suvranil/project/dashboard/app.py)).

---

## 3. What Remains (Upcoming Phases)
- [x] **Phase MEDICAL-01**: Medical Dataset Verification (PneumoniaMNIST) (COMPLETED).
- [x] **Phase MEDICAL-02**: Centralized Medical Small CNN Baseline (COMPLETED).
- [x] **Phase 3**: **10-Hospital Federated Learning Implementation (Flower + FedAvg, IID baseline)** (COMPLETED).
- [x] **Phase 4**: **Non-IID Hospital Data Heterogeneity Simulation (Dirichlet case-mix skew)** (COMPLETED).
- [x] **Phase 5**: **Malicious Hospital Attack Simulation (Label flipping, poisoning, backdoors)** (COMPLETED).
- [x] **Phase 7 Technical Foundation**: **Byzantine-Robust Aggregation, Memory-Safe Execution Engine & Blockchain Audit Integration** (COMPLETED).
- [ ] **Phase 7 Full Matrix Execution**: **351-Simulation Execution Suite across Seeds 211, 212, 213** — **ACTIVE / IN PROGRESS**.
- [ ] **Phase 8**: **Decentralized Multi-Node Blockchain Anchoring** — **FUTURE WORK**.
- [ ] **Phase 9**: **Empirical Evaluation & Comparative Ablations** — **FUTURE WORK**.
- [ ] **Phase 10**: **Publication Artifacts & Final Report** — **FUTURE WORK**.

---

## 4. Scope Boundary Enforcement
- Phases 1 through 6 are fully implemented, audited, and empirically verified.
- Phase 7 memory-safe sequential runner, persistent registry, and lightweight permissioned blockchain audit layer are fully operational.
- The authoritative 351-simulation matrix manifest (`PHASE7_EXPERIMENT_MATRIX.json`) remains completely unchanged.
- The frozen Phase 6 tri-state detector v1 remains strictly read-only and untouched.
- Decentralized multi-node public blockchain smart contracts remain strictly deferred to Phase 8.
