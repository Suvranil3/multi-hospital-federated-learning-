# Project Status

## Project Title
**Multi-Hospital Federated Learning for Chest X-Ray Classification with Malicious Participant Detection**

---

## Current Status Overview
- **Active Primary Dataset**: **PneumoniaMNIST** (Pediatric Chest X-Rays from MedMNIST v3)
- **Problem Scenario**: **Hospital-based simulated Federated Learning** across 10 hospital clients
- **Current Model**: **SmallCNN** (2 Conv blocks, BatchNorm, MaxPool, Dense 128, 2-class output; **420,802 parameters**)
- **Current Milestone**: **Medical Centralized Baseline Complete** (Phase MEDICAL-01 and Phase MEDICAL-02)
- **Immediate Next Step**: **Phase 3 — 10-Client IID Federated Learning Baseline** (Flower + FedAvg)
- **Later Research Focus**: **Phase 4 — Non-IID Hospital Data Heterogeneity** (Dirichlet case-mix skew $\alpha \in \{1.0, 0.5, 0.1\}$)
- **Future Work**: Adversarial attacks (Phase 5), multi-factor malicious detector (Phase 6), robust aggregation (Phase 7), and blockchain audit trail (Phase 8)
- **Status Date**: October 8, 2026
- **Lead Developer**: AI Research Pair Programmer
- **Strict Scope Boundary**: Medical centralized baseline complete; Federated Learning, Non-IID skew, attacks, detection, and blockchain strictly NOT started yet.

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

### Quality, Tests & Dashboard
- [x] Automated test suite in `tests/`: **14 / 14 PyTest tests passing (100%)** in 3.20 seconds.
- [x] Updated interactive teacher demonstration dashboard ([`dashboard/app.py`](file:///home/suvranil/project/dashboard/app.py)) with the multi-hospital story, presentation script, metrics scorecard, medical disclaimer, and an archived CIFAR-10 viewer.

---

## 3. What Remains (Upcoming Phases)
- [x] **Phase MEDICAL-01**: Medical Dataset Verification (PneumoniaMNIST) (COMPLETED).
- [x] **Phase MEDICAL-02**: Centralized Medical Small CNN Baseline (COMPLETED).
- [ ] **Phase 3**: **10-Hospital Federated Learning Implementation (Flower + FedAvg, IID baseline)** — **NEXT IMMEDIATE STEP**.
- [ ] **Phase 4**: **Non-IID Hospital Data Heterogeneity Simulation (Dirichlet case-mix skew)** — **LATER RESEARCH FOCUS**.
- [ ] **Phase 5**: **Malicious Hospital Attack Simulation (Label flipping, poisoning, backdoors)** — **FUTURE WORK**.
- [ ] **Phase 6**: **Multi-Factor Non-IID-Aware Malicious Hospital Detector** — **FUTURE WORK**.
- [ ] **Phase 7**: **Robust Aggregation & Tri-State Consensus** — **FUTURE WORK**.
- [ ] **Phase 8**: **Blockchain Tamper-Evident Medical Audit Trail** — **FUTURE WORK**.
- [ ] **Phase 9**: **Empirical Evaluation & Comparative Ablations** — **FUTURE WORK**.
- [ ] **Phase 10**: **Publication Artifacts & Final Report** — **FUTURE WORK**.

---

## 4. Scope Boundary Enforcement
- Federated Learning (Phase 3) is **NOT** started yet.
- Non-IID partitioning (Phase 4) is **NOT** started yet.
- Malicious attacks and detection (Phases 5-6) are **NOT** started yet.
- Blockchain smart contracts (Phase 8) are **NOT** started yet.
- Phase 3 will begin strictly upon explicit user instruction.
