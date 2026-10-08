# Project Status

## Project Title
**Multi-Hospital Federated Learning for Chest X-Ray Classification with Malicious Participant Detection**

---

## Current Status Overview
- **Current Phase**: Phase MEDICAL-02 — Centralized Medical Small CNN Baseline on PneumoniaMNIST (COMPLETED)
- **Status Date**: October 8, 2026
- **Lead Developer**: AI Research Pair Programmer
- **Research Boundary**: Centralized baseline complete; Federated Learning, Non-IID skew, attacks, detection, and blockchain strictly NOT started yet.

---

## 1. What is Completed

### Migration to Hospital Scenario
- [x] Created Git checkpoint branch `checkpoint/pre-medical-migration` preserving pre-migration state (`3f2e0ab`).
- [x] Designated **PneumoniaMNIST** (Pediatric Chest X-Rays) as the **Primary Medical Project Dataset**.
- [x] Safely archived all previous CIFAR-10 proof-of-concept work in `results/archive/cifar10/` (figures, logs, models, and documentation) without mixing with medical artifacts.
- [x] Installed `medmnist` (v3.0.2) and `scikit-learn` in the virtual environment and added them to `requirements.txt`.

### Phase MEDICAL-01: Dataset Verification
- [x] Verified automatic download and integrity of PneumoniaMNIST in `data/verify_dataset.py`.
- [x] Verified exact official split sizes: Train = 4,708, Val = 524, Test = 624 (Total = 5,856).
- [x] Verified image format: 1 × 28 × 28 grayscale.
- [x] Verified binary diagnosis labels: 0 = Normal (1,214 train, 135 val, 234 test), 1 = Pneumonia (3,494 train, 389 val, 390 test).
- [x] Computed empirical pixel statistics: min=0, max=255, mean=0.57192, std=0.16835.
- [x] Generated visual figures:
  - `results/figures/pneumonia_sample_grid.png`
  - `results/figures/pneumonia_class_distribution.png`
- [x] Emitted structured JSON verification log: `results/logs/pneumonia_mnist_verification_report.json`.

### Phase MEDICAL-02: Centralized Medical Small CNN Baseline
- [x] Adapted `SmallCNN` (`federated_learning/model.py`) for 1-channel 28x28 grayscale input and 2-class output (Normal vs Pneumonia).
- [x] Documented exact parameter count: **420,802 total and trainable parameters**.
- [x] Built centralized training pipeline in `experiments/train_medical.py` (and mirrored in `experiments/train_centralized.py`) using PyTorch, Adam optimizer, CrossEntropyLoss, and seed 42.
- [x] Preserved official MedMNIST train (4,708), val (524), and untouched test (624) splits without artificial re-splitting.
- [x] Trained model on NVIDIA RTX 3050 Laptop GPU (15 epochs, 13.35s):
  - Best Validation Accuracy: **97.90%** (Epoch 14, Val Loss: 0.0760)
  - Untouched Test Accuracy: **81.25%** (507 / 624 correct)
  - Pneumonia Sensitivity / Recall: **99.74%** (389 / 390 pneumonia cases correctly identified)
  - Pneumonia Precision: **77.03%**
  - Pneumonia F1-Score: **0.8693**
  - Macro F1-Score: **0.7689**
  - Confusion Matrix: TN=118, FP=116, FN=1, TP=389
- [x] Emitted structured JSON experiment logs: `results/logs/medical_centralized_results.json` and `results/logs/centralized_cnn_results.json`.
- [x] Generated visual figures:
  - `results/figures/medical_training_loss.png`
  - `results/figures/medical_training_accuracy.png`
  - `results/figures/medical_confusion_matrix.png`
- [x] Saved best model checkpoint: `results/models/best_centralized_medical_cnn.pt` (and alias `best_centralized_cnn.pt`).

### Quality, Tests & Dashboard
- [x] Updated automated test suite (`tests/test_data_loading.py`, `tests/test_model.py`, `tests/test_training.py`).
- [x] Executed PyTest: **14 / 14 tests passing** with 100% pass rate.
- [x] Updated interactive teacher demonstration dashboard (`dashboard/app.py`, `dashboard/demo/` modules) to tell the multi-hospital story with presentation script, metrics scorecard, medical disclaimer, and an archived CIFAR-10 view.

---

## 2. What Remains (Upcoming Phases)
- [x] **Phase MEDICAL-01**: Medical Dataset Verification (PneumoniaMNIST) (COMPLETED).
- [x] **Phase MEDICAL-02**: Centralized Medical Small CNN Baseline (COMPLETED).
- [ ] **Phase 3**: 10-Hospital Federated Learning Implementation (Flower + FedAvg, IID baseline across simulated hospitals).
- [ ] **Phase 4**: Non-IID Hospital Data Heterogeneity Simulation (Dirichlet case-mix skew across hospitals).
- [ ] **Phase 5**: Malicious Hospital Attack Simulation (Label flipping, poisoned updates, backdoors).
- [ ] **Phase 6**: Multi-Factor Non-IID-Aware Malicious Hospital Detector.
- [ ] **Phase 7**: Robust Aggregation & Tri-State Consensus.
- [ ] **Phase 8**: Blockchain Tamper-Evident Medical Audit Trail.
- [ ] **Phase 9**: Empirical Evaluation & Comparative Ablations.
- [ ] **Phase 10**: Publication Artifacts & Final Report.

---

## 3. Scope Boundary Enforcement
- Federated Learning is **NOT** started yet.
- Non-IID partitioning is **NOT** started yet.
- Malicious attacks and detection are **NOT** started yet.
- Blockchain smart contracts are **NOT** started yet.
- Next phase starts strictly upon user instruction.
