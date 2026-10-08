# Changelog

All notable changes to the **Multi-Hospital Federated Learning for Chest X-Ray Classification with Malicious Participant Detection** research project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### [0.3.0] - 2026-10-08 (Migration to Hospital Scenario & PneumoniaMNIST)
#### Added
- Created Git checkpoint branch `checkpoint/pre-medical-migration` before applying changes.
- Designated **PneumoniaMNIST** (Pediatric Chest X-Rays from MedMNIST v3) as the **Primary Medical Project Dataset**.
- Added `medmnist>=3.0.2` and `scikit-learn>=1.4.0` to `requirements.txt` and installed in `.venv`.
- Preserved historical CIFAR-10 Phase 1 & Phase 2 proof-of-concept results into `results/archive/cifar10/` (figures, logs, models, and archival README).
- Implemented medical dataset verification pipeline in `data/verify_dataset.py` (Phase MEDICAL-01):
  - Verified 4,708 train, 524 validation, and 624 test samples (5,856 total).
  - Verified 1 × 28 × 28 grayscale dimensions and binary labels (0: Normal, 1: Pneumonia).
  - Computed empirical statistics (min=0, max=255, mean=0.57192, std=0.16835).
  - Generated `results/figures/pneumonia_sample_grid.png` and `results/figures/pneumonia_class_distribution.png`.
  - Emitted `results/logs/pneumonia_mnist_verification_report.json`.
- Adapted `SmallCNN` in `federated_learning/model.py`:
  - 1-channel 28x28 grayscale input, 2-class output (Normal vs. Pneumonia).
  - 2 Conv-BN-ReLU-MaxPool blocks, Flatten (3,136), Dense 128 (Dropout 0.2), Linear head.
  - Parameter count: **420,802 parameters** (~1.61 MB).
- Implemented centralized medical baseline training in `experiments/train_medical.py` (Phase MEDICAL-02):
  - Trained on CUDA (NVIDIA RTX 3050 Laptop GPU, 15 epochs, 13.35s).
  - Best Validation Accuracy: 97.90% (Epoch 14).
  - Untouched Test Set Accuracy: **81.25%** (507 / 624).
  - Pneumonia Sensitivity / Recall: **99.74%** (389 / 390 cases detected).
  - Pneumonia Precision: **77.03%**.
  - Pneumonia F1-Score: **0.8693** | Macro F1-Score: **0.7689**.
  - Generated loss and accuracy curves and medical 2x2 confusion matrix.
  - Saved checkpoint `results/models/best_centralized_medical_cnn.pt` and report `results/logs/medical_centralized_results.json`.
- Updated PyTest unit test suite (`tests/test_data_loading.py`, `tests/test_model.py`, `tests/test_training.py` — **14/14 tests passing**).
- Updated interactive teacher dashboard (`dashboard/app.py`, `dashboard/demo/` modules) with multi-hospital scenario, metrics scorecard, medical disclaimer, and an archived CIFAR-10 view.
- Added ADR-009 through ADR-013 to `DECISIONS.md`.

#### Reason
Complete project alignment with the professor's requirement that the entire project be developed and presented in a hospital/medical scenario prior to beginning Federated Learning.

---

### [0.2.0] - 2026-10-08 (Phase 2 Centralized CNN Baseline - Generic Proof-of-Concept)
#### Added
- Implemented initial PyTorch `SmallCNN` (189,130 parameters) on CIFAR-10.
- Stratified 80/20 train/val split (40,000 train, 10,000 val) and 10,000 test set.
- Centralized benchmark on GPU: 76.94% test accuracy (now archived in `results/archive/cifar10/`).

---

### [0.1.0] - 2026-10-08 (Phase 1 CIFAR-10 Verification & Scaffolding)
#### Added
- Initialized Git repository, modular directory structure, research documentation, and dependency specifications.
- Verified CIFAR-10 dataset (now archived in `results/archive/cifar10/`).
