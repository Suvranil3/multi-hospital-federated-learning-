# Changelog

All notable changes to the **Blockchain-Assisted Non-IID-Aware Malicious Participant Detection in Federated Learning** research project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### [0.2.0] - 2026-10-08 (Phase 2 Centralized CNN Baseline)
#### Added
- Implemented reusable PyTorch model `SmallCNN` (`federated_learning/model.py`, 189,130 parameters):
  - 3 Conv2d-BatchNorm2d-ReLU-MaxPool2d blocks with Dropout regularization.
  - Linear classification head yielding 10 logits.
  - Parameter counting utility and forward-pass dummy batch sanity check.
- Added centralized baseline hyperparameters to `experiments/config.py` (epochs=15, batch_size=64, lr=0.001, val_ratio=0.2, weight_decay=1e-4).
- Implemented centralized training script `experiments/train_centralized.py`:
  - Deterministic stratified 80/20 train/val split (40,000 train, 10,000 val).
  - Preserved untouched 10,000 CIFAR-10 test set for final independent evaluation.
  - Model checkpointing based on best validation accuracy (`results/models/best_centralized_cnn.pt`).
- Trained model on CUDA (NVIDIA GeForce RTX 3050 Laptop GPU, 100.97 seconds):
  - Final Train Loss: 0.7310, Train Accuracy: 74.50%.
  - Best Validation Accuracy: 77.34% (Epoch 15, Val Loss: 0.6400).
  - Untouched Test Set Accuracy: 76.94% (Test Loss: 0.6578, 7,694/10,000 correct).
- Generated visualizations and reports:
  - `results/figures/centralized_training_loss.png`
  - `results/figures/centralized_training_accuracy.png`
  - `results/figures/confusion_matrix.png`
  - `results/logs/centralized_cnn_results.json`
- Implemented unit tests for model forward pass, output shapes, loss calculation, optimization, and dataset splits (`tests/test_model.py`, `tests/test_training.py` — 13/13 passing tests).
- Added `ADR-008` to `DECISIONS.md`.
- Upgraded demonstration dashboard into an Interactive Teacher Demonstration system (`dashboard/app.py`, `dashboard/demo/pipeline_state.py`, `dashboard/demo/dataset_demo.py`, `dashboard/demo/cnn_demo.py`, `dashboard/demo/future_stubs.py`, `dashboard/README.md`, `requirements-dashboard.txt`) featuring one-click presentation flow, optional live training execution, copyable teacher speech, and extensible future-phase plug-in stubs.



#### Reason
Establish a solid, reproducible centralized machine learning benchmark before introducing federated orchestration.


#### Added
- Initialized Git version control on `main` branch.
- Created core modular project directory structure:
  - `data/raw/`, `data/processed/`
  - `notebooks/`
  - `federated_learning/`
  - `detection/`
  - `attacks/`
  - `blockchain/`
  - `experiments/`
  - `results/figures/`, `results/tables/`, `results/logs/`
  - `tests/`
- Created core research governance and specification documentation:
  - `README.md`
  - `PROJECT_STATUS.md`
  - `RESEARCH_OBJECTIVE.md`
  - `RESEARCH_GAP.md`
  - `ARCHITECTURE.md`
  - `DECISIONS.md`
  - `CHANGELOG.md`
  - `EXPERIMENT_LOG.md`
  - `data/README.md`
- Created `.gitignore` ignoring virtual environments, caches, raw archives, and checkpoints.
- Created `requirements.txt` specifying PyTorch, torchvision, numpy, matplotlib, `flwr[simulation]`, and pytest.
- Created `experiments/config.py` defining central reproducibility constants, paths, and seed 42.
- Created Python 3.14 virtual environment `.venv/` and installed required packages:
  - Python 3.14.4, PyTorch 2.14.1+cu130, Torchvision 0.29.1+cu130, NumPy 2.5.3, Matplotlib 3.11.2, Flower 1.39.0, PyTest 9.1.1.
- Implemented and executed CIFAR-10 verification pipeline (`data/verify_dataset.py`):
  - Verified 50,000 train samples and 10,000 test samples.
  - Confirmed 10 mutually exclusive classes with exact uniform balance (5,000 train/class, 1,000 test/class).
  - Computed empirical RGB channel means `[0.49140, 0.48216, 0.44653]` and stds `[0.24703, 0.24349, 0.26159]`.
  - Generated visual assets: `results/figures/cifar10_sample_grid.png`, `results/figures/cifar10_class_distribution.png`.
  - Emitted machine-readable summary: `results/logs/cifar10_verification_report.json`.
- Implemented automated data loading test suite in `tests/test_data_loading.py` (3 unit tests, all passed).
- Confirmed zero premature implementation of downstream phases (no CNN training, FL, non-IID partitioning, attacks, detection, or blockchain).

#### Reason
Initial scaffolding, research governance, and comprehensive dataset verification for Phase 1.

