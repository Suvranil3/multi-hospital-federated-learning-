# Project Status

## Project Title
**Blockchain-Assisted Non-IID-Aware Malicious Participant Detection in Federated Learning**

---

## Current Status Overview
- **Current Phase**: Phase 2 — Centralized Small CNN Baseline on CIFAR-10 (COMPLETED)
- **Status Date**: October 8, 2026
- **Lead Developer**: AI Research Pair Programmer

---

## 1. What is Completed
- [x] Initialized Git repository at project root (`/home/suvranil/project`).
- [x] Created complete modular directory structure (`data/raw`, `data/processed`, `notebooks`, `federated_learning`, `detection`, `attacks`, `blockchain`, `experiments`, `results/figures`, `results/tables`, `results/logs`, `tests`).
- [x] Authored core research governance documentation:
  - `RESEARCH_OBJECTIVE.md` (Primary research question, hypotheses, scope boundaries)
  - `RESEARCH_GAP.md` (Investigation of Byzantine aggregation limits under non-IID skew)
  - `ARCHITECTURE.md` (Decoupled 4-tier design, dataflow, security model)
  - `DECISIONS.md` (ADR-001 through ADR-008)
  - `.gitignore` (Configured for Python, venv, and large dataset caches)
- [x] Configured centralized reproducibility settings in `experiments/config.py` (Deterministic seed 42, paths, constants).
- [x] Defined dependency specifications in `requirements.txt` with `flwr[simulation]`, PyTorch, torchvision, numpy, matplotlib, and pytest.
- [x] Installed dependencies in Python 3.14 virtual environment (`.venv`) (PyTorch 2.14.1+cu130, Torchvision 0.29.1+cu130, NumPy 2.5.3, Matplotlib 3.11.2, Flower 1.39.0, pytest 9.1.1).
- [x] Created and executed the CIFAR-10 dataset verification pipeline (`data/verify_dataset.py`).
- [x] Generated dataset visualizations (`results/figures/cifar10_sample_grid.png`, `results/figures/cifar10_class_distribution.png`).
- [x] Produced machine-readable verification summary (`results/logs/cifar10_verification_report.json`).
- [x] Implemented and passed automated test suite for data loading (`tests/test_data_loading.py` — 3/3 tests passed).
- [x] Recorded empirical dataset verification metrics in `EXPERIMENT_LOG.md` (EXP-000-DATA) and `data/README.md`.
- [x] Implemented reusable `SmallCNN` architecture (`federated_learning/model.py`, 189,130 parameters).
- [x] Built stratified train/validation split (40,000 train, 10,000 val) preserving untouched 10,000 test set.
- [x] Implemented centralized training pipeline in `experiments/train_centralized.py`.
- [x] Executed centralized benchmark training on CUDA (NVIDIA RTX 3050 Laptop GPU, 15 epochs, 100.97s):
  - Train Loss: 0.7310 | Train Accuracy: 74.50%
  - Best Validation Accuracy: 77.34% (Epoch 15, Val Loss: 0.6400)
  - Untouched Test Accuracy: 76.94% (Test Loss: 0.6578, 7,694/10,000 correct)
- [x] Generated training curve and evaluation figures:
  - `results/figures/centralized_training_loss.png`
  - `results/figures/centralized_training_accuracy.png`
  - `results/figures/confusion_matrix.png`
- [x] Emitted structured JSON experiment log: `results/logs/centralized_cnn_results.json`.
- [x] Implemented unit tests for model forward pass, shape, loss, optimization, and data splits (`tests/test_model.py`, `tests/test_training.py` — 13/13 tests passed).
- [x] Maintained strict scope boundary: zero implementation of FL rounds, non-IID skew, attacks, detection, or blockchain.

---

## 2. What is Currently Being Worked On
- [x] Phase 2 Centralized CNN Baseline completed.
- Standing by for user gate review and approval before beginning Phase 3.

---

## 3. What Remains (Subsequent Phases)
- [x] **Phase 1**: Dataset verification and environment scaffolding (COMPLETED).
- [x] **Phase 2**: Centralized Small CNN training baseline on CIFAR-10 (COMPLETED).
- **Phase 3**: Basic Federated Learning implementation (10 clients, FedAvg, IID baseline).
- **Phase 4**: Non-IID Federated Learning (Dirichlet partitioning with $\alpha \in \{1.0, 0.5, 0.1\}$).
- **Phase 5**: Attack simulation (Label flipping, model poisoning/scaling, backdoor).
- **Phase 6**: Modular Non-IID-aware detection mechanism implementation.
- **Phase 7**: Robust tri-state aggregation (Normal, Suspicious, Malicious) and comparison with Multi-Krum and Trimmed Mean.
- **Phase 8**: Local Ethereum-compatible blockchain audit layer and metadata logging.
- **Phase 9**: Comprehensive evaluation and ablation experiments.
- **Phase 10**: Research report, publication tables, and empirical analysis.

---

## 4. Known Issues & Constraints
- All subsequent phases strictly blocked until Phase 2 sign-off from user.
- GPU acceleration: CUDA 13.0 operational with NVIDIA GeForce RTX 3050 Laptop GPU.

---

## 5. Next Immediate Task
Phase 3: Basic Federated Learning implementation (10 clients, FedAvg, IID baseline) with Flower framework (awaiting explicit user instruction to proceed).

