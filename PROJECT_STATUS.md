# Project Status

## Project Title
**Blockchain-Assisted Non-IID-Aware Malicious Participant Detection in Federated Learning**

---

## Current Status Overview
- **Current Phase**: Phase 1 — Environment Setup, Repository Scaffolding, & CIFAR-10 Dataset Verification (COMPLETED)
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
  - `DECISIONS.md` (ADR-001 through ADR-007)
  - `.gitignore` (Configured for Python, venv, and large dataset caches)
- [x] Configured centralized reproducibility settings in `experiments/config.py` (Deterministic seed 42, paths, constants).
- [x] Defined dependency specifications in `requirements.txt` with `flwr[simulation]`, PyTorch, torchvision, numpy, matplotlib, and pytest.
- [x] Installed dependencies in Python 3.14 virtual environment (`.venv`) (PyTorch 2.14.1+cu130, Torchvision 0.29.1+cu130, NumPy 2.5.3, Matplotlib 3.11.2, Flower 1.39.0, pytest 9.1.1).
- [x] Created and executed the CIFAR-10 dataset verification pipeline (`data/verify_dataset.py`).
- [x] Generated dataset visualizations (`results/figures/cifar10_sample_grid.png`, `results/figures/cifar10_class_distribution.png`).
- [x] Produced machine-readable verification summary (`results/logs/cifar10_verification_report.json`).
- [x] Created and executed automated test suite for data loading (`tests/test_data_loading.py` — 3/3 tests passed).
- [x] Recorded empirical dataset verification metrics in `EXPERIMENT_LOG.md` (EXP-000-DATA) and `data/README.md`.
- [x] Verified zero leakage of out-of-scope modules (no CNN training, FL, non-IID partitioning, attacks, detection, or blockchain prematurely implemented).

---

## 2. What is Currently Being Worked On
- [x] Phase 1 Final Verification Audit completed.
- Standing by for user gate review and approval before beginning Phase 2.

---

## 3. What Remains (Subsequent Phases)
- **Phase 2**: Centralized Small CNN training baseline on CIFAR-10.
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
- All subsequent phases strictly blocked until Phase 1 verification sign-off from user.
- GPU acceleration: CUDA 13.0 detected in PyTorch wheels; CPU fallback fully operational.

---

## 5. Next Immediate Task
Phase 2: Baseline Centralized CNN training on CIFAR-10 (awaiting explicit user instruction to proceed).
