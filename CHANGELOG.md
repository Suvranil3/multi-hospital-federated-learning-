# Changelog

All notable changes to the **Blockchain-Assisted Non-IID-Aware Malicious Participant Detection in Federated Learning** research project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### [0.1.0] - 2026-10-08 (Phase 1 Initialization & Verification)
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

