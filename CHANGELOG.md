# Changelog

All notable changes to the **Blockchain-Assisted Non-IID-Aware Malicious Participant Detection in Federated Learning** research project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### [0.1.0] - 2026-10-07 (Phase 1 Initialization)
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
- Created Python 3.14 virtual environment `.venv/`.

#### Reason
Initial scaffolding and research foundation setup for Phase 1 as approved by user.
