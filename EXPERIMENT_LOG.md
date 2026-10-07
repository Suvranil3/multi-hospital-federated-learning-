# Experiment Log

This document records every empirical experiment conducted in the project.
No entries may contain simulated, guessed, or fabricated numbers. All metrics must strictly originate from recorded experiment execution logs.

---

## Log Format Schema
Each experiment must record:
- **Experiment ID**: Standardized identifier (e.g., `EXP-001-CENTRAL-CNN`, `EXP-002-FL-IID-FEDAVG`)
- **Date**: Execution timestamp
- **Dataset Configuration**: Dataset name, sample counts, splits
- **Number of Clients**: Total participating edge clients
- **IID / Non-IID Setting**: Data distribution type
- **Dirichlet Alpha**: Heterogeneity concentration parameter ($\alpha$) or N/A
- **Attack Type**: Injected attack or None
- **Malicious Clients**: Count and percentage of adversaries
- **Model Architecture**: Neural network definition
- **FL Algorithm**: Federated learning framework / protocol
- **Detector Version**: Detection algorithm version or None
- **Aggregation Method**: Model update aggregation rule
- **Random Seed**: Master seed used for run
- **Results**: Quantitative metrics (Accuracy, Loss, FPR, FNR, ASR, etc.)
- **Execution Status**: Succeeded or Failed
- **Notes**: Key observations and reproducibility pointers

---

## Experiment History

### [EXP-000-DATA] - CIFAR-10 Dataset Verification & Statistics
- **Date**: 2026-10-07
- **Dataset Configuration**: CIFAR-10 (raw download from torchvision official mirror)
- **Number of Clients**: N/A (Phase 1 Central Dataset Verification)
- **IID / Non-IID Setting**: N/A (Unpartitioned global dataset)
- **Dirichlet Alpha**: N/A (Partitioning scheduled for Phase 4)
- **Attack Type**: None
- **Malicious Clients**: 0 (0%)
- **Model Architecture**: None (Data verification only)
- **FL Algorithm**: None
- **Detector Version**: None
- **Aggregation Method**: None
- **Random Seed**: 42
- **Results**:
  - Training Samples: [Pending measurement]
  - Test Samples: [Pending measurement]
  - Classes: 10
  - Images Shape: [Pending measurement]
  - Class Distribution: [Pending measurement]
  - Channel Means (R, G, B): [Pending measurement]
  - Channel Stds (R, G, B): [Pending measurement]
- **Execution Status**: IN PROGRESS (Phase 1)
- **Notes**: Dataset verification pipeline script `data/verify_dataset.py` executing.
