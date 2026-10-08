# Architectural & Technical Decision Records (ADR)

## Project: Multi-Hospital Federated Learning for Chest X-Ray Classification with Malicious Participant Detection

---

### ADR-001: Selection of CIFAR-10 Dataset *(Superseded / Archived)*
- **Status**: Superseded by ADR-009
- **Context**: Initial generic proof-of-concept benchmark.
- **Decision**: Used CIFAR-10 as generic baseline in Phase 1 & Phase 2.
- **Note**: Preserved in full under `results/archive/cifar10/`.

---

### ADR-002: Selection of Small CNN Model Architecture
- **Status**: Modified by ADR-010
- **Context**: The model must balance representation power with training speed across 10 simulated federated clients.
- **Decision**: Use a compact Convolutional Neural Network (SmallCNN) with modular blocks.

---

### ADR-003: Selection of Flower (`flwr[simulation]`) Framework
- **Status**: Accepted
- **Context**: We require an FL framework capable of orchestrating multi-hospital client simulations natively in Python and PyTorch.
- **Decision**: Use Flower with the simulation engine (`flwr[simulation]`).
- **Rationale**:
  - Native Python/PyTorch support with minimal boilerplate.
  - Highly efficient resource virtualization enabling 10 simulated hospitals on a single GPU/workstation.
  - Clean separation between client compute and custom server aggregation strategies (`Strategy` interface).

---

### ADR-004: Centrality of Dirichlet Non-IID Partitioning
- **Status**: Accepted
- **Context**: Real-world hospitals treat different patient populations, leading to natural case-mix heterogeneity.
- **Decision**: Parameterize hospital data heterogeneity using a Dirichlet distribution over label distributions with concentration parameter $\alpha \in \{1.0, 0.5, 0.1\}$.
- **Rationale**:
  - Gold standard in federated learning research for simulating statistical heterogeneity.
  - Models the clinical reality where specialized clinics (e.g., pediatric pulmonary centers) treat distinct proportions of conditions compared to community hospitals.

---

### ADR-005: Decoupled Blockchain Role: Audit Layer Only
- **Status**: Accepted
- **Context**: Blockchain is frequently misapplied in FL research as a computational engine or storage layer.
- **Decision**:
  - Storing complete model weights on-chain is **strictly prohibited**.
  - Storing raw patient images or Protected Health Information (PHI) on-chain is **strictly prohibited**.
  - Executing ML detection algorithms inside smart contracts is **strictly prohibited**.
  - Blockchain serves exclusively as an immutable, tamper-evident audit log for round metadata, hospital IDs, SHA-256 update hashes, detector classification decisions, and actions taken.
- **Rationale**: Gas costs, storage bloat, and healthcare privacy compliance require keeping ML and medical data off-chain.

---

### ADR-006: Tri-State Classification Policy (Normal / Suspicious / Malicious)
- **Status**: Accepted
- **Context**: Binary classification (accept vs. reject) often penalizes honest hospitals with naturally specialized case mixes whose updates marginally cross outlier thresholds.
- **Decision**: Adopt a tri-state classification:
  - `Normal`: Fully accepted (weight = 1.0).
  - `Suspicious`: Down-weighted proportionally to anomaly severity ($0 < \lambda < 1$).
  - `Malicious`: Completely rejected (weight = 0.0).

---

### ADR-007: Phased Modular Development Strategy
- **Status**: Accepted
- **Context**: Complex multi-component systems (FL + attacks + detector + blockchain) fail easily if built monolithically.
- **Decision**: Implement the system strictly across sequential verified phases.

---

### ADR-008: Centralized Small CNN Architecture on CIFAR-10 *(Archived)*
- **Status**: Archived
- **Context**: Completed Phase 1 and Phase 2 generic baseline (76.94% accuracy). Results archived in `results/archive/cifar10/`.

---

### ADR-009: Migration to Hospital Scenario and PneumoniaMNIST Dataset
- **Status**: Accepted
- **Context**: The professor clarified that the complete project must be developed and presented in a hospital/medical scenario.
- **Decision**: Replace CIFAR-10 with **PneumoniaMNIST** (from MedMNIST v3, 5,856 pediatric chest X-rays, 28x28 grayscale, Normal vs. Pneumonia) as the primary project dataset.
- **Rationale**:
  - Provides a clinically meaningful problem (Chest X-Ray pneumonia classification) well-suited for multi-hospital federated learning.
  - Enables studying how non-IID hospital case mixes interact with Byzantine poisoning attacks.
  - Directly fulfills the academic requirement for a hospital/medical scenario.

---

### ADR-010: Adaptation of SmallCNN for 1x28x28 Grayscale Chest X-Rays
- **Status**: Accepted
- **Context**: PneumoniaMNIST consists of 1-channel grayscale 28x28 images with a 2-class diagnosis head (0: Normal, 1: Pneumonia).
- **Decision**:
  - Update `SmallCNN` in `federated_learning/model.py`:
    - Input: `1 x 28 x 28`
    - Conv Block 1: `Conv2d(1 -> 32, 3x3)`, `BatchNorm2d(32)`, `ReLU`, `MaxPool2d(2x2)` -> `32 x 14 x 14`
    - Conv Block 2: `Conv2d(32 -> 64, 3x3)`, `BatchNorm2d(64)`, `ReLU`, `MaxPool2d(2x2)` -> `64 x 7 x 7`
    - Flatten: `3,136` features
    - Dense Head: `Linear(3136 -> 128)`, `ReLU`, `Dropout(0.2)`, `Linear(128 -> 2)`
    - Total parameter count: **420,802 parameters** (~1.61 MB).
- **Rationale**: Modular, fast to train (~13s on GPU), lightweight for 10-hospital simulation, with excellent feature capacity for chest X-ray textures.

---

### ADR-011: Preservation of Official MedMNIST Splits (No Artificial Re-Splitting)
- **Status**: Accepted
- **Context**: Unlike CIFAR-10, PneumoniaMNIST provides an official, clinically designed benchmark split: 4,708 train, 524 validation, and 624 test images.
- **Decision**: Use the official train split for training, the official validation split for model checkpoint selection, and keep the official test split strictly untouched until final evaluation. Do NOT create arbitrary random splits.
- **Rationale**: Aligns with standard MedMNIST benchmarks and ensures clinical validation integrity.

---

### ADR-012: Comprehensive Medical Evaluation Metrics (Accuracy, Precision, Recall, F1)
- **Status**: Accepted
- **Context**: In medical classification (especially with class imbalance: 390 pneumonia vs. 234 normal in the test set), accuracy alone is misleading because a trivial classifier could predict pneumonia for all patients and achieve ~62.5% accuracy.
- **Decision**: Mandatory reporting of:
  - Accuracy
  - Pneumonia Precision (Positive Predictive Value)
  - Pneumonia Recall / Sensitivity (True Positive Rate)
  - Pneumonia F1-Score
  - Macro-averaged F1-Score
  - 2x2 Confusion Matrix (TN, FP, FN, TP)
  - Clear medical disclaimer noting these are research benchmark metrics.
- **Rationale**: Ensures clinical evaluation rigor and prevents false conclusions based solely on accuracy.

---

### ADR-013: Archival of Historical CIFAR-10 Proof-of-Concept Work
- **Status**: Accepted
- **Context**: Phases 1 and 2 were previously completed on CIFAR-10. We must preserve this work without mixing it with medical data.
- **Decision**: Archive all CIFAR-10 artifacts into `results/archive/cifar10/` (figures, logs, models, README).
- **Rationale**: Preserves proof of progress while maintaining strict separation between generic and medical results.
