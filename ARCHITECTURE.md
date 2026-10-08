# System Architecture

## Project Title
**Multi-Hospital Federated Learning for Chest X-Ray Classification with Malicious Participant Detection and Blockchain Audit**

---

## 1. High-Level Architectural Overview
The system is structured into four decoupled, modular subsystems designed for clinical environments:

1. **Hospital Simulation Subsystem**: Manages local patient chest X-ray data loading (PneumoniaMNIST), hospital case-mix heterogeneity partitioning (Dirichlet non-IID), and local SmallCNN training on 10 simulated hospitals.
2. **Federated Orchestration Subsystem**: Manages Flower FL rounds, model weight distribution, and secure aggregation.
3. **Non-IID-Aware Malicious Participant Detector**: An intelligent security layer evaluating hospital gradient/weight updates using multiple orthogonal features (deviation, directional cosine similarity, validation impact, historical behavioral trajectory, and clinical heterogeneity context) to output a tri-state classification (`Normal`, `Suspicious`, `Malicious`).
4. **Blockchain Audit Trail Subsystem**: A tamper-evident, append-only Ethereum-compatible ledger recording lightweight cryptographic hashes and clinical governance metadata.

```text
       +-------------------------------------------------------------+
       |               Simulated Hospital Clients (10 Nodes)         |
       |  [Hospital 1]  [Hospital 2] ... [Hospital 9]  [Hospital 10] |
       |  (Pediatric)   (Pulmonary)       (Compromised) (Community)  |
       |  [Local X-Rays][Local X-Rays]   [Poisoned]    [Local X-Rays]|
       +-------------------------------------------------------------+
                                      |
                                      | Hospital Model Updates (Weights / Gradients)
                                      v
       +-------------------------------------------------------------+
       |             FL Server Orchestrator (Flower)                 |
       |  - Collects local hospital model updates                    |
       |  - Preserves patient confidentiality (Zero image transfer)  |
       +-------------------------------------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |      Non-IID-Aware Malicious Participant Detector           |
       |                                                             |
       |  [Feature 1: Update Deviation (L2 Norm)]                    |
       |  [Feature 2: Update Similarity (Cosine Similarity)]         |
       |  [Feature 3: Validation Impact (Loss/Acc Delta on Bench)]   |
       |  [Feature 4: Hospital Historical Trajectory]                |
       |  [Feature 5: Clinical Heterogeneity Context]                |
       |                                                             |
       |  Output: Classification Decision & Aggregation Weight       |
       |    - NORMAL     -> Weight = 1.0 (Accepted)                  |
       |    - SUSPICIOUS -> Weight = lambda_down (Down-weighted)     |
       |    - MALICIOUS  -> Weight = 0.0 (Rejected)                  |
       +-------------------------------------------------------------+
               |                                             |
               | Aggregated Weights                          | Audit Metadata
               v                                             v
+-------------------------------+             +-------------------------------+
|     Global Medical Model      |             |   Blockchain Audit Ledger     |
|   (Weighted Aggregation)      |             |   - Round ID, Hospital ID     |
|                               |             |   - SHA-256(Model Update)     |
| -> Broadcast to Next Round    |             |   - Decision (Normal/Susp/Mal)|
|                               |             |   - Action Taken, Timestamp   |
+-------------------------------+             +-------------------------------+
```

---

## 2. Component Specifications

### 2.1. Hospital Client Simulation Tier
- **Framework**: Flower (`flwr[simulation]`) with PyTorch.
- **Participants**: 10 simulated hospitals.
- **Dataset**: PneumoniaMNIST (5,856 pediatric chest X-rays, 1x28x28 grayscale, Normal vs. Pneumonia).
- **Partitioning Strategy (Future Phase 4)**: Dirichlet distribution ($\alpha \in \{1.0, 0.5, 0.1\}$) simulating realistic clinical case-mix differences among hospitals.
- **Model**: `SmallCNN` (2 Convolutional blocks with BatchNorm and MaxPool, Flatten, Dense 128, 2-class output, **420,802 parameters**).

### 2.2. Non-IID-Aware Malicious Hospital Detector
Evaluates hospital updates prior to aggregation:
- **Multi-Feature Analysis**:
  1. **Update Deviation ($L_2$ norm)**: Measures distance relative to median and mean trajectories.
  2. **Directional Similarity (Cosine similarity)**: Identifies adversarial alignment vs. benign clinical directional variance.
  3. **Validation Impact**: Evaluates client update performance on a benchmark validation set to catch stealthy diagnostic degradation.
  4. **Behavioral History**: Tracks exponential moving averages across training rounds to distinguish temporary clinical variance from persistent adversarial attacks.
  5. **Clinical Heterogeneity Awareness**: Calibrates thresholds to account for legitimate hospital case-mix specialization, preventing false accusations against honest specialized clinics.
- **Tri-State Action Policy**:
  - `Normal`: Fully accepted ($w = 1.0$).
  - `Suspicious`: Down-weighted ($0 < w < 1.0$).
  - `Malicious`: Completely rejected ($w = 0.0$).

### 2.3. Robust Medical Aggregator
- Dynamically applies weights $w_i^{(t)}$ to combine non-rejected updates into the global medical model.

### 2.4. Blockchain Medical Audit Layer
- **Role**: Append-only, tamper-evident cryptographic log for healthcare compliance.
- **Stored On-Chain**: Round number, Hospital ID, SHA-256 update hash, decision enum, action weight, timestamp.
- **NOT Stored On-Chain**: Raw patient images, patient PHI, model weights, or machine learning execution.

---

## 3. Threat Model
- Up to 30% of hospitals (e.g., 3 out of 10) may be compromised by an adversary.
- Attacks tested include label flipping (flipping Pneumonia to Normal to cause diagnostic misses), gradient scaling, and backdoor injection.
