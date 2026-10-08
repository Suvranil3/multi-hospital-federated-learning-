# Research Roadmap & Phase Progression

## 1. Overview of Phased Progression

This research project follows a strict, disciplined phased methodology. Each phase must be verified, tested, and documented with empirical logs before proceeding to subsequent stages.

```text
[✓] Phase MEDICAL-01: PneumoniaMNIST Dataset Verification       (COMPLETED)
           │
           ▼
[✓] Phase MEDICAL-02: Centralized Medical Small CNN Baseline     (COMPLETED - 81.25% Acc, 99.74% Sens)
           │
           ▼
[→] Phase 3: 10-Hospital Federated Learning Baseline (IID)      (NEXT IMMEDIATE STEP)
           │
           ▼
[○] Phase 4: Non-IID Hospital Case-Mix Heterogeneity             (LATER RESEARCH FOCUS)
           │
           ▼
[○] Phase 5: Adversarial Attacks (Label Flip, Poisoning)        (FUTURE WORK)
           │
           ▼
[○] Phase 6: Multi-Factor Non-IID-Aware Detector                 (FUTURE WORK)
           │
           ▼
[○] Phase 7: Robust Aggregation & Tri-State Consensus            (FUTURE WORK)
           │
           ▼
[○] Phase 8: Blockchain Tamper-Evident Medical Audit Trail       (FUTURE WORK)
           │
           ▼
[○] Phase 9: Comprehensive Empirical Evaluation & Ablations      (FUTURE WORK)
           │
           ▼
[○] Phase 10: Publication Artifacts & Final Report               (FUTURE WORK)
```

---

## 2. Phase-by-Phase Detail

### Phase MEDICAL-01: Medical Dataset Verification (COMPLETED)
- **Objective**: Verify download, format, sample counts, and statistical properties of PneumoniaMNIST.
- **Key Deliverables**:
  - Verification script: [`data/verify_dataset.py`](file:///home/suvranil/project/data/verify_dataset.py).
  - Empirical figures: [`results/figures/pneumonia_sample_grid.png`](file:///home/suvranil/project/results/figures/pneumonia_sample_grid.png), [`results/figures/pneumonia_class_distribution.png`](file:///home/suvranil/project/results/figures/pneumonia_class_distribution.png).
  - Machine-readable report: [`results/logs/pneumonia_mnist_verification_report.json`](file:///home/suvranil/project/results/logs/pneumonia_mnist_verification_report.json).
  - Status: 5,856 images verified; exact official splits confirmed (4,708 train, 524 val, 624 test).

### Phase MEDICAL-02: Centralized Medical Baseline (COMPLETED)
- **Objective**: Establish centralized SmallCNN benchmark on GPU before distributing across hospitals.
- **Key Deliverables**:
  - Model definition: [`federated_learning/model.py`](file:///home/suvranil/project/federated_learning/model.py#L21-L71) (420,802 parameters).
  - Training script: [`experiments/train_medical.py`](file:///home/suvranil/project/experiments/train_medical.py).
  - Empirical figures: [`results/figures/medical_training_loss.png`](file:///home/suvranil/project/results/figures/medical_training_loss.png), [`results/figures/medical_training_accuracy.png`](file:///home/suvranil/project/results/figures/medical_training_accuracy.png), [`results/figures/medical_confusion_matrix.png`](file:///home/suvranil/project/results/figures/medical_confusion_matrix.png).
  - Machine-readable report: [`results/logs/medical_centralized_results.json`](file:///home/suvranil/project/results/logs/medical_centralized_results.json).
  - Status: 81.25% test accuracy, 99.74% pneumonia sensitivity, 77.03% precision, 0.8693 F1 on untouched test set.

---

### Phase 3: 10-Hospital Federated Learning Baseline (IID) — NEXT STEP
- **Status**: **NOT STARTED (Awaiting explicit instruction)**
- **Objective**: Distribute the verified [`SmallCNN`](file:///home/suvranil/project/federated_learning/model.py#L21-L71) across 10 simulated hospitals using the Flower framework (`flwr[simulation]`) and FedAvg under an IID data split.
- **Why IID First?**: Establishes the ideal distributed benchmark where all hospitals draw from an identical data distribution. This provides the control baseline against which non-IID performance drops (Phase 4) and attack impacts (Phase 5) are measured.
- **Planned Configuration**:
  - Framework: Flower (`flwr[simulation]`) with Ray backend.
  - Clients: 10 simulated hospitals.
  - Partitioning: Uniform IID distribution of training data across hospitals (~470 samples/hospital).
  - Aggregation Strategy: Standard FedAvg.
  - Communication Rounds: 20 rounds, 2 local epochs per round.

---

### Phase 4: Non-IID Hospital Data Heterogeneity Simulation — LATER RESEARCH FOCUS
- **Status**: **PLANNED**
- **Objective**: Model realistic clinical case-mix divergence across hospitals using Dirichlet label partitioning.
- **Heterogeneity Severity**:
  - $\alpha = 1.0$: Mild non-IID (moderate variation in pneumonia rates).
  - $\alpha = 0.5$: Moderate non-IID (significant clinical specialization).
  - $\alpha = 0.1$: Extreme non-IID (hospitals specialized almost entirely in one class).
- **Core Study**: Quantify convergence slowdown and accuracy degradation in standard FedAvg caused purely by benign clinical divergence.

---

### Phase 5: Adversarial Attacks Simulation — FUTURE WORK
- **Status**: **PLANNED**
- **Objective**: Simulate compromised or malicious hospital nodes mounting healthcare-targeted attacks.
- **Attacks Investigated**:
  1. **Label Flipping**: Flipping pneumonia scans to normal, aiming to cause dangerous missed diagnoses.
  2. **Gradient Scaling / Model Poisoning**: Submitting scaled, inverted, or random updates to disrupt global convergence.
  3. **Backdoor Injection**: Embedding stealthy triggers that force misclassification only on specific scan patterns.

---

### Phase 6: Multi-Factor Non-IID-Aware Malicious Hospital Detector — FUTURE WORK
- **Status**: **PLANNED**
- **Objective**: Implement the core research contribution—an intelligent security layer evaluating updates on multiple orthogonal criteria:
  - $L_2$ update deviation.
  - Directional cosine similarity.
  - Validation impact on benchmark proxy data.
  - Historical trajectory consistency across rounds.
  - Hospital clinical heterogeneity context.
- **Decision Policy**: Tri-state classification (`Normal`, `Suspicious`, `Malicious`).

---

### Phase 7: Robust Aggregation & Tri-State Consensus — FUTURE WORK
- **Status**: **PLANNED**
- **Objective**: Combine non-rejected hospital updates using weighted aggregation and compare performance against classical Byzantine-robust aggregators (Multi-Krum, Trimmed Mean, Coordinate Median).
- **Primary Metric**: Reduction in False Positive Rate (FPR) on honest specialized hospitals while maintaining high Attack Detection Recall.

---

### Phase 8: Blockchain Tamper-Evident Medical Audit Trail — FUTURE WORK
- **Status**: **PLANNED**
- **Objective**: Deploy a lightweight, local Ethereum-compatible smart contract to log SHA-256 update hashes, hospital IDs, and detector classifications for clinical regulatory accountability.
- **Rule**: Zero patient images and zero neural network weights on-chain.

---

### Phase 9: Comprehensive Empirical Evaluation & Ablation — FUTURE WORK
- **Status**: **PLANNED**
- **Objective**: Run exhaustive multi-seed ablation experiments evaluating each detector component individually across varying non-IID levels ($\alpha \in \{1.0, 0.5, 0.1\}$) and attack percentages ($10\%, 20\%, 30\%$).

---

### Phase 10: Publication Artifacts & Final Report — FUTURE WORK
- **Status**: **PLANNED**
- **Objective**: Synthesize empirical findings into final project thesis, publication-ready figures, and academic defense slide deck.
