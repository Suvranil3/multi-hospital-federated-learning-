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
[✓] Phase 3: 10-Hospital Federated Learning Baseline (IID)      (COMPLETED - 85.74% Acc, 97.95% Sens)
           │
           ▼
[✓] Phase 4: Non-IID Hospital Case-Mix Heterogeneity             (COMPLETED - alpha in {1.0, 0.5, 0.1})
           │
           ▼
[✓] Phase 5: Adversarial Attacks (Label Flip, Poisoning)        (COMPLETED - 12 Attack Scenarios)
           │
           ▼
[✓] Phase 6: Multi-Factor Non-IID-Aware Detector                 (COMPLETED - AUROC 0.9814, 0.00% Hard FPR)
           │
           ▼
[✓] Phase 7: Robust Aggregation & Consensus Under Non-IID Skew   (COMPLETED & AUDITED - 72 Primary Cells)
           │
           ▼
[→] Phase 8: Research Consolidation & Paper Preparation          (CURRENT / PLANNED - Zero Experiments Run)
           │
           ▼
[○] Phase 9: Extended Ablations & Robust Buffer Auditing         (FUTURE WORK)
           │
           ▼
[○] Phase 10: Publication Artifacts & Final Defense Deck         (FUTURE WORK)
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

### Phase 3: 10-Hospital Federated Learning Baseline (IID) (COMPLETED)
- **Status**: **COMPLETED (Recorded in `EXP-MED-002-FED-IID`)**
- **Objective**: Distribute the verified [`SmallCNN`](file:///home/suvranil/project/federated_learning/model.py#L21-L71) across 10 simulated hospitals using the Flower framework (`flwr[simulation]`) and FedAvg under an IID data split.
- **Key Deliverables**:
  - Partitioning module: [`federated_learning/dataset.py`](file:///home/suvranil/project/federated_learning/dataset.py) (4,708 samples across 10 clients: 8x471, 2x470).
  - Hospital client module: [`federated_learning/client.py`](file:///home/suvranil/project/federated_learning/client.py).
  - Strategy & server eval module: [`federated_learning/strategy.py`](file:///home/suvranil/project/federated_learning/strategy.py).
  - Training pipeline: [`experiments/train_federated.py`](file:///home/suvranil/project/experiments/train_federated.py).
  - Empirical figures: [`results/figures/federated_iid_training_loss.png`](file:///home/suvranil/project/results/figures/federated_iid_training_loss.png), [`results/figures/federated_iid_training_accuracy.png`](file:///home/suvranil/project/results/figures/federated_iid_training_accuracy.png), [`results/figures/federated_iid_confusion_matrix.png`](file:///home/suvranil/project/results/figures/federated_iid_confusion_matrix.png), [`results/figures/centralized_vs_federated_comparison.png`](file:///home/suvranil/project/results/figures/centralized_vs_federated_comparison.png).
  - Checkpoints: [`results/models/best_federated_iid_model.pt`](file:///home/suvranil/project/results/models/best_federated_iid_model.pt) & [`results/models/final_federated_iid_model.pt`](file:///home/suvranil/project/results/models/final_federated_iid_model.pt).
  - Report: [`results/logs/federated_iid_results.json`](file:///home/suvranil/project/results/logs/federated_iid_results.json).
  - Measured Performance: **85.74% test accuracy** (+4.49% over centralized), **97.95% pneumonia sensitivity**, **82.51% precision**, **65.38% normal specificity**, **0.8352 macro F1**.

---

### Phase 4: Non-IID Hospital Data Heterogeneity Simulation (COMPLETED)
- **Status**: **COMPLETED (Recorded in `EXP-MED-003` to `EXP-MED-005`)**
- **Objective**: Model realistic clinical case-mix divergence across hospitals using population-calibrated Beta/Dirichlet case-mix partitioning with controlled client capacities.
- **Heterogeneity Configurations**:
  - $\alpha = 1.0$: Normal Ratio Variance = 0.0709 (Test Acc: 79.81%, Sens: 98.97%, Spec: 47.86%, Macro F1: 0.7498).
  - $\alpha = 0.5$: Normal Ratio Variance = 0.1382 (Test Acc: 80.45%, Sens: 98.46%, Spec: 50.43%, Macro F1: 0.7611).
  - $\alpha = 0.1$: Normal Ratio Variance = 0.1482 (Test Acc: 85.10%, Sens: 91.79%, Spec: 73.93%, Macro F1: 0.8366).
- **Key Deliverables**:
  - Partitioning algorithm: [`federated_learning/dataset.py`](file:///home/suvranil/project/federated_learning/dataset.py).
  - Training runner: [`experiments/train_federated_non_iid.py`](file:///home/suvranil/project/experiments/train_federated_non_iid.py).
  - Models: [`results/models/best_federated_non_iid_alpha*.pt`](file:///home/suvranil/project/results/models).
  - Figures: [`results/figures/iid_vs_non_iid_heterogeneity_comparison.png`](file:///home/suvranil/project/results/figures/iid_vs_non_iid_heterogeneity_comparison.png).
  - Logs: [`results/logs/phase4_heterogeneity_comparison.json`](file:///home/suvranil/project/results/logs/phase4_heterogeneity_comparison.json).

---

### Phase 5: Adversarial Attacks Simulation (COMPLETED)
- **Status**: **COMPLETED (Recorded in `EXP-MED-006` through `EXP-MED-014`)**
- **Objective**: Simulate compromised or malicious hospital nodes mounting healthcare-targeted attacks under standard FedAvg.
- **Attacks Investigated & Key Findings**:
  1. **Targeted Label Flipping ($1 \to 0$)**:
     - At $M=1$ (10%), False Negatives doubled from 7 to 14 ($\Delta \text{FN} = +7$).
     - At $M=2$ (20%), False Negatives surged from 7 to 60 ($\Delta \text{FN} = +53$, $+757\%$, 8.57x jump in missed diagnoses), dropping sensitivity to 84.62%.
     - In Non-IID $\alpha=0.5$, False Negatives increased from 4 to 11 ($\Delta \text{FN} = +7$).
  2. **Model Update Poisoning**:
     - Inversion ($\gamma=-1.0$) slightly degraded performance (FN=8 vs control FN=7) but was partially damped by the 9 honest clients in unweighted FedAvg.
     - Scaling ($\gamma=+10.0$) distorted convergence, increasing FN from 7 to 9.
  3. **Backdoor Trigger Injection ($\rho=0.50$, $3\times3$ white watermark)**:
     - Preserved clean test accuracy (CDA: 86.06% in IID, 84.46% in Non-IID).
     - In IID: Triggered ASR reached 7.44% vs 0.51% control ($\Delta \text{ASR} = \mathbf{+6.93\%}$, 14.6x jump).
     - In Non-IID $\alpha=0.5$: Triggered ASR reached 8.46% vs 0.26% control ($\Delta \text{ASR} = \mathbf{+8.20\%}$, 32.5x jump).
     - Confirmed that clinical data heterogeneity provides natural camouflage that amplifies backdoor vulnerability.

---

### Phase 6: Multi-Factor Non-IID-Aware Malicious Hospital Detector — COMPLETED & VALIDATED
- **Status**: **COMPLETED & EMPIRICALLY AUDITED (October 31, 2026)**
- **Objective**: Implement the core research contribution—an intelligent security layer evaluating updates on multiple orthogonal criteria:
  - Global and layer-wise update magnitude ($L_2$ norm and Modified Z-Score).
  - Directional cosine similarity against robust coordinate-wise median reference.
  - Class-disentangled leave-one-out counterfactual proxy validation impact (Normal vs Pneumonia).
  - Symmetrical clinical trade-off protection for specialized hospitals.
  - Historical round-to-round trajectory consistency and EMA anomaly accumulator.
  - Calibrated tri-state decision policy (`NORMAL`, `SUSPICIOUS`, `MALICIOUS`).
- **Key Empirical Results**:
  - **Calibration-Excluded Benchmark (Out-of-Sample, 9 Simulations, 1,800 Obs: 1,640 honest, 160 malicious)**:
    - Continuous discrimination: **AUROC = 0.9814**, **AUPRC = 0.8801**
    - Binary Malicious Decision: **TPR = 13.13%**, **FPR = 0.00%** (0 / 1,640), **Precision = 100.00%**, **F1 = 23.20%**
    - Flagged Decision (`SUSPICIOUS` or `MALICIOUS`): **TPR = 96.25%** (154 / 160), **FPR = 10.00%** (164 / 1,640), **Precision = 48.43%**, **F1 = 64.44%**
    - Stratified False Malicious Rate across all data heterogeneities: **0.00%** in IID (0/880), **0.00%** in $\alpha=1.0$ (0/200), **0.00%** in $\alpha=0.5$ (0/360), **0.00%** in $\alpha=0.1$ (0/200).
  - **Independent-Seed Honest Evaluation Suite (Seeds 43, 44, 45 across IID and Non-IID; 2,400 Obs)**:
    - Aggregate False Malicious Rate: **0.04%** (1 / 2,400 across all 12 simulations).
  - **Forensic Non-IID Negative Cosine Audit (3,200 Honest Observations)**:
    - 199 negative cosine events observed (up to 13.5% in $\alpha=0.1$). Symmetrical Clinical Trade-Off Protection protected **100% (199 / 199)** from false malicious classifications, designating them as `SUSPICIOUS` for weighted aggregation.
  - **Instrumentation Parity Proof**:
    - Bit-for-bit exact mathematical equivalence verified ($0.000$ diff across all 12 trainable tensors and 6 BatchNorm buffers).
- **Artifacts**: Code: [`federated_learning/detection/`](file:///home/suvranil/project/federated_learning/detection), Parity Test: [`tests/test_instrumentation_parity.py`](file:///home/suvranil/project/tests/test_instrumentation_parity.py), Report: [`results/phase6_detector/evaluation/detection_results.json`](file:///home/suvranil/project/results/phase6_detector/evaluation/detection_results.json), Figure: [`results/phase6_detector/figures/phase6_detector_performance.png`](file:///home/suvranil/project/results/phase6_detector/figures/phase6_detector_performance.png)

---

### Phase 7: Robust Aggregation & Consensus Under Non-IID Skew — COMPLETED & AUDITED
- **Status**: **COMPLETED & AUDITED (November 17, 2026)**
- **Objective**: Evaluate Byzantine-robust aggregation strategies (Standard FedAvg, Multi-Krum $f=1$, Tri-State Monitor, Tri-State Discount $\lambda=0.50$) across varying clinical Dirichlet heterogeneity ($\alpha \in \{1.0, 0.5, 0.1\}$) and attack vectors (Label Flipping, Backdoors, BatchNorm buffer poisoning).
- **Execution & Scope**: 72 balanced primary matrix cells (Seeds 211 and 212) + 24 preserved supplementary historical experiments = 96 unique simulations; 97 physical launches (including 1 documented crash recovery in Batch 2). Strictly sequential in-process execution with memory circuit breaker.
- **Untouched Test Benchmark**: $N=624$ images (234 Normal, 390 Pneumonia).
- **Statistical Disclosure**: Evaluated across $N=2$ distinct seeds (Seeds 211, 212) as exploratory benchmark trends; not claimed as hypothesis-tested statistical significance. Client nodes are simulated partitions over PneumoniaMNIST and do not represent actual clinical hospital infrastructure.
- **Key Empirical Results**:
  1. **Non-IID Label-Flipping Defense**: Tri-State Discounting ($\lambda=0.50$) achieved **83.09% mean accuracy and 58.98% specificity** under Non-IID label flipping ($\alpha=0.5, M=1$), outperforming FedAvg (73.96% Acc / 30.77% Spec) and Multi-Krum (71.72% Acc / 25.00% Spec).
  2. **Multi-Krum Trade-Off (IID vs Non-IID)**: Multi-Krum ($f=1$) was the top performer against IID backdoors (reducing test ASR to **2.43%** vs. FedAvg 8.46%), but suffered severe specificity collapse down to **23.08%** under honest $\alpha=0.5$ skew due to geometric rejection of the primary Normal-specialized client node (verified in telemetry: Client 2 dropped in 18/20 rounds in Seed 211; Client 3 dropped in 19/20 rounds in Seed 212).
  3. **Extreme Skew ($\alpha=0.1$) Class Starvation**: Under severe skew, honest minority-class updates were discounted, causing Tri-State Discounting to collapse to All-Pneumonia (0.00% Spec, 62.50% Acc). Unweighted FedAvg was the most resilient aggregator (78.77% Acc, 70.72% Spec).
  4. **Universal BatchNorm Poisoning Vulnerability**: In all 16 simulations, parameter-only defense metrics failed to detect buffer poisoning (Client 0 classified `NORMAL` in 100% of rounds), and linear buffer averaging caused global model collapse (37.50% All-Normal or 62.50% All-Pneumonia).
- **Ledger Verification**: Cryptographic blockchain audit ledger validated at tip Block #2133 (2,134 blocks, `is_valid = True`, 0 broken links).
- **Artifacts**: Registry: [`results/phase7_robust_aggregation/execution_registry_reduced_96.json`](file:///home/suvranil/project/results/phase7_robust_aggregation/execution_registry_reduced_96.json), Manifest: [`results/phase7_robust_aggregation/PHASE7_REDUCED_EXPERIMENT_MATRIX_96.json`](file:///home/suvranil/project/results/phase7_robust_aggregation/PHASE7_REDUCED_EXPERIMENT_MATRIX_96.json), Audit Trail: [`results/blockchain_audit/audit_ledger.jsonl`](file:///home/suvranil/project/results/blockchain_audit/audit_ledger.jsonl), Synthesis: [`docs/PHASE7_ROBUST_AGGREGATION_SYNTHESIS.md`](file:///home/suvranil/project/docs/PHASE7_ROBUST_AGGREGATION_SYNTHESIS.md).

---

### Phase 8: Research Consolidation, Paper Artifacts & Publication Preparation — CURRENT / PLANNED
- **Status**: **PLANNED / CURRENT SCOPE (Zero Simulations Launched)**
- **Scope Note**: *Explicitly no new training, simulation runs, or smart-contract network deployments have been executed for Phase 8.* This phase focuses strictly on consolidating empirical findings into publication-ready tables, manuscript drafting, and architectural documentation.
- **Key Objectives**:
  1. Author a comprehensive research paper manuscript synthesizing the multi-factor detector (Phase 6), the robust aggregation trade-offs (Phase 7), and the cryptographic audit trail.
  2. Generate high-resolution publication figures (vector PDF / PNG):
     - Empirical ROC/PR curves from Phase 6.
     - 4-strategy comparative performance radar and bar charts across Dirichlet $\alpha$ regimes.
     - Telemetry visualization showing Client 0 attenuation and Multi-Krum geometric minority rejection.
     - BatchNorm buffer corruption distribution diagram.
  3. Formalize the mathematical formulation of Tri-State Soft-Discount Aggregation and prove bounds on adversarial gradient influence under bounded heterogeneity.
  4. Package complete experiment registries, manifests, and verification scripts for open-source reproducibility.

---

### Phase 9: Extended Ablations & Robust Buffer Auditing — FUTURE WORK
- **Status**: **FUTURE RESEARCH DIRECTION (Unexecuted)**
- **Future Research Directions**:
  1. **Non-Linear Batch Normalization Buffer Aggregation**:
     - *Coordinate-Wise Median Buffers*: Aggregate floating-point running statistics (`running_mean`, `running_var`) using coordinate-wise medians rather than linear sample-weighted averaging, preventing single-client outliers ($10^6$) from corrupting global batch statistics.
     - *Buffer Divergence Auditing*: Extend the Phase 6 detector to audit client buffer updates against peer distributions (e.g. running mean $L_2$ norm and Kolmogorov-Smirnov distance) before allowing buffer incorporation.
  2. **Adaptive Extreme-Skew Trust Thresholding**:
     - Dynamically adapt the cosine similarity threshold $\tau_{\text{cos}}$ as a function of measured local data imbalance to avoid minority-class starvation under extreme Dirichlet skew ($\alpha \le 0.1$).
  3. **Multi-Seed Scaling ($N \ge 5$)**:
     - Expand beyond exploratory benchmark seeds ($N=2$) to formal hypothesis-tested statistical evaluations across 5+ independent random seeds.
  4. **Smart Contract Deployment on Ethereum-Compatible Testnets**:
     - Deploy lightweight Solidity verification contracts to test gas costs and on-chain event indexing for multi-hospital regulatory audits.

---

### Phase 10: Publication Artifacts & Defense Slide Deck — FUTURE WORK
- **Status**: **FUTURE WORK**
- **Objective**: Finalize thesis dissertation chapters, produce academic conference slide decks, and archive the complete research repository.
