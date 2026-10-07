# Research Objective

## Project Title
**Blockchain-Assisted Non-IID-Aware Malicious Participant Detection in Federated Learning**

---

## 1. Primary Research Question
> **"Can a non-IID-aware malicious participant detection mechanism reduce false positives while still detecting poisoning attacks in Federated Learning?"**

---

## 2. Context & Problem Statement
Federated Learning (FL) enables collaborative machine learning training across distributed edge participants without centralizing raw training data. While preserving data privacy, FL introduces major vulnerabilities:
1. **Susceptibility to Poisoning Attacks**: Malicious participants can inject crafted updates to corrupt the global model (untargeted model poisoning) or embed backdoors / misclassification behaviors (targeted attacks and label flipping).
2. **Data Heterogeneity (Non-IID Dilemma)**: In realistic edge environments, participants hold non-identically and independently distributed (non-IID) local datasets. Honest clients with specialized, skewed data distributions naturally produce local gradient updates that deviate significantly from the central average or majority consensus.
3. **The False Positive Penalty**: Conventional Byzantine-robust aggregation and anomaly-detection schemes often treat update divergence as a proxy for maliciousness. Under high non-IID skew, this heuristic frequently misclassifies benign, highly specialized clients as adversaries, degrading fairness, diversity, and convergence.

---

## 3. Core Hypotheses
- **Hypothesis 1 (Heterogeneity-Aware Differentiation)**: Incorporating client data heterogeneity context, historical behavioral consistency, and validation impact alongside update similarity/deviation enables more accurate separation of benign statistical outliers from adversarial poisoning updates.
- **Hypothesis 2 (False Positive Reduction)**: A multi-factor detector will significantly decrease the False Positive Rate (FPR) on honest non-IID clients compared to standard distance/clustering-based Byzantine filters, without sacrificing the True Positive Rate (TPR) or Attack Detection Recall.
- **Hypothesis 3 (Tamper-Evident Auditability)**: An immutable blockchain audit ledger recording lightweight cryptographic hashes and detection metadata provides post-hoc verifiability and non-repudiation of aggregation decisions with negligible computational overhead compared to model training.

---

## 4. Scope and Phased Boundaries
- **Dataset**: CIFAR-10 (10-class image classification, 50,000 train, 10,000 test).
- **Federated Setup**: 10 simulated clients using Flower framework and PyTorch.
- **Partitioning**: Dirichlet distribution ($\alpha \in \{1.0, 0.5, 0.1\}$) representing mild, moderate, and extreme non-IID conditions.
- **Attacks Investigated**:
  1. Label Flipping (targeted/untargeted)
  2. Model Poisoning / Gradient Scaling
  3. Backdoor Injection
- **Baseline Aggregation Schemes**:
  1. Standard FedAvg
  2. Multi-Krum
  3. Trimmed Mean
- **Detection Actions**:
  - `Normal`: Update accepted into global aggregation.
  - `Suspicious`: Update down-weighted based on anomaly score.
  - `Malicious`: Update rejected from global aggregation.
- **Blockchain Role**: Strictly an immutable audit trail for metadata (client ID, round, update hash, decision, action, timestamp). **The blockchain does NOT store weights and is NOT the detection engine.**

---

## 5. Research Rigor & Validation Standards
- **No Fictitious Numbers**: All metrics in reports and papers will strictly derive from reproducible experimental logs.
- **Status Labelling**: All features and claims must be categorized as `[PLANNED]`, `[IMPLEMENTED]`, `[MEASURED]`, or `[OBSERVED]`.
- **Ablation Studies**: Rigorous ablation of each detection factor (similarity, deviation, validation performance, historical trajectory, heterogeneity context) to quantify individual contributions.
