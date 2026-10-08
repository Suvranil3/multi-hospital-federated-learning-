# Research Objective

## Project Title
**Multi-Hospital Federated Learning for Chest X-Ray Classification with Malicious Participant Detection and Blockchain Audit**

---

## 1. Primary Research Question
> **"Can a non-IID-aware malicious participant detection mechanism reduce false positives against honest specialized hospitals while reliably identifying poisoning attacks in multi-hospital Federated Learning?"**

---

## 2. Context & Problem Statement
In healthcare environments, patient chest X-rays are highly confidential medical records subject to strict data-privacy regulations (e.g., HIPAA, GDPR). Multi-hospital collaboration is essential for training robust clinical models, yet hospitals cannot centralize raw patient data.

Federated Learning (FL) enables 10 simulated hospitals to collaboratively train a shared Convolutional Neural Network on chest X-rays without sharing patient images. However, this setup faces key challenges:
1. **Clinical Data Heterogeneity (Non-IID Case Mix)**: Different hospitals have naturally distinct patient populations and disease prevalences (e.g., specialized pediatric pulmonary hospitals vs. general community clinics). Consequently, legitimate model updates from honest specialized hospitals deviate significantly from the central consensus.
2. **Adversarial Vulnerability**: A compromised or malicious hospital can upload poisoned weight updates (e.g., label flipping, model replacement) to degrade diagnostic accuracy or manipulate specific diagnoses.
3. **The False Positive Penalty in Healthcare**: Conventional Byzantine-robust defenses (Krum, Trimmed Mean) penalize any divergent update. In clinical non-IID settings, this mistakenly flags and discards updates from honest specialized hospitals, depriving the global model of valuable regional medical expertise.

---

## 3. Core Hypotheses
- **Hypothesis 1 (Clinical Heterogeneity-Aware Differentiation)**: Incorporating hospital case-mix heterogeneity context, historical update consistency, and validation impact alongside update cosine similarity/deviation enables accurate separation of benign clinical diversity from adversarial poisoning attacks.
- **Hypothesis 2 (False Positive Reduction)**: A multi-factor detector will significantly decrease the False Positive Rate (FPR) on honest specialized hospitals compared to standard geometric Byzantine filters, without sacrificing Attack Detection Recall.
- **Hypothesis 3 (Tamper-Evident Medical Auditability)**: An immutable blockchain audit ledger recording lightweight cryptographic hashes of model updates and detector classifications provides verifiable healthcare auditability and non-repudiation with negligible computation overhead.

---

## 4. Scope and Phased Boundaries
- **Primary Dataset**: PneumoniaMNIST (MedMNIST v3, 5,856 pediatric chest X-rays, 28x28 grayscale, Normal vs. Pneumonia).
- **Previous Proof-of-Concept**: CIFAR-10 preserved as archived historical benchmark in `results/archive/cifar10/`.
- **Federated Setup**: 10 simulated hospital clients using Flower framework and PyTorch.
- **Partitioning Plan**: Dirichlet distribution ($\alpha \in \{1.0, 0.5, 0.1\}$) simulating realistic clinical case-mix skew across hospitals.
- **Attacks Investigated**:
  1. Label Flipping (Normal ↔ Pneumonia)
  2. Model Poisoning / Gradient Scaling
  3. Backdoor Trigger Injection
- **Baseline Aggregation Schemes**:
  1. Standard FedAvg
  2. Multi-Krum
  3. Trimmed Mean
- **Detection Actions**:
  - `Normal`: Hospital update accepted into global aggregation ($w = 1.0$).
  - `Suspicious`: Hospital update down-weighted based on anomaly score ($0 < w < 1.0$).
  - `Malicious`: Hospital update rejected from global aggregation ($w = 0.0$).
- **Blockchain Role**: Strictly an immutable audit trail for metadata (hospital ID, round, update hash, decision, action, timestamp). **The blockchain does NOT store weights, does not store patient images, and is NOT the detection engine.**

---

## 5. Research Rigor & Validation Standards
- **Empirical Grounding**: All reported numbers derive from reproducible experimental logs (seed 42).
- **Status Labelling**: All features are categorized as `[PLANNED]`, `[IMPLEMENTED]`, `[MEASURED]`, or `[OBSERVED]`.
- **Medical Disclaimer**: Reported metrics are academic benchmark scores for simulated multi-hospital machine learning research, not certified clinical diagnostics.
