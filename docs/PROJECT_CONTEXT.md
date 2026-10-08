# Project Context

## Project Title
**Multi-Hospital Federated Learning for Chest X-Ray Classification with Malicious Participant Detection and Blockchain Audit**

---

## 1. Executive Summary & Clinical Context
In healthcare systems, collaborative machine learning offers tremendous potential for training robust, high-performance diagnostic models across institutions. However, patient imaging data—such as chest X-rays—is strictly regulated under clinical privacy frameworks, including HIPAA (Health Insurance Portability and Accountability Act) in the United States and GDPR (General Data Protection Regulation) in the European Union. Hospitals cannot centralize patient medical scans into a single external repository due to patient privacy, institutional liability, and data sovereignty laws.

**Federated Learning (FL)** provides a privacy-preserving solution: participating hospitals collaboratively train a shared global model while retaining sensitive chest X-rays strictly inside their local firewalls. Instead of sending raw patient data, hospitals compute model weight updates locally and share only parameter deltas with a central aggregation server.

---

## 2. Research Background & Project Evolution

### The Academic Mandate
The project initially investigated Byzantine-robust Federated Learning on a generic benchmark dataset (CIFAR-10) during Phase 1 and Phase 2. However, our faculty advisor and project committee clarified that the **complete research project must be developed, evaluated, and presented in an authentic medical/hospital scenario**.

### Dataset & Scenario Migration
To fulfill this requirement:
1. **PneumoniaMNIST** (a standardized pediatric chest X-ray benchmark from MedMNIST v3) was designated as the **Primary Medical Project Dataset**.
2. The initial CIFAR-10 work was fully preserved and archived under [`results/archive/cifar10/`](file:///home/suvranil/project/results/archive/cifar10/README.md) as a historical proof-of-concept.
3. The model architecture, verification pipelines, baseline experiments, testing suites, and interactive teacher dashboard were re-engineered around the multi-hospital medical classification workflow.

---

## 3. Four Core Pillars of the Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   PILLAR 1: HOSPITAL CLIENT TIER                       │
│  10 simulated hospitals with private local chest X-rays (PneumoniaMNIST)│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Local Model Updates
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               PILLAR 2: FEDERATED ORCHESTRATION TIER                   │
│   Flower (flwr[simulation]) framework with FedAvg aggregation          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Candidate Updates
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               PILLAR 3: NON-IID-AWARE SECURITY TIER                    │
│   Multi-factor detector separating benign clinical case-mix diversity  │
│   from adversarial poisoning attacks (tri-state decision policy)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Audit Digests & Decisions
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               PILLAR 4: BLOCKCHAIN AUDIT LAYER                         │
│   Append-only tamper-evident ledger for clinical regulatory compliance  │
│   (zero patient data and zero heavy ML weights on-chain)               │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Hospital Simulation Subsystem**: 10 distinct hospital nodes, each maintaining a local partition of chest X-rays and training a shared [`SmallCNN`](file:///home/suvranil/project/federated_learning/model.py#L21-L71) model locally.
2. **Federated Orchestration Subsystem**: Coordinates communication rounds, parameter distribution, and model aggregation using Flower (`flwr[simulation]`).
3. **Non-IID-Aware Malicious Participant Detector**: An intelligent security layer that evaluates hospital updates across multiple orthogonal features (gradient deviation, cosine similarity, validation impact, historical consistency, heterogeneity context) to prevent poisoning while protecting honest specialized hospitals from false accusations.
4. **Blockchain Medical Audit Trail**: An Ethereum-compatible append-only ledger that records cryptographic digests (SHA-256) of hospital updates and detector decisions for post-hoc regulatory verification and dispute resolution.

---

## 4. Current State vs. Research Boundaries

- **Completed**:
  - Phase MEDICAL-01: PneumoniaMNIST automated verification, empirical statistics, and diagnostic figures.
  - Phase MEDICAL-02: Centralized [`SmallCNN`](file:///home/suvranil/project/federated_learning/model.py#L21-L71) baseline training on GPU (81.25% test accuracy, 99.74% pneumonia recall).
  - 14 automated unit regression tests passing.
  - Interactive demonstration dashboard updated with the hospital story and archived CIFAR-10 viewer.
- **Strictly NOT Implemented Yet**:
  - Phase 3: 10-client IID Federated Learning baseline (**Immediate next milestone**).
  - Phase 4: Non-IID hospital data partitioning (Dirichlet case-mix skew).
  - Phase 5: Adversarial attacks (label flipping, gradient scaling, backdoor injection).
  - Phase 6: Multi-factor detector implementation.
  - Phase 8: Blockchain smart contract deployment.
