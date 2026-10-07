# Architectural & Technical Decision Records (ADR)

## Project: Blockchain-Assisted Non-IID-Aware Malicious Participant Detection in Federated Learning

---

### ADR-001: Selection of CIFAR-10 Dataset
- **Status**: Accepted
- **Context**: We need a standard benchmark dataset that exhibits non-trivial visual feature complexity, has well-established classification baselines in literature, and allows controlled non-IID partitioning.
- **Decision**: Use CIFAR-10 (10 classes, 50,000 training images, 10,000 test images, 32x32 RGB).
- **Rationale**:
  - Complex enough to demonstrate poisoning and non-IID divergence (unlike MNIST/Fashion-MNIST, where gradients easily converge even with skewed classes).
  - Computationally light enough to simulate 10 clients across multiple rounds within feasible execution times on commodity hardware or Google Colab.
  - Standard benchmark across published FL security literature (e.g., Blanchard et al., Fung et al., Cao et al.), enabling direct comparison.

---

### ADR-002: Selection of Small CNN Model
- **Status**: Accepted
- **Context**: The model must balance representation power for CIFAR-10 with training speed across 10 federated clients.
- **Decision**: Use a compact Convolutional Neural Network (small CNN: 2-3 conv layers + max pooling + fully connected layers) with ~100k-500k parameters.
- **Rationale**:
  - Overly deep models (e.g., ResNet-50) incur high computational and communication latency across 20+ FL rounds on CPU/light GPU.
  - Linear models or shallow MLPs underfit CIFAR-10.
  - A small CNN provides sufficient capacity (~65-75% accuracy) while allowing fast simulation and reproducible gradients.

---

### ADR-003: Selection of Flower (`flwr[simulation]`) Framework
- **Status**: Accepted
- **Context**: We require an FL framework capable of orchestrating multi-client simulations natively in Python and PyTorch.
- **Decision**: Use Flower with the simulation engine (`flwr[simulation]`).
- **Rationale**:
  - Native Python/PyTorch support with minimal boilerplate.
  - Highly efficient resource virtualization via Ray in `flwr.simulation`, enabling 10 simulated clients on a single workstation or Colab instance.
  - Clean separation between client compute and custom server aggregation strategies (`Strategy` interface), ideal for plugging in our custom detector and robust aggregation logic.

---

### ADR-004: Centrality of Dirichlet Non-IID Partitioning
- **Status**: Accepted
- **Context**: Real-world FL clients rarely have identical data distributions.
- **Decision**: Parameterize client data heterogeneity using a Dirichlet distribution over label distributions with concentration parameter $\alpha \in \{1.0, 0.5, 0.1\}$.
- **Rationale**:
  - Continuous tuning of non-IID severity: $\alpha = 1.0$ (mild non-IID), $\alpha = 0.5$ (moderate non-IID), $\alpha = 0.1$ (extreme non-IID).
  - Gold standard in federated learning research for simulating statistical heterogeneity.
  - Essential for answering our primary research question regarding false positive reduction under benign distribution divergence.

---

### ADR-005: Decoupled Blockchain Role: Audit Layer Only
- **Status**: Accepted
- **Context**: Blockchain is frequently misapplied in FL research as a computational engine or storage layer.
- **Decision**:
  - Storing complete model weights on-chain is **strictly prohibited**.
  - Executing ML detection algorithms inside smart contracts is **strictly prohibited**.
  - Blockchain serves exclusively as an immutable, tamper-evident audit log for round metadata, SHA-256 update hashes, detector classification decisions, and actions taken.
- **Rationale**:
  - Smart contract gas costs and block size limits make storing neural network parameters or running floating-point tensor operations impractical.
  - Offloading heavy ML computation to the server and logging cryptographic digests on-chain provides the required accountability, non-repudiation, and auditability without prohibitive gas overhead or transaction latency.

---

### ADR-006: Tri-State Classification Policy (Normal / Suspicious / Malicious)
- **Status**: Accepted
- **Context**: Binary classification (accept vs. reject) often penalizes honest clients with naturally skewed data whose updates marginally cross outlier thresholds.
- **Decision**: Adopt a tri-state classification:
  - `Normal`: Fully accepted (weight = 1.0).
  - `Suspicious`: Down-weighted proportionally to anomaly severity ($0 < \lambda < 1$).
  - `Malicious`: Completely rejected (weight = 0.0).
- **Rationale**:
  - Prevents total information loss from honest specialized clients while safeguarding the global model against catastrophic corruption.
  - Provides a soft margin that naturally accommodates statistical noise from non-IID distributions.

---

### ADR-007: Phased Modular Development Strategy
- **Status**: Accepted
- **Context**: Complex multi-component systems (FL + attacks + detector + blockchain) fail easily if built monolithically.
- **Decision**: Implement the system strictly across 10 sequential phases, starting from Phase 1 (dataset verification) to Phase 10 (scientific analysis), requiring explicit verification at each step.
- **Rationale**: Ensures each subsystem is verified and unit-tested in isolation before integration, preventing hard-to-debug cross-cutting bugs.
