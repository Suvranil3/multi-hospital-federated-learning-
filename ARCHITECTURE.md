# System Architecture

## Project Title
**Blockchain-Assisted Non-IID-Aware Malicious Participant Detection in Federated Learning**

---

## 1. High-Level Architectural Overview
The architecture is structured into four decoupled, modular subsystems:
1. **Data & Client Simulation Subsystem**: Manages dataset partitioning (Dirichlet non-IID), local data loading, and PyTorch local CNN training on simulated Flower clients.
2. **Federated Orchestration Subsystem**: Manages Flower FL rounds, parameter exchange, and robust aggregation.
3. **Non-IID-Aware Malicious Participant Detector**: An intelligent security layer evaluating client gradient/model updates using multiple orthogonal features (deviation, directional similarity, validation impact, historical behavioral consistency, and heterogeneity indicators) to output a tri-state classification (`Normal`, `Suspicious`, `Malicious`).
4. **Blockchain Audit Trail Subsystem**: A tamper-evident, append-only Ethereum-compatible ledger recording lightweight cryptographic hashes and decision metadata.

```
       +-------------------------------------------------------------+
       |               Federated Learning Clients                    |
       |  [Client 1]    [Client 2]   ...   [Client 9]    [Client 10] |
       |  (Non-IID)     (Non-IID)          (Adversary)   (Non-IID)   |
       +-------------------------------------------------------------+
                                      |
                                      | Client Model Updates (Weights / Gradients)
                                      v
       +-------------------------------------------------------------+
       |             FL Server Orchestrator (Flower)                 |
       |  - Collects client updates                                  |
       |  - Computes global metrics                                  |
       +-------------------------------------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |      Non-IID-Aware Malicious Participant Detector           |
       |                                                             |
       |  [Feature 1: Update Deviation (L2 Norm)]                    |
       |  [Feature 2: Update Similarity (Cosine Similarity)]         |
       |  [Feature 3: Validation Performance Impact (Loss/Acc Delta)]|
       |  [Feature 4: Historical Behavioral Trajectory]              |
       |  [Feature 5: Client Heterogeneity Context]                  |
       |                                                             |
       |  Output: Classification Decision & Aggregation Weight       |
       |    - NORMAL     -> Weight = 1.0 (Accepted)                  |
       |    - SUSPICIOUS -> Weight = alpha_down (Down-weighted)      |
       |    - MALICIOUS  -> Weight = 0.0 (Rejected)                  |
       +-------------------------------------------------------------+
               |                                             |
               | Aggregated Weights                          | Audit Metadata
               v                                             v
+-------------------------------+             +-------------------------------+
|     Global Model Update       |             |   Blockchain Audit Ledger     |
|   (Weighted Aggregation)      |             |   - Round ID, Client ID       |
|                               |             |   - SHA-256(Model Update)     |
| -> Broadcast to Next Round    |             |   - Decision (Normal/Susp/Mal)|
|                               |             |   - Action Taken, Timestamp   |
+-------------------------------+             +-------------------------------+
```

---

## 2. Component Specifications

### 2.1. Client Simulation Tier
- **Framework**: Flower (`flwr[simulation]`) with Ray execution backend.
- **Participants**: 10 edge clients.
- **Data Partitioning**: Dirichlet distribution over classes with concentration parameter $\alpha \in \{1.0, 0.5, 0.1\}$ representing varying degrees of label skew.
- **Model**: Compact Convolutional Neural Network (CNN) tailored for CIFAR-10 (2 convolutional blocks + max pooling + dense classification head).

### 2.2. Non-IID-Aware Malicious Participant Detector
The detector acts as an evaluation middleware prior to model aggregation.
- **Multi-Feature Analysis**:
  1. **Update Deviation**: Measures Euclidean ($L_2$) distance of the update relative to median and mean trajectories.
  2. **Directional Similarity**: Pairwise and reference cosine similarity to distinguish co-aligned adversarial collusions from benign directional variance.
  3. **Local Validation Impact**: Evaluates client update performance on a balanced auxiliary validation subset or local cross-validation proxy to detect performance degradation.
  4. **Behavioral History**: Tracks exponential moving averages of client scores across training rounds to separate temporary statistical variance from persistent adversarial behavior.
  5. **Heterogeneity Awareness**: Normalizes deviation penalties according to estimated client data distribution skew, preventing unfair penalties against honest specialized clients.
- **Tri-State Action Policy**:
  - `Normal`: Update accepted into global aggregation.
  - `Suspicious`: Update down-weighted based on anomaly score.
  - `Malicious`: Update rejected entirely.

### 2.3. Robust Aggregator
- Dynamic weighting aggregator incorporating detector outputs:
  $$w_{i}^{(t)} = \begin{cases} 
  \frac{n_i}{N} & \text{if } \text{decision} = \text{NORMAL} \\
  \lambda \cdot \frac{n_i}{N} & \text{if } \text{decision} = \text{SUSPICIOUS } (0 < \lambda < 1) \\
  0 & \text{if } \text{decision} = \text{MALICIOUS}
  \end{cases}$$
- Re-normalized across non-rejected clients for global update.

### 2.4. Blockchain Audit Layer
- **Explicit Role**: Append-only, tamper-evident cryptographic log.
- **What is Stored On-Chain**:
  - `round`: Communication round number (`uint256`).
  - `clientId`: Client identifier (`uint256`).
  - `updateHash`: SHA-256 cryptographic digest of serialized model parameters (`bytes32`).
  - `decision`: Detection outcome (`uint8`: 0 = Normal, 1 = Suspicious, 2 = Malicious).
  - `action`: Aggregation weight applied (`uint256` scaled).
  - `timestamp`: Block timestamp (`uint256`).
- **What is NOT Stored On-Chain**:
  - Raw model weights or gradient vectors (prohibitive gas costs, unnecessary storage bloat).
  - The detection engine logic (detection is executed off-chain by the ML security server).

---

## 3. Security Boundaries & Threat Model
- **Adversary Model**: Up to 30% of clients ($m \le 3$ out of 10) may be Byzantine (malicious).
- **Adversary Capabilities**:
  - Arbitrary data manipulation (label flipping).
  - Arbitrary parameter scaling or targeted backdoor injection.
  - Collusion among adversarial clients.
- **Assumptions**:
  - The central FL server orchestrator executes uncompromised detection code.
  - Client data remains strictly private on local clients (FL privacy preservation).
  - Blockchain consensus guarantees immutable history of aggregation events.
