# Blockchain Audit Layer Architecture

**Date:** November 2, 2026  
**Status:** IMPLEMENTED, TESTED & INTEGRATED  
**Package:** [`blockchain_audit/`](file:///home/suvranil/project/blockchain_audit/)  
**Ledger Storage:** [`results/blockchain_audit/audit_ledger.jsonl`](file:///home/suvranil/project/results/blockchain_audit/audit_ledger.jsonl)  

---

## 1. Architectural Scope & Design Philosophy

The `blockchain_audit/` subsystem introduces a tamper-evident audit ledger into the federated learning pipeline to cryptographically record hospital participation, detector classifications, model aggregation decisions, and final experimental artifacts.

```
+---------------------------------------------------------------------------------------+
|                               FEDERATED LEARNING PIPELINE                             |
|                                                                                       |
|  [Hospitals 0..9] ---> [Fit Updates] ---> [Frozen Detector] ---> [Robust Aggregator]  |
|                                                  |                        |           |
+--------------------------------------------------|------------------------|-----------+
                                                   v                        v
+---------------------------------------------------------------------------------------+
|                             BLOCKCHAIN AUDIT SUBSYSTEM                                |
|                                                                                       |
|  [AuditLogger]                                                                        |
|        |                                                                              |
|        v                                                                              |
|  [AuditBlock] <--- Canonical JSON <--- SHA-256 (Deterministically Serialized Payload) |
|        |                                                                              |
|        v                                                                              |
|  [BlockchainLedger] (Disk-Backed JSONL, O(1) RAM Streaming, Append-Fsnyc, Auto-Repair) |
|        |                                                                              |
|        v                                                                              |
|  [results/blockchain_audit/audit_ledger.jsonl]                                        |
+---------------------------------------------------------------------------------------+
```

> [!IMPORTANT]
> **Scope & Context: Local Permissioned Ledger vs Decentralized Public Chains**  
> 1. **Experimental Permissioned Ledger:** This module implements an experimental, lightweight, locally simulated permissioned blockchain audit layer.
> 2. **No Mining or Heavy Infrastructure:** It intentionally avoids heavyweight consensus mechanisms (Proof-of-Work, Proof-of-Stake), mining protocols, and distributed peer-to-peer gossip overlays. This enables deterministic cryptographic audit trails without exceeding the strict 8 GB workstation RAM budget.
> 3. **Honest Security Boundaries:** Cryptographic hashing alone does not make a system decentralized or tamper-proof against an adversary with host filesystem root access. Its clinical value lies in providing an append-only, chronologically linked, tamper-evident audit trail for clinical governance, reproducibility verification, and post-hoc forensic review.
> 4. **Algorithmic Decoupling:** The blockchain audit layer is strictly an audit observer; it does not alter federated aggregation mathematics, detector decision logic, or client local optimization.

---

## 2. Cryptographic Block Structure

Every block in the audit ledger is represented by the `AuditBlock` data structure defined in [`blockchain_audit/block.py`](file:///home/suvranil/project/blockchain_audit/block.py):

```python
@dataclass
class AuditBlock:
    index: int                                  # 0 for Genesis, 1, 2, ...
    timestamp: str                              # ISO 8601 UTC timestamp
    previous_hash: str                          # 64-hex SHA-256 of block (i - 1)
    experiment_id: str                          # Standard experiment identifier
    simulation_seed: Optional[int]              # Data partition and training seed
    aggregation_strategy: Optional[str]         # e.g., 'fedavg', 'tri_state_discount_050'
    round_number: Optional[int]                 # Communication round (1..20) or null
    participating_hospitals: List[str]          # Hospital IDs participating in round
    detector_decisions: Dict[str, str]          # Hospital ID -> 'NORMAL' | 'SUSPICIOUS' | 'MALICIOUS'
    hospital_decisions: Dict[str, str]          # Hospital ID -> 'INCLUDED' | 'EXCLUDED'
    aggregation_event_summary: Dict[str, Any]   # Effective weights, included clients, val loss/acc
    artifact_hashes: Dict[str, str]             # Artifact name -> SHA-256 checksum
    current_hash: str                           # Sealed SHA-256 hash over canonical payload
```

### 2.1 Deterministic Hash Calculation
To guarantee byte-for-byte reproducible hashing across platforms and runtime invocations:
1. Block attributes (excluding `current_hash`) are placed in a normalized dictionary.
2. Hospital ID lists are sorted alphabetically.
3. Dictionary keys are sorted alphabetically (`sort_keys=True`).
4. JSON whitespace separators are strictly compacted: `separators=(',', ':')`.
5. ASCII encoding is enforced: `ensure_ascii=True`.
6. SHA-256 is computed over the resulting UTF-8 bytes:
   $$\mathcal{H}_i = \text{SHA-256}(\text{CanonicalJSON}(\text{Block}_i \setminus \{\text{current\_hash}\}))$$

### 2.2 Genesis Block Specification
The Genesis block initializes the ledger with fixed constants:
- `index = 0`
- `previous_hash = "0000000000000000000000000000000000000000000000000000000000000000"` (64 zeroes)
- `experiment_id = "GENESIS-AUDIT-LEDGER"`

---

## 3. Medical Data Minimization & Privacy Safeguards

The audit ledger enforces strict data-minimization safeguards designed for clinical federated research environments (without making formal statutory HIPAA compliance certifications):

| Category | Storage Policy | Enforcement Mechanism |
|---|---|---|
| **Patient Identifiers** | **STRICTLY PROHIBITED** | No patient names, MRNs, dates of birth, or clinical notes are ingested or stored. |
| **Medical Imagery** | **STRICTLY PROHIBITED** | Raw chest X-ray pixel arrays are never serialized into blocks. |
| **Private Datasets** | **STRICTLY PROHIBITED** | Hospital local training splits remain on the simulated client nodes. |
| **Model Tensors** | **STRICTLY PROHIBITED** | Raw weights, gradient vectors, and multi-megabyte model checkpoints are never stored in the ledger. |
| **Cryptographic Hashes** | **PERMITTED & ENFORCED** | SHA-256 hashes of disk artifacts (`.pt` checkpoints, evaluation `.json`, feature `.jsonl`) are committed to anchor proof of existence. |
| **Audit Metadata** | **PERMITTED & ENFORCED** | Hospital identifiers (`"0"` through `"9"`), tri-state detector states (`NORMAL`, `SUSPICIOUS`, `MALICIOUS`), and inclusion flags are recorded. |

---

## 4. Module Architecture & Implementation

The component is partitioned into five specialized modules:

### 4.1 [`block.py`](file:///home/suvranil/project/blockchain_audit/block.py) — Cryptographic Block Model
- Implements `AuditBlock` data class.
- Implements deterministic serialization via canonical JSON.
- Provides `seal()`, `compute_hash()`, and `is_valid()` methods.
- Generates canonical Genesis blocks.

### 4.2 [`blockchain.py`](file:///home/suvranil/project/blockchain_audit/blockchain.py) — Disk-Backed Ledger Engine
- Manages append-only persistence in JSONL format (`audit_ledger.jsonl`).
- Guarantees immediate persistence via `flush()` and POSIX `os.fsync()`.
- Thread-safe block appending via `threading.Lock`.
- Provides `stream_blocks_from_file()` for $O(1)$ memory streaming.
- Implements `recover_interrupted_write()`: detects trailing syntax errors or partial lines caused by power loss or process kill, automatically truncating only trailing corrupted bytes to restore the last valid block.

### 4.3 [`audit_logger.py`](file:///home/suvranil/project/blockchain_audit/audit_logger.py) — Federated Learning Lifecycle Bridge
- Bridges the federated learning runner to the ledger.
- Encapsulates three core lifecycle events:
  1. `log_experiment_init(...)`: Experiment configuration, seed, scenario, and participants.
  2. `log_round_event(...)`: Batched communication round decisions, detector evaluations, hospital exclusions, and aggregator metrics.
  3. `log_experiment_result(...)`: Final test set diagnostic performance and SHA-256 hashes of all result artifacts on disk.
- **Fault-Tolerant Scientific Decoupling:** All ledger writes are wrapped in exception handlers. If a blockchain disk operation fails, a warning is logged, ensuring scientific results and checkpoints are never lost.

### 4.4 [`verification.py`](file:///home/suvranil/project/blockchain_audit/verification.py) — Cryptographic Audit Verifier
- Standalone verification engine for comprehensive auditing:
  - Verifies Genesis block index and predefined previous hash.
  - Verifies self-hash integrity of every block.
  - Verifies backwards hash linkage from block $i$ to block $i-1$.
  - Verifies monotonic sequential indices ($0, 1, 2, \dots$).
  - Detects duplicate experiment completions.
  - Audits referenced disk artifacts against stored SHA-256 hashes.
- Provides programmatic API (`verify_ledger()`) and CLI executable (`python -m blockchain_audit.verification`).

### 4.5 [`README.md`](file:///home/suvranil/project/blockchain_audit/README.md) — Technical Reference & Usage Guide
- Operational documentation, command references, and architecture overview.

---

## 5. Distinction Between Blockchain Audit and Aggregation Algorithms

The blockchain audit layer operates strictly as an independent observer. It does **NOT**:
1. Modify client model update vectors.
2. Alter detector thresholds or feature extraction.
3. Replace Byzantine-robust aggregation algorithms (FedAvg, Coordinate Median, Trimmed Mean, Multi-Krum, Tri-State Consensus).
4. Perform consensus voting on gradient weights.

The robust aggregators determine which models are mathematically combined to form the global model. The blockchain ledger creates an immutable, verifiable historical record of those actions.
