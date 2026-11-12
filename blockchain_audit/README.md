# Blockchain Audit Ledger for Federated Healthcare Learning

This module provides a lightweight, tamper-evident cryptographic audit ledger for federated learning experiments, hospital update submissions, frozen detector decisions, and Byzantine-robust aggregation events.

---

## 1. Architecture & Scope

> [!IMPORTANT]
> **Scope Clarification: Simulated Permissioned Audit Ledger**  
> This module implements an experimental, lightweight, locally simulated permissioned blockchain audit layer utilizing deterministic SHA-256 cryptographic hashing and backwards-linked blocks.  
> 
> It is **not** a decentralized public blockchain, does not utilize proof-of-work (PoW) or proof-of-stake (PoS) mining infrastructure, and does not require a distributed peer-to-peer network. It is specifically engineered to run within the memory constraints of an 8 GB development workstation.
>
> Cryptographic hashing alone does not make a system decentralized or secure against an adversary with host filesystem root access; rather, it provides an auditable, append-only, tamper-evident record of all federated screening rounds and decisions.

---

## 2. Block Data Schema

Every block in the audit ledger contains:

| Field | Type | Description |
|---|---|---|
| `index` | Integer | Monotonically increasing block index ($0$ for Genesis). |
| `timestamp` | String | ISO 8601 UTC timestamp of block creation. |
| `previous_hash` | String | 64-character hex SHA-256 hash of the immediately preceding block. |
| `experiment_id` | String | Unique federated learning experiment identifier. |
| `simulation_seed` | Integer / null | Pseudo-random seed used for dataset partitioning and training. |
| `aggregation_strategy`| String / null | Model aggregation rule identifier (e.g. `fedavg`, `coordinate_median`, `tri_state_discount_050`). |
| `round_number` | Integer / null | Communication round index ($1..20$), or null for experiment-level events. |
| `participating_hospitals` | List[str] | Identifiers of hospitals participating in the round. |
| `detector_decisions` | Dict[str, str] | Frozen tri-state detector classification per hospital (`NORMAL`, `SUSPICIOUS`, `MALICIOUS`). |
| `hospital_decisions` | Dict[str, str] | Final inclusion/exclusion decision per hospital (`INCLUDED`, `EXCLUDED`, `DISCOUNTED`). |
| `aggregation_event_summary` | Dict | Summary metrics, client counts, effective weights, and clinical trade-off outcomes. |
| `artifact_hashes` | Dict[str, str] | SHA-256 cryptographic checksums of saved model checkpoints and result files. |
| `current_hash` | String | SHA-256 hash computed over the deterministic canonical JSON serialization of the block. |

The Genesis block has index $0$ and a defined previous hash:
`0000000000000000000000000000000000000000000000000000000000000000`.

---

## 3. Medical Data Minimization & Privacy Safeguards

In strict adherence to medical data-minimization principles and clinical research privacy safeguards:

1. **NEVER STORED:**
   - Patient names, IDs, medical record numbers (MRNs), or demographic information.
   - Raw chest X-ray image arrays or pixel data.
   - Private hospital local datasets or splits.
   - Full model parameter tensors, gradient vectors, or uncompressed checkpoints.

2. **STRICTLY AUDITED:**
   - Non-identifiable hospital identifiers (`"0"`, `"1"`, ..., `"9"`).
   - High-level detector decisions and anomaly scores.
   - Aggregator weight distributions.
   - Cryptographic SHA-256 fingerprints of immutable disk artifacts.

---

## 4. Resilience & Memory Safety (8 GB RAM Profile)

1. **Disk-Backed Append-Only Ledger:**
   Blocks are serialized to a single-line JSONL file (`audit_ledger.jsonl`). The entire history is never kept in RAM. Blocks are streamed one-by-one from disk during verification, maintaining $O(1)$ memory usage.

2. **Crash & Interrupted Write Recovery:**
   If a crash or process termination interrupts a write mid-line, `recover_interrupted_write()` reads up to the last structurally and cryptographically valid block, safely truncating corrupted partial bytes.

3. **Decoupled Result Integrity:**
   Saving scientific experiment results remains completely independent of the blockchain logger. If a ledger write fails due to filesystem or disk space issues, the error is logged as a warning, guaranteeing that scientific results and checkpoints are never lost.

---

## 5. Module Structure

- [`block.py`](file:///home/suvranil/project/blockchain_audit/block.py): `AuditBlock` data class, deterministic serialization, and SHA-256 hash computation.
- [`blockchain.py`](file:///home/suvranil/project/blockchain_audit/blockchain.py): `BlockchainLedger` engine managing disk persistence, atomic append, and interrupted write recovery.
- [`audit_logger.py`](file:///home/suvranil/project/blockchain_audit/audit_logger.py): High-level lifecycle interface connecting federated simulations to the blockchain.
- [`verification.py`](file:///home/suvranil/project/blockchain_audit/verification.py): Standalone cryptographic verifier and tamper detection utility.

---

## 6. Verification Usage

To audit the integrity of the ledger:

```bash
.venv/bin/python -m blockchain_audit.verification --ledger-path results/blockchain_audit/audit_ledger.jsonl
```

To verify both ledger linkage and SHA-256 checksums of result artifacts on disk:

```bash
.venv/bin/python -m blockchain_audit.verification --ledger-path results/blockchain_audit/audit_ledger.jsonl --verify-artifacts
```
