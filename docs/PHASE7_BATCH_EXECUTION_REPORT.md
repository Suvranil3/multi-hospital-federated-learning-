# Phase 7 Batch Execution & Validation Report

**Date:** November 2, 2026  
**Status:** COMPLETED & CRYPTOGRAPHICALLY VERIFIED  
**Authoritative Matrix:** [`results/phase7_robust_aggregation/PHASE7_EXPERIMENT_MATRIX.json`](file:///home/suvranil/project/results/phase7_robust_aggregation/PHASE7_EXPERIMENT_MATRIX.json) (351 simulations total)  
**Execution Script:** [`experiments/run_phase7_memory_safe.py`](file:///home/suvranil/project/experiments/run_phase7_memory_safe.py)  
**Execution Registry:** [`results/phase7_robust_aggregation/execution_registry.json`](file:///home/suvranil/project/results/phase7_robust_aggregation/execution_registry.json)  
**Blockchain Audit Ledger:** [`results/blockchain_audit/audit_ledger.jsonl`](file:///home/suvranil/project/results/blockchain_audit/audit_ledger.jsonl)  
**Full Test Suite:** 89 / 89 Passed (100%) in 11.14s  

---

## 1. Executive Summary

This report documents the rigorous pre-execution audit, scientific parity verification, and subsequent batch execution of five simulations in Phase 7 of the Multi-Hospital Federated Learning research project. 

The execution strictly enforced all memory safety, privacy, and methodological constraints on the 8 GB development workstation:
- **No Multiprocessing:** Sequential, in-process client fitting without Ray worker pools or background daemons.
- **Circuit Breaker Active:** Continuous monitoring via `/proc/meminfo` with an automatic pause circuit breaker at $<250\text{ MB}$ available RAM.
- **Audit Logging:** Every simulation lifecycle event (initialization, 20 communication rounds, final diagnostic evaluation) and disk artifact SHA-256 fingerprints were appended to a local SHA-256 hash-linked audit ledger.
- **Scientific Boundaries:** The authoritative 351-simulation matrix and the frozen Phase 6 detector v1 remain strictly unchanged.

---

## 2. Part 1 — Verification of Existing Results

Before launching further runs, an inspection of the registry, matrix manifest, and result artifacts was conducted:

1. **Matrix Membership:**
   - `EXP-P7-FEDAVG-HONEST-IID-S211` (Matrix Index 1)
   - `EXP-P7-COORDINATE_MEDIAN-HONEST-IID-S211` (Matrix Index 2)
   - `EXP-P7-TRIMMED_MEAN_BETA1-HONEST-IID-S211` (Matrix Index 3)
   - All three map directly to valid, unique entries in `PHASE7_EXPERIMENT_MATRIX.json`.
2. **Round Completion:**
   - All three simulations completed all 20 communication rounds ($1..20$).
3. **Client Participation:**
   - Verified in feature streams (`results/phase7_robust_aggregation/instrumentation/*_features.jsonl`): each simulation trained exactly 10 simulated hospital clients per round across all 20 rounds, producing exactly 200 client observations per experiment.
4. **Artifact Integrity:**
   - Results JSON files (`results/phase7_robust_aggregation/evaluation/*_results.json`), best model checkpoints (`*_best_checkpoint.pt`), final round-20 checkpoints (`*_final_checkpoint.pt`), and feature streams exist, are non-empty, and load without error.
5. **Cryptographic Consistency:**
   - Computed SHA-256 hashes of all disk artifacts match the values recorded in the audit ledger completion blocks with bit-for-bit precision.
6. **Registry Hygiene:**
   - No experiments were incorrectly marked complete. No duplicate entries exist in `execution_registry.json`.

---

## 3. Part 2 — Investigation of Blockchain Ledger Numbering

The initial run reported completion blocks at #43, #65, and #87. A block-by-block audit of Blocks 0–87 was conducted to explain this sequence:

### 3.1 Architectural Clarification
The audit ledger is a **locally simulated, permissioned, SHA-256 hash-linked append-only ledger**, designed specifically to provide tamper-evident provenance within the memory constraints of an 8 GB development machine. It is **not** a decentralized production blockchain, does not require Proof-of-Work / Proof-of-Stake consensus mechanisms, and does not broadcast over a P2P network. Furthermore, while the architecture enforces strict medical data-minimization principles (zero patient identifiers, zero raw X-rays, zero private client splits, and zero uncompressed model tensors stored in ledger blocks), it does **not** claim formal statutory HIPAA compliance certifications.

### 3.2 Block Breakdown (Blocks 0–87)

| Block Range | Event Type | Experiment Identifier | Status & Description |
|---|---|---|---|
| **Block #0** | `GENESIS-AUDIT-LEDGER` | `GENESIS-AUDIT-LEDGER` | Canonical Genesis block (`previous_hash = "0"*64`). |
| **Blocks #1–21** | `EXPERIMENT_INITIALIZATION`, `ROUND_AGGREGATION_EVENT` (1..20) | `EXP-P7-FEDAVG-HONEST-IID-S211` | **Initial Aborted Execution Attempt:** Completed 20 training rounds but crashed during final test evaluation unpacking prior to emitting a completion block. In accordance with append-only immutability, these orphaned intermediate round events were retained in the chain. |
| **Blocks #22–43** | `EXPERIMENT_INITIALIZATION`, `ROUND_AGGREGATION_EVENT` (1..20), `FINAL_EXPERIMENT_RESULT` | `EXP-P7-FEDAVG-HONEST-IID-S211` | **Successful Complete Execution:** Re-run of FedAvg. Block #22 initialized the experiment, Blocks #23–42 recorded rounds 1–20, and Block #43 committed the final test metrics and 4 artifact SHA-256 hashes. |
| **Blocks #44–65** | `EXPERIMENT_INITIALIZATION`, `ROUND_AGGREGATION_EVENT` (1..20), `FINAL_EXPERIMENT_RESULT` | `EXP-P7-COORDINATE_MEDIAN-HONEST-IID-S211` | **Successful Complete Execution:** Block #44 init, Blocks #45–64 rounds 1–20, Block #65 completion with 4 artifact hashes. |
| **Blocks #66–87** | `EXPERIMENT_INITIALIZATION`, `ROUND_AGGREGATION_EVENT` (1..20), `FINAL_EXPERIMENT_RESULT` | `EXP-P7-TRIMMED_MEAN_BETA1-HONEST-IID-S211` | **Successful Complete Execution:** Block #66 init, Blocks #67–86 rounds 1–20, Block #87 completion with 4 artifact hashes. |

Every simulation lifecycle strictly consumes **22 blocks** (1 Init + 20 Communication Rounds + 1 Final Result).

---

## 4. Part 3 — Scientific Equivalence & Parity Verification

The sequential runner (`experiments/run_phase7_memory_safe.py`) was audited against the original Flower-based framework across all scientific dimensions:

| Dimension | Original Flower Framework | Phase 7 Sequential Runner | Equivalence Status |
|---|---|---|:---:|
| **Model Initialization** | `initial_federated_model.pt` (`SmallCNN`, 420,802 params) | Exact same checkpoint file loaded; SHA-256 verified (`0c411b...`) | **IDENTICAL** |
| **Client Partitioning** | `prepare_federated_data_loaders` / `prepare_controlled_non_iid_data_loaders` | Exact same loader generation functions and partition seeds | **IDENTICAL** |
| **Client Training** | `HospitalClient.fit` | Direct in-process invocation of `HospitalClient.fit` | **IDENTICAL** |
| **Optimizer & Local Epochs**| Adam ($\text{lr}=0.001$, $\text{weight\_decay}=10^{-4}$), $E=1$ | Adam ($\text{lr}=0.001$, $\text{weight\_decay}=10^{-4}$), $E=1$ | **IDENTICAL** |
| **Batch Size & Hyperparameters** | $B=32$, CrossEntropyLoss | $B=32$, CrossEntropyLoss | **IDENTICAL** |
| **Client Sample Counts** | Passed via `num_examples` in Flower `FitRes` | Passed via `num_examples` in `FitRes` | **IDENTICAL** |
| **Server-Round Protocol** | 20 rounds, all 10 clients selected per round | 20 rounds, all 10 clients fitted sequentially per round | **IDENTICAL** |
| **Model Parameter Order** | 18 tensors in `state_dict` | Exact same 18 tensors extracted and set via model helpers | **IDENTICAL** |
| **BatchNorm Float Buffers** | Sample-weighted linear combination over included clients | Exact same `aggregate_bn_buffers` method called | **IDENTICAL** |
| **BatchNorm Integer Counters**| Max/sample-weighted advancement of `num_batches_tracked` | Exact same integer scalar step advancement | **IDENTICAL** |
| **Aggregation Strategies** | `create_robust_strategy` factory | Exact same strategy instances and `strategy.aggregate_fit` method | **IDENTICAL** |
| **Validation Checkpointing** | Evaluated on clean validation set ($N=524$) each round | Evaluated on clean validation set ($N=524$) each round | **IDENTICAL** |
| **Test-Set Isolation** | Pristine test set ($N=624$) evaluated strictly post-round 20 | Pristine test set ($N=624$) evaluated strictly post-round 20 | **IDENTICAL** |

### 4.1 Reproducibility & Parity Unit Test Suite
A dedicated unit test suite ([`tests/test_phase7_sequential_parity.py`](file:///home/suvranil/project/tests/test_phase7_sequential_parity.py)) was implemented:
- Verified checkpoint parameter hash invariance.
- Verified deterministic parity of `MockClientProxy` + `FitRes` aggregation across multiple strategies (`fedavg`, `coordinate_median`, `trimmed_mean_beta1`, `multi_krum_f1`).
- Verified exact 18-tensor state dict preservation.
- Verified BatchNorm floating-point running buffer updates and integer scalar step handling.

All 6 parity tests passed, bringing the project regression suite to **89 passed, 0 failed** in 11.14s.

---

## 5. Part 4 & 5 — Post-Batch Execution Scorecard (Simulations 4–8)

The batch of five simulations was executed sequentially via:
```bash
.venv/bin/python experiments/run_phase7_memory_safe.py --max-experiments 5
```

### 5.1 High-Level Metrics Summary

| Metric # | Required Metric Description | Measured Result |
|:---:|---|---|
| **1** | Newly Completed Experiments | **5 simulations** |
| **2** | Total Valid Completed Experiments | **8 simulations** |
| **3** | Remaining Experiments in Matrix | **343 simulations** (out of 351) |
| **4** | Runtime for Each Simulation | See detailed table below ($96.75\text{s} - 98.57\text{s}$) |
| **5** | Peak Process RSS | **1,049.2 MB** (~1.05 GB) |
| **6** | Minimum Available System RAM | **1,373.4 MB** (~1.37 GB, $>5.4\times$ the 250 MB threshold) |
| **7** | Per-Experiment Test Accuracy | $85.58\% - 86.70\%$ |
| **8** | Pneumonia Sensitivity | $95.90\% - 97.44\%$ |
| **9** | Normal Specificity | $65.81\% - 71.37\%$ |
| **10** | Macro-F1 Score | $83.40\% - 85.05\%$ |
| **11** | Blockchain Block Numbers | Blocks **#109, #131, #153, #175, #197** |
| **12** | Blockchain Verification Status | **VALID (198 blocks verified, 0 errors, 0 warnings)** |
| **13** | Failures or Warnings | **Zero failures, zero warnings, zero OOM events** |

---

### 5.2 Detailed Empirical Scorecard (Newly Completed Batch)

All simulations operated under condition **`HONEST-IID`** on **Seed 211** across 20 communication rounds with 10 hospitals:

| Experiment ID | Aggregation Strategy | Runtime | Peak RSS | Test Acc | Sensitivity | Specificity | Macro-F1 | Test Loss | Confusion Matrix (TN, FP, FN, TP) | Completion Block |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `EXP-P7-TRIMMED_MEAN_BETA2-HONEST-IID-S211` | `trimmed_mean_beta2` | 97.99s | 856.2 MB | **86.70%** | 95.90% | 71.37% | **85.05%** | 0.3804 | (167, 67, 16, 374) | **#109** |
| `EXP-P7-MULTI_KRUM_F1-HONEST-IID-S211` | `multi_krum_f1` | 98.57s | 1,041.1 MB | **85.58%** | 97.44% | 65.81% | **83.40%** | 0.3957 | (154, 80, 10, 380) | **#131** |
| `EXP-P7-MULTI_KRUM_F2-HONEST-IID-S211` | `multi_krum_f2` | 97.62s | 1,045.8 MB | **85.74%** | 96.15% | 68.38% | **83.82%** | 0.4072 | (160, 74, 15, 375) | **#153** |
| `EXP-P7-TRI_STATE_MONITOR-HONEST-IID-S211` | `tri_state_monitor` | 96.75s | 1,049.2 MB | **85.74%** | 96.41% | 67.95% | **83.78%** | 0.4299 | (159, 75, 14, 376) | **#175** |
| `EXP-P7-TRI_STATE_DISCOUNT_050-HONEST-IID-S211` | `tri_state_discount_050` | 97.70s | 1,049.2 MB | **85.74%** | 97.44% | 66.24% | **83.61%** | 0.4285 | (155, 79, 10, 380) | **#197** |

---

### 5.3 Complete Baseline Suite Comparison (Seed 211 Honest IID, All 8 Completed Strategies)

| Aggregation Strategy | Test Acc | Pneumonia Sensitivity | Normal Specificity | Macro-F1 | Test Loss | False Negatives | False Positives | Blockchain Block |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `fedavg` | 85.74% | 96.41% | 67.95% | 83.78% | 0.4299 | 14 | 75 | #43 |
| `coordinate_median` | 86.54% | 96.67% | 69.66% | 84.74% | 0.3815 | 13 | 71 | #65 |
| `trimmed_mean_beta1` | 86.22% | 96.67% | 68.80% | 84.34% | 0.4486 | 13 | 73 | #87 |
| `trimmed_mean_beta2` | **86.70%** | 95.90% | **71.37%** | **85.05%** | **0.3804** | 16 | **67** | #109 |
| `multi_krum_f1` | 85.58% | **97.44%** | 65.81% | 83.40% | 0.3957 | **10** | 80 | #131 |
| `multi_krum_f2` | 85.74% | 96.15% | 68.38% | 83.82% | 0.4072 | 15 | 74 | #153 |
| `tri_state_monitor` | 85.74% | 96.41% | 67.95% | 83.78% | 0.4299 | 14 | 75 | #175 |
| `tri_state_discount_050` | 85.74% | **97.44%** | 66.24% | 83.61% | 0.4285 | **10** | 79 | #197 |

*Note on Tri-State Consensus in Honest Environments:*
Under the honest IID baseline, all hospital updates are classified as `NORMAL` by the frozen detector. Consequently:
- `tri_state_monitor` applies monitor-only pass-through, producing numerical results identical to standard `fedavg` (Test Acc 85.74%, Macro-F1 83.78%).
- `tri_state_discount_050` preserves 100% weights for all honest clients and selects a high-sensitivity model checkpoint with only 10 False Negatives (97.44% pneumonia sensitivity).

---

## 6. Audit Ledger Verification & Integrity Certification

Verification was executed via the standalone verification module:
```bash
.venv/bin/python -m blockchain_audit.verification \
    --ledger-path results/blockchain_audit/audit_ledger.jsonl \
    --verify-artifacts
```

### 6.1 Cryptographic Report
- **Total Blocks in Chain:** 198 (Index 0 through 197)
- **Genesis Block Integrity:** Verified (`previous_hash = "0"*64`, index 0)
- **Sequential Continuity:** Strict monotonically increasing index sequence ($0, 1, 2, \dots, 197$)
- **Cryptographic Linkage:** 100% of blocks contain `previous_hash` matching the SHA-256 canonical hash of the preceding block
- **Artifact Anchoring:** All 8 completed simulations anchor 4 disk artifact hashes each (total 32 artifact fingerprints verified against current filesystem contents)
- **Chain Status:** **VALID (0 errors, 0 warnings)**

---

## 7. Next Steps

Execution has stopped strictly after five simulations as requested. 

The memory-safe sequential runner, audit ledger, and persistent registry are verified to be stable and reproducible. The remaining 343 simulations of the 351-experiment matrix are ready to be scheduled in subsequent batches upon review.
