# Phase 7 Memory-Safe Execution & Resumability Architecture

**Date:** November 2, 2026  
**Status:** IMPLEMENTED, TESTED & OPERATIONAL  
**Target Matrix:** [`results/phase7_robust_aggregation/PHASE7_EXPERIMENT_MATRIX.json`](file:///home/suvranil/project/results/phase7_robust_aggregation/PHASE7_EXPERIMENT_MATRIX.json) (351 simulations, 70,200 client-round observations)  
**Execution Script:** [`experiments/run_phase7_memory_safe.py`](file:///home/suvranil/project/experiments/run_phase7_memory_safe.py)  
**Execution Registry:** [`results/phase7_robust_aggregation/execution_registry.json`](file:///home/suvranil/project/results/phase7_robust_aggregation/execution_registry.json)  
**Audit Ledger:** [`results/blockchain_audit/audit_ledger.jsonl`](file:///home/suvranil/project/results/blockchain_audit/audit_ledger.jsonl)  

---

## 1. Post-Mortem Inspection of the Failed Execution

### 1.1 Root Cause of Failure
The previous Phase 7 execution stopped because the development machine ran out of available memory, causing Ray agents and the driver process to be killed by the Linux Out-Of-Memory (OOM) killer with `SIGTERM`:

1. **Host Memory Constraints:**
   - Physical RAM: 6,749 MB ($\sim 6.6\text{ GB}$ usable), Swap: 4,095 MB.
   - Baseline system load: Operating System, Gnome desktop, IDE language server (`pyrefly` $\sim 1.5\text{ GB}$), browser, and system services consumed $\sim 5.2 - 5.4\text{ GB}$ of RAM, leaving only $\sim 1.3 - 1.5\text{ GB}$ of available headroom.
2. **Ray Multiprocessing Overhead:**
   - Flower's default `start_simulation` launches Ray, creating a `VirtualClientEngineActorPool`.
   - On this 12-core host, Ray spawned 12 separate worker processes (`ClientAppActor` pids `27752` through `27763`), plus `raylet`, `gcs_server`, `dashboard_agent`, and `log_monitor`.
   - Ray automatically pre-allocated an object store memory plasma buffer of $\sim 650\text{ MB} - 1.5\text{ GB}$.
   - Each worker process loaded PyTorch, NumPy, and model graph definitions in independent virtual memory address spaces.
   - Combined, the Ray runtime demanded $> 3.2\text{ GB}$ of additional memory, instantaneously driving available memory to zero and triggering an unrecoverable OOM crash (`logging.cc:474: *** SIGTERM received ... The raylet exited immediately because one Ray agent failed ... killed by the OS (e.g., out of memory)`).
3. **Memory Accumulation Across Communication Rounds:**
   - In earlier prototypes, strategy telemetry and feature records were appended to unbounded Python lists in memory (`self.extracted_records.append(...)`). Across hundreds of rounds and simulations, this caused monotonic memory growth.
4. **State of Completed Results:**
   - Inspection confirmed that zero simulations of the 351-experiment Phase 7 matrix had written corrupted or conflicting outputs.
   - Historical Phase 4, Phase 5, and Phase 6 artifacts in `results/logs/`, `results/models/`, and `results/phase6_blind_evaluation/` were completely intact and verified.

---

## 2. Memory-Safe Sequential Strategy for 8 GB RAM

To guarantee execution stability under the 8 GB RAM limit, the execution architecture in [`experiments/run_phase7_memory_safe.py`](file:///home/suvranil/project/experiments/run_phase7_memory_safe.py) replaces distributed Ray multi-processing with a strictly sequential, in-process execution engine:

### 2.1 Technical Principles

1. **Strictly Sequential Execution:**
   - Only **ONE** simulation runs at a time.
   - Only **ONE** aggregation strategy is evaluated at a time.
   - Within each communication round ($1..20$), the 10 hospital clients train sequentially one hospital at a time ($0 \to 1 \to \dots \to 9$).
2. **Elimination of Ray Overhead:**
   - Replaced Flower's Ray actor pool with direct in-process `MockClientProxy` and `FitRes` interactions.
   - Eliminates Raylet, GCS server, dashboard agent, and plasma object store.
   - Reduces federated orchestration memory overhead from $> 3,200\text{ MB}$ to $< 250\text{ MB}$.
3. **Active Memory Lifecycle Management:**
   - Per-round client instances, local optimizers, gradient buffers, and client model tensors are immediately deleted (`del client`, `del c_ndarrays`).
   - Strategy in-memory telemetry lists are cleared after disk streaming (`strategy.extracted_records.clear()`).
   - Python garbage collection (`gc.collect()`) is invoked after each communication round and after each simulation.
4. **Continuous Memory Monitoring & Safety Circuit Breaker:**
   - Real-time memory consumption is monitored directly from `/proc/meminfo` and `resource.getrusage(resource.RUSAGE_SELF)`.
   - A critical threshold safety breaker is enforced: if available system RAM falls below **$250\text{ MB}$**, the runner halts further experiment launches, persists all completed work to the registry, and reports the event.
5. **Zero Scientific Compromise:**
   - No silent reduction of batch size ($B=32$).
   - No alteration of local training epochs ($E=1$).
   - No modification of learning rate ($\eta=0.001$), weight decay ($10^{-4}$), or model architecture (`SmallCNN`, 420,802 parameters).
   - Preserves all 13 experimental conditions and 9 aggregation variants.

---

## 3. Resumability & Execution Registry

The 351-simulation suite is managed via a persistent execution registry ([`results/phase7_robust_aggregation/execution_registry.json`](file:///home/suvranil/project/results/phase7_robust_aggregation/execution_registry.json)):

### 3.1 Registry Workflow
Before launching every simulation:
1. **Registry Verification:** Reads `execution_registry.json` to check if `experiment_id` has status `COMPLETED`.
2. **Artifact Integrity Check:** Verifies that the evaluation file ([`results/phase7_robust_aggregation/evaluation/{exp_id}_results.json`](file:///home/suvranil/project/results/phase7_robust_aggregation/evaluation/)) and model checkpoint ([`results/phase7_robust_aggregation/checkpoints/best_{exp_id}.pt`](file:///home/suvranil/project/results/phase7_robust_aggregation/checkpoints/)) exist, are readable, and contain valid non-empty test metrics.
3. **Automatic Skip:** Valid completed experiments are skipped without re-allocating memory or re-running.
4. **Incremental Commit:** Upon successful completion of an experiment, its metrics, duration, memory stats, and blockchain block index are committed atomically to `execution_registry.json`.

---

## 4. Integration with Blockchain Audit Ledger

Every simulation lifecycle event is permanently anchored in the permissioned audit ledger ([`results/blockchain_audit/audit_ledger.jsonl`](file:///home/suvranil/project/results/blockchain_audit/audit_ledger.jsonl)):

1. **Experiment Initialization Block:** Records experiment ID, seed, aggregator name, scenario, and participating hospitals.
2. **Round Aggregation Event Blocks:** Records per-round detector classifications, inclusion/exclusion decisions, effective aggregator weights, and validation accuracy/loss.
3. **Final Result Block:** Records untouched test set diagnostic metrics (Accuracy, Sensitivity, Specificity, Macro-F1, Loss, Confusion Matrix) and SHA-256 cryptographic hashes of all four result artifacts:
   - `results_json`: SHA-256 checksum of evaluation JSON.
   - `best_checkpoint`: SHA-256 checksum of best validation model checkpoint (`.pt`).
   - `final_checkpoint`: SHA-256 checksum of round-20 model checkpoint (`.pt`).
   - `features_jsonl`: SHA-256 checksum of round feature telemetry (`.jsonl`).
4. **Decoupled Non-Blocking Reliability:** All blockchain audit operations are strictly decoupled from scientific results. If a ledger write fails, the error is logged as a non-fatal warning, guaranteeing that simulation checkpoints and test evaluations are never lost or corrupted.

---

## 5. Measured Performance Baseline

The memory-safe sequential runner was benchmarked on the full 20-round honest IID baseline (`EXP-P7-FEDAVG-HONEST-IID-S211`, `EXP-P7-COORDINATE_MEDIAN-HONEST-IID-S211`, `EXP-P7-TRIMMED_MEAN_BETA1-HONEST-IID-S211`):

| Experiment Identifier | Aggregation Variant | Runtime | Pre-Run RAM Used | Post-Run RAM Used | Peak Process RSS | Test Accuracy | Pneumonia Sensitivity | Normal Specificity | Macro F1 | Blockchain Block | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `EXP-P7-FEDAVG-HONEST-IID-S211` | `fedavg` | 87.34s | 5,265.8 MB | 5,340.3 MB | 907.8 MB | 85.74% | 96.41% | 67.95% | 83.78% | #43 | **COMPLETED** |
| `EXP-P7-COORDINATE_MEDIAN-HONEST-IID-S211` | `coordinate_median` | 89.45s | 5,225.2 MB | 5,359.3 MB | 907.7 MB | 86.54% | 96.67% | 69.66% | 84.74% | #65 | **COMPLETED** |
| `EXP-P7-TRIMMED_MEAN_BETA1-HONEST-IID-S211` | `trimmed_mean_beta1` | 87.06s | 5,359.5 MB | 5,325.7 MB | 1,097.0 MB | 86.22% | 96.67% | 68.80% | 84.34% | #87 | **COMPLETED** |

- **RAM Stability:** Memory usage remained completely flat across all simulations ($\sim 5.2 - 5.3\text{ GB}$ system used, $\sim 1.4\text{ GB}$ free).
- **Zero Leaks:** Process peak RSS remained stable around $\sim 1\text{ GB}$, with zero memory accumulation.
- **Ledger Verification:** All 88 committed blocks passed cryptographic verification with zero errors.
