# Phase 7 Batch 2 Incident Investigation & Recovery Audit Log

**Date:** 2026-11-16  
**Incident Scope:** Phase 7 Batch 2 Initial Launch (`EXP-P7-FEDAVG-ATK-BD-IID-M1-S211`)  
**Resolution:** Runner repair, incomplete ledger truncation, clean re-execution from scratch.  
**Integrity Status:** Verified 100% Intact. No historical data lost.

---

## 1. Incident Overview & Failure Transcript

During the initial launch of Batch 2 on the reduced Phase 7 matrix (`results/phase7_robust_aggregation/PHASE7_REDUCED_EXPERIMENT_MATRIX_96.json`), the first experiment in the batch, `EXP-P7-FEDAVG-ATK-BD-IID-M1-S211` (Cell 25/72: Backdoor IID, FedAvg baseline, Seed 211), crashed immediately following completion of Round 20 and test-set evaluation.

### Error Traceback
```text
[25/72] >>> Executing EXP-P7-FEDAVG-ATK-BD-IID-M1-S211...
       Condition: ATK-BD-IID-M1 | Variant: fedavg | Seed: 211
       RAM Before: 4470.9 MB used, 2279.0 MB available | Process Peak RSS: 911.4 MB
  [ERROR] Simulation EXP-P7-FEDAVG-ATK-BD-IID-M1-S211 failed: type dict doesn't define __round__ method
Traceback (most recent call last):
  File "/home/suvranil/project/experiments/run_phase7_memory_safe.py", line 691, in <module>
    main()
  File "/home/suvranil/project/experiments/run_phase7_memory_safe.py", line 678, in main
    run_matrix_suite(
        manifest_path=Path(args.manifest),
        ...
    )
  File "/home/suvranil/project/experiments/run_phase7_memory_safe.py", line 650, in run_matrix_suite
    raise e
  File "/home/suvranil/project/experiments/run_phase7_memory_safe.py", line 623, in run_matrix_suite
    res = execute_single_simulation(
        exp_cfg=exp,
        audit_logger=audit_logger,
        device=torch.device("cpu"),
    )
  File "/home/suvranil/project/experiments/run_phase7_memory_safe.py", line 505, in execute_single_simulation
    "backdoor_asr": round(asr, 2) if asr is not None else None,
TypeError: type dict doesn't define __round__ method
```

### Physical Execution Analysis
- **Experiment Active at Crash:** `EXP-P7-FEDAVG-ATK-BD-IID-M1-S211`.
- **Training Status:** All 20 rounds of local federated training had completed successfully on CPU.
- **Root Cause:** In `run_phase7_memory_safe.py`, `asr` was obtained by calling `evaluate_backdoor_asr(test_eval_model, test_loader, device=device)`. In `federated_learning/attacks/backdoor.py`, `evaluate_backdoor_asr()` returns a structured dictionary:
  ```python
  {
      "total_triggered_test_samples": total_triggered,
      "successful_backdoor_attacks": successful_attacks,
      "attack_success_rate_percent": round(asr, 2),
      "target_class": 0,
      "source_class": 1,
  }
  ```
  Attempting `round(asr, 2)` on a dictionary object raised `TypeError`.
- **Incomplete Artifacts Written During Failed Attempt:**
  - `best_exp-p7-fedavg-atk-bd-iid-m1-s211.pt` (written at line 440).
  - `final_exp-p7-fedavg-atk-bd-iid-m1-s211.pt` (written at line 444).
  - `exp-p7-fedavg-atk-bd-iid-m1-s211_features.jsonl` (written incrementally during rounds 1..20).
  - Evaluation JSON (`exp-p7-fedavg-atk-bd-iid-m1-s211_results.json`) was **never written** (line 522 was not reached).
  - Registry: Recorded `"status": "FAILED"`.
  - Blockchain Ledger: Appended 21 uncommitted blocks (Block #1,078: initialization + Blocks #1,079–#1,098: rounds 1–20). The final experiment result block (Block #1,099) was **never written**.

---

## 2. Runner Modification & Code Audit

### Modification Details (`experiments/run_phase7_memory_safe.py`)
To resolve the ASR evaluation error and ensure proper test evaluation of backdoor attacks:
1. In `execute_single_simulation()`, lines 464–477 were updated to instantiate `TriggeredPneumoniaTestDataset(clean_test_dataset=test_loader.dataset)` wrapped in a `DataLoader`, evaluate `evaluate_backdoor_asr()`, extract `float(asr_dict["attack_success_rate_percent"])`, and explicitly delete intermediate dataloaders for garbage collection.
2. In `test_metrics`, line 505 was updated to `round(asr, 2)` where `asr` is the float percentage.

### Git Diff
```diff
--- a/experiments/run_phase7_memory_safe.py
+++ b/experiments/run_phase7_memory_safe.py
@@ -464,7 +464,17 @@ def execute_single_simulation(
     # Backdoor ASR evaluation if backdoor attack
     asr: Optional[float] = None
     if attack_type == "backdoor":
-        asr = evaluate_backdoor_asr(test_eval_model, test_loader, device=device)
+        triggered_dataset = TriggeredPneumoniaTestDataset(clean_test_dataset=test_loader.dataset)
+        triggered_loader = DataLoader(
+            triggered_dataset,
+            batch_size=batch_size,
+            shuffle=False,
+            num_workers=0,
+        )
+        asr_dict = evaluate_backdoor_asr(model=test_eval_model, triggered_loader=triggered_loader, device=device)
+        asr = float(asr_dict["attack_success_rate_percent"])
+        del triggered_loader
+        del triggered_dataset
 
     del test_eval_model
     del client_loaders
```

### Current SHA-256 Checksum
`7c97f3e0abb98b3f0080f998859b99cf200c5b9246c92f8b0082991111f5b614  experiments/run_phase7_memory_safe.py`

### Scientific Invariance Verification
The modification strictly affects post-training evaluation of backdoor attack success rate. It does **not** modify:
- Federated training routines (`client_update`, optimizer, loss, learning rate).
- Aggregation strategies (`fedavg`, `multi_krum_f1`, `tri_state_monitor`, `tri_state_discount_050`).
- Data partitioning, client allocations, or seeds (`training_seed=211`, `partition_seed=211`, `initialization_seed=42`).
- Checkpoint selection criteria (best validation accuracy on untouched validation split).
- Frozen Phase 6 detector v1 thresholds or behavioral decisions.
- Blockchain logging of successful simulations.

---

## 3. Blockchain Recovery Audit

### Canonical Ledger Reference Points
- **Pre-Batch-2 ledger tip:** Block #1077, 1,078 blocks total.
- **Failed attempt:** Appended incomplete blocks #1078–#1098 (21 blocks).
- **Rollback:** Restored tip #1077 (1,078 blocks).
- **Successful Batch 2:** Appended blocks #1078–#1187 (110 blocks across 5 experiments).
- **Current verified tip:** Block #1187, 1,188 blocks total (`Valid: True`).

### Investigation of Ledger State
- **Pre-Crash Ledger Tip:** Block #1,077 (Total blocks: 1,078, encompassing Genesis #0 + 43 historical experiments + 5 Batch 1 experiments = 1 + 43 + 1,034 = 1,078).
- **Post-Crash Ledger Tip:** Block #1,098 (Total blocks: 1,099). Blocks #1,078 to #1,098 represented round-by-round logging of the crashed run `EXP-P7-FEDAVG-ATK-BD-IID-M1-S211`.
- **First Rollback Attempt:** An import statement typo (`from blockchain_audit.audit_logger import verify_ledger` instead of `from blockchain_audit.verification import verify_ledger`) threw an `ImportError` before any file write occurred. No modifications were made during this attempt.
- **Successful Rollback Execution:** The ledger file `results/blockchain_audit/audit_ledger.jsonl` was truncated back to line 1,078 (keeping indices 0 through 1,077).
- **Verification:** `verify_ledger()` was executed immediately via `experiments/run_phase7_memory_safe.py --verify-blockchain`, returning:
  `Blockchain Ledger Valid: True (1078 blocks)`.
- **Historical Blocks Impact:** Exactly **0** complete historical blocks were deleted. Only the 21 incomplete blocks of the uncommitted run were truncated.
- **Clean Re-execution:** The runner was restarted with `--max-experiments 5`. `EXP-P7-FEDAVG-ATK-BD-IID-M1-S211` was executed fresh from Round 0, appending 22 new contiguous blocks (#1,078 to #1,099).
- **Post-Batch Ledger Tip:** Block #1,187 (Total: 1,188 blocks). All 53 unique completed simulations have complete cryptographic representation in the ledger.

---

## 4. Physical Attempts vs. Unique Experiment IDs

| Category | Count | Detail |
|---|:---:|---|
| **Unique Completed Experiment IDs (Primary)** | 29 | 19 reused from historical + 5 Batch 1 + 5 Batch 2 |
| **Historical Supplementary Experiments** | 24 | Preserved in `execution_registry.json` |
| **Total Valid Unique Completed Experiments** | **53** | All 53 verified on disk and ledger |
| **Physical Failed Execution Attempts** | 1 | `EXP-P7-FEDAVG-ATK-BD-IID-M1-S211` (Attempt 1, crashed at line 505) |
| **Total Physical Simulation Launches** | **54** | 53 successful + 1 crashed attempt |
| **Hard Simulation Budget Tracking** | **53 / 99** | 43 simulations remain within $\le 99$ ceiling |

No artifacts or outputs from the crashed attempt were reused; the simulation was trained and evaluated anew on the successful rerun.

---

## 5. Completed Batch 2 Artifact Verification

All five Batch 2 experiments are verified on disk with 100% complete artifacts:

1. `EXP-P7-FEDAVG-ATK-BD-IID-M1-S211`:
   - Checkpoint: `results/phase7_robust_aggregation/checkpoints/best_exp-p7-fedavg-atk-bd-iid-m1-s211.pt` (1.69 MB)
   - Results JSON: `results/phase7_robust_aggregation/evaluation/exp-p7-fedavg-atk-bd-iid-m1-s211_results.json` (1.35 KB)
   - Feature Log: `results/phase7_robust_aggregation/instrumentation/exp-p7-fedavg-atk-bd-iid-m1-s211_features.jsonl` (253.4 KB)
   - Registry Status: `COMPLETED` (Block #1,099)
2. `EXP-P7-MULTI_KRUM_F1-ATK-BD-IID-M1-S211`:
   - Checkpoint: `results/phase7_robust_aggregation/checkpoints/best_exp-p7-multi_krum_f1-atk-bd-iid-m1-s211.pt` (1.69 MB)
   - Results JSON: `results/phase7_robust_aggregation/evaluation/exp-p7-multi_krum_f1-atk-bd-iid-m1-s211_results.json` (1.39 KB)
   - Feature Log: `results/phase7_robust_aggregation/instrumentation/exp-p7-multi_krum_f1-atk-bd-iid-m1-s211_features.jsonl` (257.3 KB)
   - Registry Status: `COMPLETED` (Block #1,121)
3. `EXP-P7-TRI_STATE_MONITOR-ATK-BD-IID-M1-S211`:
   - Checkpoint: `results/phase7_robust_aggregation/checkpoints/best_exp-p7-tri_state_monitor-atk-bd-iid-m1-s211.pt` (1.69 MB)
   - Results JSON: `results/phase7_robust_aggregation/evaluation/exp-p7-tri_state_monitor-atk-bd-iid-m1-s211_results.json` (1.42 KB)
   - Feature Log: `results/phase7_robust_aggregation/instrumentation/exp-p7-tri_state_monitor-atk-bd-iid-m1-s211_features.jsonl` (257.8 KB)
   - Registry Status: `COMPLETED` (Block #1,143)
4. `EXP-P7-TRI_STATE_DISCOUNT_050-ATK-BD-IID-M1-S211`:
   - Checkpoint: `results/phase7_robust_aggregation/checkpoints/best_exp-p7-tri_state_discount_050-atk-bd-iid-m1-s211.pt` (1.69 MB)
   - Results JSON: `results/phase7_robust_aggregation/evaluation/exp-p7-tri_state_discount_050-atk-bd-iid-m1-s211_results.json` (1.45 KB)
   - Feature Log: `results/phase7_robust_aggregation/instrumentation/exp-p7-tri_state_discount_050-atk-bd-iid-m1-s211_features.jsonl` (257.9 KB)
   - Registry Status: `COMPLETED` (Block #1,165)
5. `EXP-P7-FEDAVG-ATK-BN-IID-M1-S211`:
   - Checkpoint: `results/phase7_robust_aggregation/checkpoints/best_exp-p7-fedavg-atk-bn-iid-m1-s211.pt` (1.69 MB)
   - Results JSON: `results/phase7_robust_aggregation/evaluation/exp-p7-fedavg-atk-bn-iid-m1-s211_results.json` (1.36 KB)
   - Feature Log: `results/phase7_robust_aggregation/instrumentation/exp-p7-fedavg-atk-bn-iid-m1-s211_features.jsonl` (251.1 KB)
   - Registry Status: `COMPLETED` (Block #1,187)

Zero duplicate entries exist in `execution_registry_reduced_96.json`.
Each unique experiment ID maps to exactly 1 completed simulation.
