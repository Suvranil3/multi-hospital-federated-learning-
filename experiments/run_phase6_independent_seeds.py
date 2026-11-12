"""
Phase 6: Multi-Seed Independent Honest Benchmark Runner.
Executes 12 independent honest simulations across 4 heterogeneity conditions
(IID, Non-IID alpha=1.0, alpha=0.5, alpha=0.1) using 3 independent partition/training seeds
(43, 44, 45) to rigorously evaluate honest false-positive rates without seed-42 bias.

Initialization hash remains constant:
0c411b83c2f0acef58beb5b9a968030d877e079eb2356eb3748903f8ae9bbc83
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import argparse
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ray
from experiments.run_phase6_instrumentation import run_instrumented_simulation

PHASE6_DIR = PROJECT_ROOT / "results" / "phase6_detector"
PHASE6_INSTRUMENTATION_DIR = PHASE6_DIR / "instrumentation"
PHASE6_CHECKPOINTS_DIR = PHASE6_DIR / "checkpoints"


INDEPENDENT_SEEDS = [43, 44, 45]

INDEPENDENT_HONEST_MATRIX: List[Dict[str, Any]] = []

# Generate 12 independent honest simulation configurations
for seed in INDEPENDENT_SEEDS:
    # 1. IID
    INDEPENDENT_HONEST_MATRIX.append({
        "experiment_id": f"EXP-MED-HONEST-IID-S{seed}",
        "scenario": "iid",
        "alpha": None,
        "attack_type": "none",
        "malicious_client_ids": [],
        "partition_seed": seed,
        "training_seed": seed,
        "initialization_seed": 42,
    })
    # 2. Non-IID alpha=1.0
    INDEPENDENT_HONEST_MATRIX.append({
        "experiment_id": f"EXP-MED-HONEST-A10-S{seed}",
        "scenario": "non_iid",
        "alpha": 1.0,
        "attack_type": "none",
        "malicious_client_ids": [],
        "partition_seed": seed,
        "training_seed": seed,
        "initialization_seed": 42,
    })
    # 3. Non-IID alpha=0.5
    INDEPENDENT_HONEST_MATRIX.append({
        "experiment_id": f"EXP-MED-HONEST-A05-S{seed}",
        "scenario": "non_iid",
        "alpha": 0.5,
        "attack_type": "none",
        "malicious_client_ids": [],
        "partition_seed": seed,
        "training_seed": seed,
        "initialization_seed": 42,
    })
    # 4. Non-IID alpha=0.1
    INDEPENDENT_HONEST_MATRIX.append({
        "experiment_id": f"EXP-MED-HONEST-A01-S{seed}",
        "scenario": "non_iid",
        "alpha": 0.1,
        "attack_type": "none",
        "malicious_client_ids": [],
        "partition_seed": seed,
        "training_seed": seed,
        "initialization_seed": 42,
    })


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Multi-Seed Independent Honest Suite")
    parser.add_argument("--run-id", type=str, default=None, help="Run specific simulation by ID")
    args = parser.parse_args()

    targets = [cfg for cfg in INDEPENDENT_HONEST_MATRIX if cfg["experiment_id"] == args.run_id] if args.run_id else INDEPENDENT_HONEST_MATRIX

    print(f"\n==========================================================================================")
    print(f"  PHASE 6 INDEPENDENT MULTI-SEED HONEST BENCHMARK SUITE")
    print(f"  Executing {len(targets)} Independent Simulations (Seeds: {INDEPENDENT_SEEDS})")
    print(f"  Total planned honest client-round records: {len(targets) * 20 * 10:,}")
    print(f"==========================================================================================")

    suite_results = []
    total_start = datetime.now()

    for i, cfg in enumerate(targets, 1):
        jsonl_path = PHASE6_INSTRUMENTATION_DIR / f"{cfg['experiment_id'].lower()}_features.jsonl"
        best_ckpt = PHASE6_CHECKPOINTS_DIR / f"best_{cfg['experiment_id'].lower()}.pt"

        # Check if already completed
        if jsonl_path.exists() and best_ckpt.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                lines = [l for l in f if l.strip()]
            if len(lines) >= 200:
                print(f"\n>>> Simulation {i}/{len(targets)}: {cfg['experiment_id']} already completed ({len(lines)} records). Skipping.")
                continue

        print(f"\n>>> Launching Simulation {i}/{len(targets)}: {cfg['experiment_id']} (Seed {cfg['partition_seed']})...")
        try:
            res = run_instrumented_simulation(**cfg)
            suite_results.append(res)
        finally:
            if ray.is_initialized():
                ray.shutdown()

    total_time = (datetime.now() - total_start).total_seconds()
    print(f"\n==========================================================================================")
    print(f"  PHASE 6 INDEPENDENT MULTI-SEED SUITE COMPLETED in {total_time/60:.2f} minutes.")
    print(f"  Feature streams recorded in {PHASE6_INSTRUMENTATION_DIR}")
    print(f"==========================================================================================")

    summary_path = PHASE6_DIR / "independent_honest_suite_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_simulations": len(targets),
            "seeds": INDEPENDENT_SEEDS,
            "total_time_seconds": total_time,
            "simulations": suite_results,
        }, f, indent=2)
    print(f"  Independent suite summary saved to {summary_path}")


if __name__ == "__main__":
    main()
