"""
Phase 6: Replay Reproducibility Verification Script.
Compares checkpoints and validation trajectories from instrumented simulations
against the frozen Phase 4 and Phase 5 artifacts to detect any numerical drift.
"""

import sys
from pathlib import Path
import json
import torch
import numpy as np
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PHASE6_DIR = PROJECT_ROOT / "results" / "phase6_detector"
PHASE6_CHECKPOINTS_DIR = PHASE6_DIR / "checkpoints"
PHASE6_VERIFICATION_DIR = PHASE6_DIR / "verification"
RESULTS_MODELS_DIR = PROJECT_ROOT / "results" / "models"
RESULTS_LOGS_DIR = PROJECT_ROOT / "results" / "logs"


def verify_checkpoints() -> Dict[str, Any]:
    """Verify exact parameter hash and numerical parity between original and instrumented runs."""
    PHASE6_VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    # Mapping of instrumented experiment ID to original Phase 4 / Phase 5 artifacts
    comparison_targets = [
        # Phase 5 Honest Controls
        ("EXP-MED-006-CTRL-IID", "exp_med_006_ctrl_iid"),
        ("EXP-MED-007-CTRL-NONIID-A05", "exp_med_007_ctrl_noniid_a05"),
        # Phase 4 Replayed Controls
        ("EXP-MED-003-NONIID-ALPHA10", "federated_non_iid_alpha10"),
        ("EXP-MED-005-NONIID-ALPHA01", "federated_non_iid_alpha01"),
        # Phase 5 Attacks
        ("EXP-MED-008-ATK-LF-IID-M1", "exp_med_008_atk_lf_iid_m1"),
        ("EXP-MED-009-ATK-LF-IID-M2", "exp_med_009_atk_lf_iid_m2"),
        ("EXP-MED-010-ATK-LF-NONIID-M1", "exp_med_010_atk_lf_noniid_m1"),
        ("EXP-MED-011-ATK-MP-INV-IID-M1", "exp_med_011_atk_mp_inv_iid_m1"),
        ("EXP-MED-012-ATK-MP-SCALE-IID-M1", "exp_med_012_atk_mp_scale_iid_m1"),
        ("EXP-MED-013-ATK-BD-IID-M1", "exp_med_013_atk_bd_iid_m1"),
        ("EXP-MED-014-ATK-BD-NONIID-M1", "exp_med_014_atk_bd_noniid_m1"),
    ]

    report_entries = []
    print("\n" + "=" * 105)
    print("  PHASE 6 REPRODUCIBILITY AUDIT: Comparing Instrumented Replays vs Frozen Artifacts")
    print("=" * 105)
    print(f"{'Experiment ID':<30} | {'Orig Best':<10} | {'Replay Best':<11} | {'Max Param Diff':<15} | {'Val Acc Match':<13} | {'Status'}")
    print("-" * 105)

    all_matched = True

    for exp_id, orig_prefix in comparison_targets:
        inst_best_path = PHASE6_CHECKPOINTS_DIR / f"best_{exp_id.lower()}.pt"
        if not inst_best_path.exists():
            continue

        # Load instrumented checkpoint
        inst_ckpt = torch.load(inst_best_path, map_location="cpu", weights_only=False)
        inst_sd = inst_ckpt["model_state_dict"]
        inst_val_acc = inst_ckpt.get("best_val_acc", 0.0)

        # Locate original checkpoint
        possible_paths = [
            RESULTS_MODELS_DIR / f"best_{orig_prefix}_model.pt",
            RESULTS_MODELS_DIR / f"best_{orig_prefix}.pt",
            RESULTS_MODELS_DIR / f"{orig_prefix}_model.pt",
            RESULTS_MODELS_DIR / f"{orig_prefix}.pt",
        ]
        orig_ckpt_path = next((p for p in possible_paths if p.exists()), None)

        if orig_ckpt_path is None:
            print(f"{exp_id:<30} | Missing original model for prefix {orig_prefix}")
            continue

        orig_ckpt = torch.load(orig_ckpt_path, map_location="cpu", weights_only=False)
        orig_sd = orig_ckpt["model_state_dict"]
        orig_val_acc = orig_ckpt.get("val_accuracy", orig_ckpt.get("val_acc", orig_ckpt.get("best_val_accuracy", orig_ckpt.get("best_val_acc", 0.0))))

        # Check trainable parameter-by-parameter absolute maximum difference
        trainable_max_diff = 0.0
        for k in orig_sd:
            if k in inst_sd and "num_batches_tracked" not in k:
                diff = torch.max(torch.abs(orig_sd[k].float() - inst_sd[k].float())).item()
                if diff > trainable_max_diff:
                    trainable_max_diff = diff

        val_acc_diff = abs(inst_val_acc - orig_val_acc)
        is_exact = (trainable_max_diff < 1e-5) and (val_acc_diff < 0.05)
        is_consistent = (val_acc_diff < 2.0) and (trainable_max_diff < 0.25)
        status = "BIT-EXACT" if trainable_max_diff == 0.0 else ("REPRODUCIBLE" if is_exact else ("STOCHASTIC-PARITY" if is_consistent else "DEVIATION"))

        if not is_exact:
            all_matched = False

        print(
            f"{exp_id:<30} | {orig_val_acc:8.2f}% | {inst_val_acc:9.2f}% | "
            f"{trainable_max_diff:15.2e} | {val_acc_diff:11.2f}% | {status}"
        )

        report_entries.append({
            "experiment_id": exp_id,
            "original_best_val_acc": orig_val_acc,
            "replayed_best_val_acc": inst_val_acc,
            "trainable_max_diff": trainable_max_diff,
            "val_acc_diff": val_acc_diff,
            "status": status,
        })

    print("-" * 105)
    overall_status = "PASSED" if all_matched else "INVESTIGATE"
    print(f"  Overall Verification Status: {overall_status}")
    print("=" * 105)

    report_path = PHASE6_VERIFICATION_DIR / "verification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "audit_timestamp": str(np.datetime64("now")),
            "overall_status": overall_status,
            "simulations": report_entries,
        }, f, indent=2)

    return {"overall_status": overall_status, "report": report_entries}


if __name__ == "__main__":
    verify_checkpoints()
