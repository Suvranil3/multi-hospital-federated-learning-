"""
Phase 6: Multi-Factor Malicious Hospital Detection Evaluator.
Calibrates thresholds on honest control runs, evaluates held-out honest
and adversarial test suites, generates statistical reports and publication figures.

Rigorous Dual Reporting:
1. Out-of-Sample Calibration-Excluded Benchmark (9 Simulations, 1,800 observations: 1,640 honest, 160 malicious)
2. In-Sample Benchmark (11 Simulations, 2,200 observations: 2,040 honest, 160 malicious)
3. Independent Multi-Seed Honest Generalization (Seeds 43, 44, 45 across IID and Non-IID alpha in {1.0, 0.5, 0.1})
"""

import sys
from pathlib import Path
import json
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from federated_learning.detection.detector import (
    TriStateDetector,
    DetectorThresholds,
    DetectionDecision,
    ClientEvaluationResult,
)
from federated_learning.detection.metrics import (
    compute_detection_metrics,
    compute_stratified_fpr,
)

PHASE6_DIR = PROJECT_ROOT / "results" / "phase6_detector"
PHASE6_INSTRUMENTATION_DIR = PHASE6_DIR / "instrumentation"
PHASE6_EVALUATION_DIR = PHASE6_DIR / "evaluation"
PHASE6_FIGURES_DIR = PHASE6_DIR / "figures"


# Ground truth mapping across all simulations (STRICTLY for post-hoc validation)
SIMULATION_METADATA = {
    # 2 Calibration Controls (Phase 5)
    "exp-med-006-ctrl-iid": {"scenario": "iid", "malicious_cids": set(), "attack_type": "none", "role": "calibration"},
    "exp-med-007-ctrl-noniid-a05": {"scenario": "non_iid_alpha0.5", "malicious_cids": set(), "attack_type": "none", "role": "calibration"},
    # 2 Held-Out Honest Controls (Phase 4 replayed)
    "exp-med-003-noniid-alpha10": {"scenario": "non_iid_alpha1.0", "malicious_cids": set(), "attack_type": "none", "role": "evaluation"},
    "exp-med-005-noniid-alpha01": {"scenario": "non_iid_alpha0.1", "malicious_cids": set(), "attack_type": "none", "role": "evaluation"},
    # 7 Phase 5 Attacks
    "exp-med-008-atk-lf-iid-m1": {"scenario": "iid", "malicious_cids": {"0"}, "attack_type": "label_flipping", "role": "evaluation"},
    "exp-med-009-atk-lf-iid-m2": {"scenario": "iid", "malicious_cids": {"0", "1"}, "attack_type": "label_flipping", "role": "evaluation"},
    "exp-med-010-atk-lf-noniid-m1": {"scenario": "non_iid_alpha0.5", "malicious_cids": {"0"}, "attack_type": "label_flipping", "role": "evaluation"},
    "exp-med-011-atk-mp-inv-iid-m1": {"scenario": "iid", "malicious_cids": {"0"}, "attack_type": "model_poisoning_invert", "role": "evaluation"},
    "exp-med-012-atk-mp-scale-iid-m1": {"scenario": "iid", "malicious_cids": {"0"}, "attack_type": "model_poisoning_scale", "role": "evaluation"},
    "exp-med-013-atk-bd-iid-m1": {"scenario": "iid", "malicious_cids": {"0"}, "attack_type": "backdoor_iid", "role": "evaluation"},
    "exp-med-014-atk-bd-noniid-m1": {"scenario": "non_iid_alpha0.5", "malicious_cids": {"0"}, "attack_type": "backdoor_noniid", "role": "evaluation"},
}


def load_all_feature_records() -> Dict[str, List[Dict[str, Any]]]:
    """Load extracted feature records from JSONL files."""
    records_by_exp: Dict[str, List[Dict[str, Any]]] = {}
    for jsonl_file in sorted(PHASE6_INSTRUMENTATION_DIR.glob("*_features.jsonl")):
        exp_prefix = jsonl_file.stem.replace("_features", "")
        records = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))
        records_by_exp[exp_prefix] = records
    return records_by_exp


def evaluate_cohort(
    detector: TriStateDetector,
    exp_ids: List[str],
    records_by_exp: Dict[str, List[Dict[str, Any]]],
) -> Tuple[List[ClientEvaluationResult], List[int], List[str], List[str], Dict[str, Any]]:
    """Evaluate a specific list of experiment IDs with the detector."""
    eval_results: List[ClientEvaluationResult] = []
    ground_truth: List[int] = []
    scenario_tags: List[str] = []
    exp_tags: List[str] = []
    attack_detections = defaultdict(lambda: {"total": 0, "detected_malicious": 0, "detected_suspicious": 0, "detected_normal": 0})
    latency_tracker = defaultdict(lambda: {"first_suspicious": None, "first_malicious": None})

    for exp_id in exp_ids:
        records = records_by_exp.get(exp_id, [])
        meta = SIMULATION_METADATA.get(exp_id, {
            "scenario": "unknown",
            "malicious_cids": set(),
            "attack_type": "unknown",
        })
        scenario = meta["scenario"]
        malicious_cids = meta["malicious_cids"]

        for rec in records:
            res = detector.evaluate_client(rec)
            cid = str(rec["client_id"])
            rnd = int(rec["round"])
            is_mal = int(cid in malicious_cids)

            eval_results.append(res)
            ground_truth.append(is_mal)
            scenario_tags.append(scenario)
            exp_tags.append(exp_id)

            if is_mal:
                attack_detections[exp_id]["total"] += 1
                if res.decision == DetectionDecision.MALICIOUS:
                    attack_detections[exp_id]["detected_malicious"] += 1
                    if latency_tracker[exp_id]["first_malicious"] is None:
                        latency_tracker[exp_id]["first_malicious"] = rnd
                elif res.decision == DetectionDecision.SUSPICIOUS:
                    attack_detections[exp_id]["detected_suspicious"] += 1
                    if latency_tracker[exp_id]["first_suspicious"] is None:
                        latency_tracker[exp_id]["first_suspicious"] = rnd
                else:
                    attack_detections[exp_id]["detected_normal"] += 1

    per_attack_summary = {}
    for exp_id, stats in attack_detections.items():
        if stats["total"] > 0:
            rec_rate = stats["detected_malicious"] / stats["total"]
            lat = latency_tracker[exp_id]
            per_attack_summary[exp_id] = {
                **stats,
                "recall": round(rec_rate, 4),
                "flagged_recall": round((stats["detected_malicious"] + stats["detected_suspicious"]) / stats["total"], 4),
                "latency": lat,
            }

    return eval_results, ground_truth, scenario_tags, exp_tags, per_attack_summary


def run_phase6_evaluation():
    PHASE6_EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    PHASE6_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    records_by_exp = load_all_feature_records()
    if not records_by_exp:
        raise FileNotFoundError(f"No instrumentation records found in {PHASE6_INSTRUMENTATION_DIR}")

    print("\n" + "=" * 105)
    print("  PHASE 6 MALICIOUS HOSPITAL DETECTION BENCHMARK & EVALUATION")
    print("=" * 105)

    # 1. Calibration on 2 Honest Calibration Controls (400 observations)
    calib_runs = ["exp-med-006-ctrl-iid", "exp-med-007-ctrl-noniid-a05"]
    calib_records: List[Dict[str, Any]] = []
    for exp_id in calib_runs:
        if exp_id in records_by_exp:
            calib_records.extend(records_by_exp[exp_id])

    print(f"\n[Step 1/5] Calibrating Tri-State Thresholds on Honest Calibration Runs ({len(calib_records)} records)...")
    detector = TriStateDetector()
    calibrated_thresholds = detector.calibrate_from_honest_observations(
        honest_feature_records=calib_records,
        target_malicious_fpr=0.01,
        target_suspicious_fpr=0.05,
    )

    print("  Calibrated Thresholds (Derived strictly from honest calibration controls):")
    print(f"    • Global Z_mag Thresholds:       Suspicious={calibrated_thresholds.z_mag_suspicious:.2f} | Malicious={calibrated_thresholds.z_mag_malicious:.2f}")
    print(f"    • Layer-wise Max Z_mag:          Suspicious={calibrated_thresholds.max_layer_z_suspicious:.2f} | Malicious={calibrated_thresholds.max_layer_z_malicious:.2f}")
    print(f"    • Cosine Similarity Thresholds:  Suspicious={calibrated_thresholds.cos_suspicious:.4f} | Catastrophic Invert={calibrated_thresholds.cos_strong_negative:.2f}")
    print(f"    • Total Val Loss Degradation ΔL: Suspicious={calibrated_thresholds.delta_loss_suspicious:.4f} | Malicious={calibrated_thresholds.delta_loss_malicious:.4f}")
    print(f"    • Historical Trajectory EMA:     Suspicious={calibrated_thresholds.ema_accumulator_suspicious:.4f} | Malicious={calibrated_thresholds.ema_accumulator_malicious:.4f}")

    # 2. Evaluation Cohort 1: Calibration-Excluded Evaluation (Strict Out-of-Sample, 9 Runs, 1,800 Obs)
    eval_excluded_runs = [
        "exp-med-003-noniid-alpha10",
        "exp-med-005-noniid-alpha01",
        "exp-med-008-atk-lf-iid-m1",
        "exp-med-009-atk-lf-iid-m2",
        "exp-med-010-atk-lf-noniid-m1",
        "exp-med-011-atk-mp-inv-iid-m1",
        "exp-med-012-atk-mp-scale-iid-m1",
        "exp-med-013-atk-bd-iid-m1",
        "exp-med-014-atk-bd-noniid-m1",
    ]
    print(f"\n[Step 2/5] Evaluating Calibration-Excluded Cohort ({len(eval_excluded_runs)} runs, 1,800 observations)...")
    results_exc, gt_exc, scen_exc, exp_exc, per_attack_exc = evaluate_cohort(detector, eval_excluded_runs, records_by_exp)
    metrics_exc = compute_detection_metrics(results_exc, gt_exc)
    strat_fpr_exc = compute_stratified_fpr(results_exc, gt_exc, scen_exc)

    # 3. Evaluation Cohort 2: In-Sample Reference Benchmark (All 11 Runs, 2,200 Obs)
    all_11_runs = calib_runs + eval_excluded_runs
    print(f"\n[Step 3/5] Evaluating In-Sample Reference Cohort ({len(all_11_runs)} runs, 2,200 observations)...")
    results_all, gt_all, scen_all, exp_all, per_attack_all = evaluate_cohort(detector, all_11_runs, records_by_exp)
    metrics_all = compute_detection_metrics(results_all, gt_all)
    strat_fpr_all = compute_stratified_fpr(results_all, gt_all, scen_all)

    # 4. Evaluation Cohort 3: Independent Multi-Seed Honest Evaluation
    independent_honest_runs = sorted([k for k in records_by_exp if "honest" in k])
    independent_stats = {}
    if independent_honest_runs:
        print(f"\n[Step 4/5] Evaluating Independent Multi-Seed Honest Runs ({len(independent_honest_runs)} simulations)...")
        indep_results = []
        indep_gt = []
        indep_scen = []
        for r_id in independent_honest_runs:
            for rec in records_by_exp[r_id]:
                indep_results.append(detector.evaluate_client(rec))
                indep_gt.append(0)
                if "IID" in r_id.upper():
                    indep_scen.append("iid")
                elif "A10" in r_id.upper():
                    indep_scen.append("non_iid_alpha1.0")
                elif "A05" in r_id.upper():
                    indep_scen.append("non_iid_alpha0.5")
                elif "A01" in r_id.upper():
                    indep_scen.append("non_iid_alpha0.1")
                else:
                    indep_scen.append("unknown")
        independent_stats = compute_stratified_fpr(indep_results, indep_gt, indep_scen)

    # Print Comparative Dual-Evaluation Table
    cm_exc = metrics_exc["malicious_decision_metrics"]
    cm_all = metrics_all["malicious_decision_metrics"]

    print("\n" + "=" * 105)
    print("  COMPARATIVE EVALUATION SUMMARY: CALIBRATION-EXCLUDED (OUT-OF-SAMPLE) VS IN-SAMPLE")
    print("=" * 105)
    print(f"{'Metric':<35} | {'Calibration-Excluded (9 Runs)':<30} | {'In-Sample (11 Runs)':<25}")
    print("-" * 105)
    print(f"{'Total Observations':<35} | {metrics_exc['total_observations']:<30} | {metrics_all['total_observations']:<25}")
    print(f"{'Honest Observations':<35} | {metrics_exc['total_honest']:<30} | {metrics_all['total_honest']:<25}")
    print(f"{'Malicious Observations':<35} | {metrics_exc['total_malicious']:<30} | {metrics_all['total_malicious']:<25}")
    print(f"{'True Positives (Malicious Detected)':<35} | {cm_exc['true_positives']:>4} / {metrics_exc['total_malicious']:<24} | {cm_all['true_positives']:>4} / {metrics_all['total_malicious']:<20}")
    print(f"{'False Positives (Honest Penalized)':<35} | {cm_exc['false_positives']:>4} / {metrics_exc['total_honest']:<24} | {cm_all['false_positives']:>4} / {metrics_all['total_honest']:<20}")
    print(f"{'TPR / Malicious Recall':<35} | {cm_exc['tpr_recall']*100:6.2f}%{'':<23} | {cm_all['tpr_recall']*100:6.2f}%{'':<18}")
    print(f"{'False Malicious Rate (FPR)':<35} | {cm_exc['fpr']*100:6.2f}%{'':<23} | {cm_all['fpr']*100:6.2f}%{'':<18}")
    print(f"{'Precision':<35} | {cm_exc['precision']*100:6.2f}%{'':<23} | {cm_all['precision']*100:6.2f}%{'':<18}")
    print(f"{'F1-Score':<35} | {cm_exc['f1_score']*100:6.2f}%{'':<23} | {cm_all['f1_score']*100:6.2f}%{'':<18}")
    print(f"{'AUROC (Continuous Anomaly Score)':<35} | {metrics_exc['continuous_score_metrics']['auroc']:<30.4f} | {metrics_all['continuous_score_metrics']['auroc']:<25.4f}")
    print(f"{'AUPRC (Precision-Recall)':<35} | {metrics_exc['continuous_score_metrics']['auprc']:<30.4f} | {metrics_all['continuous_score_metrics']['auprc']:<25.4f}")
    print(f"{'Total Flagged Recall (Sus or Mal)':<35} | {metrics_exc['flagged_decision_metrics']['tpr_recall']*100:6.2f}%{'':<23} | {metrics_all['flagged_decision_metrics']['tpr_recall']*100:6.2f}%{'':<18}")
    print(f"{'Total Flagged FPR (Sus or Mal)':<35} | {metrics_exc['flagged_decision_metrics']['fpr']*100:6.2f}%{'':<23} | {metrics_all['flagged_decision_metrics']['fpr']*100:6.2f}%{'':<18}")
    print("-" * 105)

    # Print Stratified FPR Table (Calibration-Excluded)
    print("\n" + "-" * 105)
    print("  STRATIFIED FALSE POSITIVE EVALUATION (CALIBRATION-EXCLUDED COHORT)")
    print("-" * 105)
    print(f"{'Scenario / Setting':<25} | {'Honest Obs':<11} | {'False Malicious':<16} | {'Suspicious (Monitored)':<23}")
    print("-" * 105)
    for scen, stat in strat_fpr_exc.items():
        print(
            f"{scen:<25} | {stat['total_honest_observations']:<11} | "
            f"{stat['false_malicious_count']:>4} ({stat['false_malicious_rate']*100:5.2f}%)   | "
            f"{stat['suspicious_count']:>4} ({stat['suspicious_rate']*100:5.2f}%)"
        )
    print("-" * 105)

    if independent_stats:
        print("\n" + "-" * 105)
        print("  STRATIFIED FALSE POSITIVE EVALUATION (INDEPENDENT MULTI-SEED HONEST RUNS)")
        print("-" * 105)
        print(f"{'Scenario / Setting':<25} | {'Honest Obs':<11} | {'False Malicious':<16} | {'Suspicious (Monitored)':<23}")
        print("-" * 105)
        for scen, stat in independent_stats.items():
            print(
                f"{scen:<25} | {stat['total_honest_observations']:<11} | "
                f"{stat['false_malicious_count']:>4} ({stat['false_malicious_rate']*100:5.2f}%)   | "
                f"{stat['suspicious_count']:>4} ({stat['suspicious_rate']*100:5.2f}%)"
            )
        print("-" * 105)

    # Print Per-Attack Breakdown
    print("\n" + "-" * 105)
    print("  PER-ATTACK DETECTION BREAKDOWN & TEMPORAL LATENCY")
    print("-" * 105)
    print(f"{'Experiment ID':<32} | {'Mal Obs':<8} | {'Malicious':<12} | {'Suspicious':<12} | {'Recall':<8} | {'Latency (Rd)'}")
    print("-" * 105)
    for exp_id, stats in per_attack_exc.items():
        rec_rate = stats["recall"]
        lat = stats["latency"]
        lat_str = f"Sus: Rd {lat['first_suspicious'] or 'N/A'}, Mal: Rd {lat['first_malicious'] or 'N/A'}"
        print(
            f"{exp_id:<32} | {stats['total']:<8} | "
            f"{stats['detected_malicious']:>4} ({rec_rate*100:5.1f}%) | "
            f"{stats['detected_suspicious']:>4} ({(stats['detected_suspicious']/stats['total'])*100:5.1f}%) | "
            f"{rec_rate*100:5.1f}% | {lat_str}"
        )
    print("-" * 105)

    # 5. Generate Publication Figures
    print("\n[Step 5/5] Generating Publication Figures...")
    fig_path = PHASE6_FIGURES_DIR / "phase6_detector_performance.png"
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: ROC Curves (Calibration-Excluded vs In-Sample)
    y_true_exc = np.array(gt_exc)
    scores_exc = np.array([r.anomaly_score for r in results_exc])
    fpr_c_exc, tpr_c_exc, _ = roc_curve(y_true_exc, scores_exc)
    auroc_exc = metrics_exc["continuous_score_metrics"]["auroc"]

    y_true_all = np.array(gt_all)
    scores_all = np.array([r.anomaly_score for r in results_all])
    fpr_c_all, tpr_c_all, _ = roc_curve(y_true_all, scores_all)
    auroc_all = metrics_all["continuous_score_metrics"]["auroc"]

    axes[0, 0].plot(fpr_c_exc, tpr_c_exc, color="#1f77b4", lw=2.5, label=f"Out-of-Sample (AUROC = {auroc_exc:.4f})")
    axes[0, 0].plot(fpr_c_all, tpr_c_all, color="#aec7e8", lw=1.8, linestyle="--", label=f"In-Sample (AUROC = {auroc_all:.4f})")
    axes[0, 0].plot([0, 1], [0, 1], color="grey", linestyle=":", lw=1.5, label="Random Guess")
    axes[0, 0].set_title("(A) ROC Curves (Out-of-Sample vs In-Sample)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("False Positive Rate (FPR)", fontsize=11)
    axes[0, 0].set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # Panel B: Precision-Recall Curves
    prec_c_exc, rec_c_exc, _ = precision_recall_curve(y_true_exc, scores_exc)
    auprc_exc = metrics_exc["continuous_score_metrics"]["auprc"]

    prec_c_all, rec_c_all, _ = precision_recall_curve(y_true_all, scores_all)
    auprc_all = metrics_all["continuous_score_metrics"]["auprc"]

    axes[0, 1].plot(rec_c_exc, prec_c_exc, color="#2ca02c", lw=2.5, label=f"Out-of-Sample (AUPRC = {auprc_exc:.4f})")
    axes[0, 1].plot(rec_c_all, prec_c_all, color="#98df8a", lw=1.8, linestyle="--", label=f"In-Sample (AUPRC = {auprc_all:.4f})")
    axes[0, 1].set_title("(B) Precision-Recall Curves (8.9% Prevalence)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Recall (TPR)", fontsize=11)
    axes[0, 1].set_ylabel("Precision", fontsize=11)
    axes[0, 1].legend(loc="lower left")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # Panel C: Stratified FPR across Data Heterogeneity (Out-of-Sample)
    scenarios = list(strat_fpr_exc.keys())
    mal_rates = [strat_fpr_exc[s]["false_malicious_rate"] * 100 for s in scenarios]
    sus_rates = [strat_fpr_exc[s]["suspicious_rate"] * 100 for s in scenarios]
    x_pos = np.arange(len(scenarios))
    w = 0.35

    axes[1, 0].bar(x_pos - w/2, mal_rates, width=w, color="#d62728", alpha=0.85, label="False Malicious Rate (%)")
    axes[1, 0].bar(x_pos + w/2, sus_rates, width=w, color="#ff7f0e", alpha=0.85, label="Suspicious Monitored Rate (%)")
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([s.replace("non_iid_", "") for s in scenarios], rotation=15, ha="right")
    axes[1, 0].set_ylabel("False Flag Rate (%)", fontsize=11)
    axes[1, 0].set_title("(C) Out-of-Sample False Alarms by Heterogeneity", fontsize=12, fontweight="bold")
    axes[1, 0].legend(loc="upper left")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    # Panel D: Per-Attack Detection Recall (Tri-State: Malicious + Suspicious)
    attack_names = list(per_attack_exc.keys())
    mal_recalls = [per_attack_exc[a]["recall"] * 100 for a in attack_names]
    sus_recalls = [(per_attack_exc[a]["detected_suspicious"] / per_attack_exc[a]["total"]) * 100 for a in attack_names]
    short_names = [a.replace("exp-med-", "").replace("-atk", "").upper() for a in attack_names]
    y_pos = np.arange(len(attack_names))

    axes[1, 1].barh(y_pos, mal_recalls, color="#7b1fa2", alpha=0.9, label="Malicious (Excluded)")
    axes[1, 1].barh(y_pos, sus_recalls, left=mal_recalls, color="#ba68c8", alpha=0.75, label="Suspicious (Alerted)")
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels(short_names, fontsize=9)
    axes[1, 1].set_xlabel("Detection Recall (% Malicious Rounds Flagged)", fontsize=11)
    axes[1, 1].set_title("(D) Tri-State Detection by Attack Category", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlim([0, 105])
    axes[1, 1].legend(loc="lower right")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Publication figure saved to {fig_path}")

    # 6. Save Structured Evaluation JSON
    eval_json_path = PHASE6_EVALUATION_DIR / "detection_results.json"
    full_output = {
        "calibrated_thresholds": {
            "z_mag_suspicious": calibrated_thresholds.z_mag_suspicious,
            "z_mag_malicious": calibrated_thresholds.z_mag_malicious,
            "max_layer_z_suspicious": calibrated_thresholds.max_layer_z_suspicious,
            "max_layer_z_malicious": calibrated_thresholds.max_layer_z_malicious,
            "cos_suspicious": calibrated_thresholds.cos_suspicious,
            "cos_strong_negative": calibrated_thresholds.cos_strong_negative,
            "cos_malicious_negative": calibrated_thresholds.cos_malicious_negative,
            "delta_loss_suspicious": calibrated_thresholds.delta_loss_suspicious,
            "delta_loss_malicious": calibrated_thresholds.delta_loss_malicious,
            "ema_accumulator_suspicious": calibrated_thresholds.ema_accumulator_suspicious,
            "ema_accumulator_malicious": calibrated_thresholds.ema_accumulator_malicious,
        },
        "calibration_excluded_evaluation": {
            "description": "Strict out-of-sample evaluation on 9 simulations (1,800 records: 1,640 honest, 160 malicious)",
            "metrics": metrics_exc,
            "stratified_fpr": strat_fpr_exc,
            "per_attack": per_attack_exc,
        },
        "in_sample_reference_evaluation": {
            "description": "Full 11-simulation cohort including the 2 calibration runs (2,200 records: 2,040 honest, 160 malicious)",
            "metrics": metrics_all,
            "stratified_fpr": strat_fpr_all,
            "per_attack": per_attack_all,
        },
        "independent_multi_seed_honest_evaluation": independent_stats,
    }

    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"  Comprehensive evaluation report saved to {eval_json_path}")


if __name__ == "__main__":
    run_phase6_evaluation()
