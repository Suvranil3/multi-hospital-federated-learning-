"""
Phase 6: Evaluation Metrics and Statistical Analysis for Federated Malicious Detection.
Computes TPR, FPR, Precision, F1, AUROC, AUPRC, stratified FPR, and per-attack breakdown.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
)

from federated_learning.detection.detector import ClientEvaluationResult, DetectionDecision


def compute_detection_metrics(
    eval_results: List[ClientEvaluationResult],
    ground_truth_labels: List[int],  # 1 for malicious, 0 for honest
) -> Dict[str, Any]:
    """
    Compute comprehensive detection metrics on evaluated client-round records.

    Args:
        eval_results: List of ClientEvaluationResult objects.
        ground_truth_labels: List of binary integer ground-truth labels (1=malicious, 0=honest).

    Returns:
        Dictionary of statistical detection metrics.
    """
    assert len(eval_results) == len(ground_truth_labels), (
        f"Length mismatch: {len(eval_results)} results vs {len(ground_truth_labels)} labels"
    )

    y_true = np.array(ground_truth_labels, dtype=int)
    scores = np.array([res.anomaly_score for res in eval_results], dtype=float)
    y_pred_malicious = np.array([1 if res.is_malicious else 0 for res in eval_results], dtype=int)
    y_pred_flagged = np.array([1 if (res.is_malicious or res.is_suspicious) else 0 for res in eval_results], dtype=int)

    # 1. Binary Confusion Matrix for MALICIOUS decision
    # tn, fp, fn, tp
    cm_mal = confusion_matrix(y_true, y_pred_malicious, labels=[0, 1])
    tn_m, fp_m, fn_m, tp_m = cm_mal.ravel()

    tpr_mal = float(tp_m / max(tp_m + fn_m, 1))
    fpr_mal = float(fp_m / max(fp_m + tn_m, 1))
    prec_mal = float(tp_m / max(tp_m + fp_m, 1))
    f1_mal = float(2 * prec_mal * tpr_mal / max(prec_mal + tpr_mal, 1e-8))

    # 2. Binary Confusion Matrix for FLAGGED (SUSPICIOUS or MALICIOUS)
    cm_flg = confusion_matrix(y_true, y_pred_flagged, labels=[0, 1])
    tn_f, fp_f, fn_f, tp_f = cm_flg.ravel()

    tpr_flg = float(tp_f / max(tp_f + fn_f, 1))
    fpr_flg = float(fp_f / max(fp_f + tn_f, 1))
    prec_flg = float(tp_f / max(tp_f + fp_f, 1))
    f1_flg = float(2 * prec_flg * tpr_flg / max(prec_flg + tpr_flg, 1e-8))

    # 3. AUROC and AUPRC (continuous score discrimination)
    # Check if both classes are present
    if len(np.unique(y_true)) > 1:
        auroc = float(roc_auc_score(y_true, scores))
        precisions, recalls, _ = precision_recall_curve(y_true, scores)
        auprc = float(auc(recalls, precisions))
    else:
        auroc = 1.0 if y_true[0] == 0 else 0.0
        auprc = 1.0 if y_true[0] == 0 else 0.0

    # 4. Tri-State Breakdown
    tri_state_counts = {
        "NORMAL": {"honest": 0, "malicious": 0},
        "SUSPICIOUS": {"honest": 0, "malicious": 0},
        "MALICIOUS": {"honest": 0, "malicious": 0},
    }
    for res, label in zip(eval_results, ground_truth_labels):
        target_group = "malicious" if label == 1 else "honest"
        tri_state_counts[res.decision.value][target_group] += 1

    return {
        "total_observations": len(eval_results),
        "total_malicious": int(np.sum(y_true == 1)),
        "total_honest": int(np.sum(y_true == 0)),
        "malicious_decision_metrics": {
            "true_positives": int(tp_m),
            "false_positives": int(fp_m),
            "true_negatives": int(tn_m),
            "false_negatives": int(fn_m),
            "tpr_recall": round(tpr_mal, 4),
            "fpr": round(fpr_mal, 4),
            "precision": round(prec_mal, 4),
            "f1_score": round(f1_mal, 4),
        },
        "flagged_decision_metrics": {
            "description": "Treating SUSPICIOUS or MALICIOUS as positive",
            "true_positives": int(tp_f),
            "false_positives": int(fp_f),
            "true_negatives": int(tn_f),
            "false_negatives": int(fn_f),
            "tpr_recall": round(tpr_flg, 4),
            "fpr": round(fpr_flg, 4),
            "precision": round(prec_flg, 4),
            "f1_score": round(f1_flg, 4),
        },
        "continuous_score_metrics": {
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
        },
        "tri_state_distribution": tri_state_counts,
    }


def compute_stratified_fpr(
    eval_results: List[ClientEvaluationResult],
    ground_truth_labels: List[int],
    scenario_tags: List[str],  # e.g., 'iid', 'non_iid_alpha1.0', 'non_iid_alpha0.5', 'non_iid_alpha0.1'
) -> Dict[str, Dict[str, Any]]:
    """
    Compute False Positive Rates stratified by client data heterogeneity setting.
    Crucial for verifying that honest specialized non-IID hospitals are not falsely penalized.
    """
    strata: Dict[str, Dict[str, Any]] = {}
    unique_scenarios = sorted(list(set(scenario_tags)))

    for scenario in unique_scenarios:
        indices = [idx for idx, tag in enumerate(scenario_tags) if tag == scenario]
        sub_results = [eval_results[idx] for idx in indices]
        sub_labels = [ground_truth_labels[idx] for idx in indices]

        # Restrict strictly to honest observations within this scenario
        honest_indices = [i for i, y in enumerate(sub_labels) if y == 0]
        n_honest = len(honest_indices)

        if n_honest == 0:
            continue

        honest_results = [sub_results[i] for i in honest_indices]
        fp_malicious = sum(1 for r in honest_results if r.is_malicious)
        fp_suspicious = sum(1 for r in honest_results if r.is_suspicious)

        strata[scenario] = {
            "total_honest_observations": n_honest,
            "false_malicious_count": fp_malicious,
            "false_malicious_rate": round(fp_malicious / n_honest, 4),
            "suspicious_count": fp_suspicious,
            "suspicious_rate": round(fp_suspicious / n_honest, 4),
            "normal_count": sum(1 for r in honest_results if r.decision == DetectionDecision.NORMAL),
            "normal_rate": round(sum(1 for r in honest_results if r.decision == DetectionDecision.NORMAL) / n_honest, 4),
        }

    return strata
