"""
Phase 6: Honest Hospital False-Positive Audit Module.
Conducts a forensic analysis of all honest hospital observations classified as MALICIOUS
or SUSPICIOUS across IID and Non-IID scenarios (alpha in {1.0, 0.5, 0.1}).

Specifically investigates:
- Negative cosine similarity among honest Non-IID hospitals.
- Class distribution correlation with directional gradient deviation.
- Symmetrical clinical trade-off protection efficacy.
- Exact triggering rules for any residual false alarms.
"""

import sys
from pathlib import Path
import json
from collections import defaultdict
from typing import Dict, Any, List, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import DATA_RAW_DIR
from federated_learning.dataset import (
    prepare_federated_data_loaders,
    prepare_controlled_non_iid_data_loaders,
)
from federated_learning.detection.detector import (
    TriStateDetector,
    DetectionDecision,
    DetectorThresholds,
)

PHASE6_DIR = PROJECT_ROOT / "results" / "phase6_detector"
PHASE6_INSTRUMENTATION_DIR = PHASE6_DIR / "instrumentation"
PHASE6_EVALUATION_DIR = PHASE6_DIR / "evaluation"


def get_client_class_distributions(scenario: str, alpha: Optional[float], seed: int = 42) -> Dict[str, Dict[str, Any]]:
    """Retrieve actual client class distributions from dataset partitioner."""
    if scenario == "iid":
        _, _, _, partition_meta = prepare_federated_data_loaders(
            data_dir=DATA_RAW_DIR,
            num_clients=10,
            batch_size=32,
            seed=seed,
        )
    else:
        _, _, _, partition_meta = prepare_controlled_non_iid_data_loaders(
            data_dir=DATA_RAW_DIR,
            num_clients=10,
            alpha=alpha if alpha is not None else 0.5,
            batch_size=32,
            seed=seed,
        )
    
    dist_map = {}
    for stat in partition_meta["client_stats"]:
        cid = str(stat["client_id"])
        dist_map[cid] = {
            "normal_count": int(stat["normal_samples"]),
            "pneumonia_count": int(stat["pneumonia_samples"]),
            "total_count": int(stat["total_samples"]),
            "pneumonia_ratio": round(float(stat["pneumonia_ratio"]), 4),
        }
    return dist_map


def audit_honest_false_positives(detector: Optional[TriStateDetector] = None) -> Dict[str, Any]:
    """Execute forensic audit of honest runs."""
    if detector is None:
        detector = TriStateDetector()
        calib_runs = ["exp-med-006-ctrl-iid", "exp-med-007-ctrl-noniid-a05"]
        calib_records = []
        for cr in calib_runs:
            c_file = PHASE6_INSTRUMENTATION_DIR / f"{cr}_features.jsonl"
            if c_file.exists():
                with open(c_file, "r", encoding="utf-8") as f:
                    calib_records.extend([json.loads(l) for l in f if l.strip()])
        if calib_records:
            detector.calibrate_from_honest_observations(
                calib_records, target_malicious_fpr=0.01, target_suspicious_fpr=0.05
            )

    # Discover all honest simulation JSONL files
    all_jsonl_files = sorted(PHASE6_INSTRUMENTATION_DIR.glob("*_features.jsonl"))
    honest_files = []
    for p in all_jsonl_files:
        stem = p.stem.lower()
        if "atk" not in stem and "features" in stem:
            honest_files.append(p)

    print("\n" + "=" * 105)
    print("  PHASE 6 FORENSIC AUDIT: HONEST NON-IID FALSE POSITIVES & COSINE DISPERSION")
    print("=" * 105)
    print(f"  Auditing {len(honest_files)} honest simulation feature streams...")

    # Cache for client distributions by (scenario, alpha, seed)
    class_dist_cache = {}

    audit_records = []
    negative_cosine_events = []
    total_honest_observations = 0
    total_malicious_false_positives = 0
    total_suspicious_events = 0

    per_scenario_stats = defaultdict(lambda: {
        "total_obs": 0,
        "negative_cos_obs": 0,
        "false_malicious": 0,
        "suspicious": 0,
        "protected_events": 0,
    })

    for jsonl_path in honest_files:
        exp_id = jsonl_path.stem.replace("_features", "").upper()
        # Parse scenario and seed (check Non-IID before IID to avoid substring collision)
        if "NONIID" in exp_id or "NON-IID" in exp_id or "-A" in exp_id:
            scenario = "non_iid"
            if "A10" in exp_id or "ALPHA10" in exp_id:
                alpha = 1.0
            elif "A05" in exp_id or "ALPHA05" in exp_id:
                alpha = 0.5
            elif "A01" in exp_id or "ALPHA01" in exp_id:
                alpha = 0.1
            else:
                alpha = 0.5
        elif "IID" in exp_id:
            scenario = "iid"
            alpha = None
        else:
            scenario = "non_iid"
            alpha = 0.5

        # Extract seed if present
        seed = 42
        for part in exp_id.split("-"):
            if part.startswith("S") and part[1:].isdigit():
                seed = int(part[1:])

        cache_key = (scenario, alpha, seed)
        if cache_key not in class_dist_cache:
            class_dist_cache[cache_key] = get_client_class_distributions(scenario, alpha, seed)
        client_dists = class_dist_cache[cache_key]

        scen_tag = f"{scenario}_{alpha}" if alpha is not None else "iid"

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line.strip())
                cid = str(rec["client_id"])
                rnd = int(rec["round"])
                total_honest_observations += 1
                per_scenario_stats[scen_tag]["total_obs"] += 1

                # Evaluate with detector
                res = detector.evaluate_client(rec)
                cos_global = float(rec.get("cos_ref_global", 1.0))
                c_dist = client_dists.get(cid, {"normal_count": 0, "pneumonia_count": 0, "pneumonia_ratio": 0.5})

                if cos_global < 0.0:
                    per_scenario_stats[scen_tag]["negative_cos_obs"] += 1
                    negative_cosine_events.append({
                        "experiment_id": exp_id,
                        "client_id": cid,
                        "round": rnd,
                        "cos_global": cos_global,
                        "decision": res.decision.value,
                        "protected": res.symmetrical_tradeoff_protected,
                        "class_dist": c_dist,
                    })

                if res.symmetrical_tradeoff_protected:
                    per_scenario_stats[scen_tag]["protected_events"] += 1

                if res.decision == DetectionDecision.MALICIOUS:
                    total_malicious_false_positives += 1
                    per_scenario_stats[scen_tag]["false_malicious"] += 1
                    audit_records.append({
                        "experiment_id": exp_id,
                        "hospital_id": cid,
                        "round": rnd,
                        "scenario": scen_tag,
                        "actual_class_distribution": c_dist,
                        "z_mag_global": float(rec.get("z_mag_global", 0.0)),
                        "max_layer_z_mag": float(rec.get("max_layer_z_mag", 0.0)),
                        "layer_z_mag": rec.get("layer_z_mag", {}),
                        "cos_ref_global": cos_global,
                        "min_layer_cos": float(rec.get("min_layer_cos", 1.0)),
                        "layer_cos_ref": rec.get("layer_cos_ref", {}),
                        "delta_loss_total": float(rec.get("delta_loss_total", 0.0)),
                        "delta_loss_normal": float(rec.get("delta_loss_normal", 0.0)),
                        "delta_loss_pneumonia": float(rec.get("delta_loss_pneumonia", 0.0)),
                        "delta_rec_normal": float(rec.get("delta_rec_normal", 0.0)),
                        "delta_rec_pneumonia": float(rec.get("delta_rec_pneumonia", 0.0)),
                        "ema_anomaly_accumulator": float(rec.get("ema_anomaly_accumulator", 0.0)),
                        "exact_triggered_reasons": res.flagged_reasons,
                        "symmetrical_tradeoff_protected": res.symmetrical_tradeoff_protected,
                    })
                elif res.decision == DetectionDecision.SUSPICIOUS:
                    total_suspicious_events += 1
                    per_scenario_stats[scen_tag]["suspicious"] += 1

    # Print Forensic Summary Table
    print("\n" + "-" * 105)
    print("  HONEST OBSERVATION AUDIT SUMMARY BY SCENARIO")
    print("-" * 105)
    print(f"{'Scenario / Setting':<22} | {'Total Obs':<9} | {'Neg Cos (<0)':<13} | {'False Malicious':<16} | {'Suspicious (Alert)':<18} | {'Trade-off Prot':<14}")
    print("-" * 105)
    for scen, s in sorted(per_scenario_stats.items()):
        f_mal_pct = (s["false_malicious"] / s["total_obs"] * 100) if s["total_obs"] else 0.0
        sus_pct = (s["suspicious"] / s["total_obs"] * 100) if s["total_obs"] else 0.0
        neg_pct = (s["negative_cos_obs"] / s["total_obs"] * 100) if s["total_obs"] else 0.0
        prot_pct = (s["protected_events"] / s["total_obs"] * 100) if s["total_obs"] else 0.0
        print(
            f"{scen:<22} | {s['total_obs']:<9} | "
            f"{s['negative_cos_obs']:>4} ({neg_pct:5.2f}%)   | "
            f"{s['false_malicious']:>4} ({f_mal_pct:5.2f}%)   | "
            f"{s['suspicious']:>4} ({sus_pct:5.2f}%)     | "
            f"{s['protected_events']:>4} ({prot_pct:5.2f}%)"
        )
    print("-" * 105)
    print(f"  TOTAL HONEST OBSERVATIONS AUDITED: {total_honest_observations}")
    print(f"  TOTAL FALSE MALICIOUS:             {total_malicious_false_positives} ({(total_malicious_false_positives/total_honest_observations*100) if total_honest_observations else 0.0:.2f}%)")
    print(f"  TOTAL SUSPICIOUS (MONITORED):      {total_suspicious_events} ({(total_suspicious_events/total_honest_observations*100) if total_honest_observations else 0.0:.2f}%)")
    print("-" * 105)

    # Forensic Negative Cosine Analysis
    print("\n" + "-" * 105)
    print("  NEGATIVE COSINE SIMILARITY ANALYSIS AMONG HONEST HOSPITALS")
    print("-" * 105)
    if negative_cosine_events:
        print(f"  Total negative cosine occurrences in honest runs: {len(negative_cosine_events)}")
        cos_vals = [e["cos_global"] for e in negative_cosine_events]
        print(f"    • Cosine range: [{min(cos_vals):.4f}, {max(cos_vals):.4f}], Mean: {np.mean(cos_vals):.4f}")
        # Check decisions for negative cosine events
        neg_decisions = Counter([e["decision"] for e in negative_cosine_events])
        print(f"    • Decision breakdown: {dict(neg_decisions)}")
        protected_neg = sum(1 for e in negative_cosine_events if e["protected"])
        print(f"    • Protected by Symmetrical Trade-Off: {protected_neg} / {len(negative_cosine_events)} ({protected_neg/len(negative_cosine_events)*100:.1f}%)")
    else:
        print("  No negative cosine occurrences found.")
    print("-" * 105)

    # Detailed False-Positive Table (if any)
    if audit_records:
        print("\n" + "=" * 105)
        print("  DETAILED RECORD OF ALL FALSE MALICIOUS CLASSIFICATIONS")
        print("=" * 105)
        for r in audit_records[:15]:  # Print up to 15
            dist = r["actual_class_distribution"]
            dist_str = f"Norm:{dist['normal_count']}/Pneu:{dist['pneumonia_count']} ({dist['pneumonia_ratio']*100:.1f}% Pneu)"
            print(f"  • Exp: {r['experiment_id']} | Hosp: {r['hospital_id']} | Rd: {r['round']} | {dist_str}")
            print(f"    Z_mag: {r['z_mag_global']:.2f} (max_layer: {r['max_layer_z_mag']:.2f}) | Cos: {r['cos_ref_global']:.4f} | EMA: {r['ema_anomaly_accumulator']:.4f}")
            print(f"    ΔLoss: total={r['delta_loss_total']:.4f}, norm={r['delta_loss_normal']:.4f}, pneu={r['delta_loss_pneumonia']:.4f}")
            print(f"    Reason: {r['exact_triggered_reasons']}")
        if len(audit_records) > 15:
            print(f"  ... and {len(audit_records) - 15} additional false malicious records (see JSON report).")
        print("=" * 105)

    # Save to JSON
    audit_output_path = PHASE6_EVALUATION_DIR / "honest_false_positive_audit.json"
    with open(audit_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_honest_observations": total_honest_observations,
            "total_false_malicious": total_malicious_false_positives,
            "false_malicious_rate": round(total_malicious_false_positives / total_honest_observations, 4) if total_honest_observations else 0.0,
            "total_suspicious": total_suspicious_events,
            "suspicious_rate": round(total_suspicious_events / total_honest_observations, 4) if total_honest_observations else 0.0,
            "per_scenario_summary": dict(per_scenario_stats),
            "negative_cosine_analysis": {
                "total_events": len(negative_cosine_events),
                "decisions": dict(Counter([e["decision"] for e in negative_cosine_events])),
                "protected_count": sum(1 for e in negative_cosine_events if e["protected"]),
            },
            "false_malicious_audit_records": audit_records,
        }, f, indent=2)

    print(f"\n  Forensic audit report saved to {audit_output_path}")
    return {
        "total_honest_obs": total_honest_observations,
        "false_malicious": total_malicious_false_positives,
        "suspicious": total_suspicious_events,
        "records": audit_records,
    }


if __name__ == "__main__":
    from collections import Counter
    audit_honest_false_positives()
