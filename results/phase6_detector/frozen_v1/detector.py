"""
Phase 6: Multi-Factor Tri-State Malicious Hospital Detector.
Implements calibrated anomaly scoring, symmetrical clinical trade-off protection,
and tri-state decision classification (NORMAL, SUSPICIOUS, MALICIOUS).
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, asdict
import numpy as np


class DetectionDecision(str, Enum):
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    MALICIOUS = "MALICIOUS"


@dataclass
class ClientEvaluationResult:
    client_id: str
    round: int
    decision: DetectionDecision
    anomaly_score: float
    is_malicious: bool
    is_suspicious: bool
    flagged_reasons: List[str]
    symmetrical_tradeoff_protected: bool
    features: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["decision"] = self.decision.value
        return res


@dataclass
class DetectorThresholds:
    """
    Calibrated empirical quantile thresholds.
    Derived strictly from honest training runs.
    """
    # Magnitude thresholds (Modified Z-Score)
    z_mag_suspicious: float = 2.5
    z_mag_malicious: float = 4.0
    max_layer_z_suspicious: float = 3.0
    max_layer_z_malicious: float = 5.0

    # Directional thresholds (Cosine similarity)
    cos_suspicious: float = 0.30
    cos_strong_negative: float = -0.75     # Severe negative cosine indicates direct sign inversion
    cos_malicious_negative: float = 0.00   # Boundary below which negative cosine requires corroboration

    # Counterfactual Validation Loss Impact thresholds
    delta_loss_suspicious: float = 0.05
    delta_loss_malicious: float = 0.15

    # Historical Trajectory EMA Accumulator thresholds
    ema_accumulator_suspicious: float = 0.35
    ema_accumulator_malicious: float = 0.65


class TriStateDetector:
    """
    Calibrated tri-state detector for identifying malicious hospital nodes in Federated Learning.
    Operates strictly on server-accessible parameters, validation evaluations, and trajectories.
    """

    def __init__(self, thresholds: Optional[DetectorThresholds] = None) -> None:
        self.thresholds = thresholds or DetectorThresholds()

    def evaluate_client(self, feature_record: Dict[str, Any]) -> ClientEvaluationResult:
        """
        Evaluate a single client's feature record at communication round t.

        Args:
            feature_record: Dictionary containing extracted metrics from FeatureExtractor.

        Returns:
            ClientEvaluationResult detailing decision, score, and explanation.
        """
        cid = feature_record["client_id"]
        rnd = feature_record["round"]

        # 1. Feature extractions
        z_mag_global = feature_record.get("z_mag_global", 0.0)
        max_layer_z = feature_record.get("max_layer_z_mag", 0.0)
        cos_global = feature_record.get("cos_ref_global", 1.0)
        delta_loss_total = feature_record.get("delta_loss_total", 0.0)
        delta_loss_normal = feature_record.get("delta_loss_normal", 0.0)
        delta_loss_pneumonia = feature_record.get("delta_loss_pneumonia", 0.0)
        delta_rec_normal = feature_record.get("delta_rec_normal", 0.0)
        delta_rec_pneumonia = feature_record.get("delta_rec_pneumonia", 0.0)
        ema_acc = feature_record.get("ema_anomaly_accumulator", 0.0)

        flagged_reasons: List[str] = []
        is_malicious = False
        is_suspicious = False

        # 2. Symmetrical Clinical Trade-Off Protection Rule
        # In specialized hospitals (Non-IID), an update may improve one class while shifting the other.
        # If client's update exhibits a beneficial contribution to EITHER class:
        # (loss reduction in Normal OR loss reduction in Pneumonia OR recall gain in Normal OR recall gain in Pneumonia)
        # then validation loss alone cannot be used as conclusive evidence of malicious activity.
        has_clinical_benefit = (
            (delta_rec_normal > 0.0) or
            (delta_rec_pneumonia > 0.0) or
            (delta_loss_normal < 0.0) or
            (delta_loss_pneumonia < 0.0)
        )
        tradeoff_protected = bool(has_clinical_benefit)

        # 3. Severe Malicious Checks (Corroborated Inversion, Norm Explosion, or Toxic Harm)
        # Condition A1: Catastrophic Sign Inversion (Model Inversion attack, gamma = -1.0)
        # Severe negative cosine (cos < -0.75) that is uncompensated by clinical trade-off
        if (cos_global < self.thresholds.cos_strong_negative) and not tradeoff_protected:
            is_malicious = True
            flagged_reasons.append(
                f"Severe update inversion: cos_global={cos_global:.4f} < {self.thresholds.cos_strong_negative:.2f}"
            )

        # Condition A2: Corroborated Negative Cosine Inversion
        # Negative cosine (cos < 0.0) that is corroborated by uncompensated validation degradation,
        # elevated magnitude, or elevated anomaly accumulator:
        elif (cos_global < self.thresholds.cos_malicious_negative) and not tradeoff_protected:
            is_toxic = (delta_loss_total > self.thresholds.delta_loss_suspicious)
            is_elevated_norm = (z_mag_global > self.thresholds.z_mag_suspicious)
            is_persistent = (ema_acc > self.thresholds.ema_accumulator_suspicious)
            if is_toxic or is_elevated_norm or is_persistent:
                is_malicious = True
                flagged_reasons.append(
                    f"Corroborated update inversion: cos_global={cos_global:.4f} < 0.0 "
                    f"with uncompensated anomaly (loss_deg={is_toxic}, norm_elev={is_elevated_norm}, persistent={is_persistent})"
                )

        # Condition B: Severe Magnitude Explosion (Scaling attacks)
        # Global norm explosion requires both statistically elevated Z-score and an actual ratio explosion (> 5.0x median)
        norm_ratio_global = feature_record.get("norm_ratio_global", 1.0)
        has_ratio = "norm_ratio_global" in feature_record
        is_exploded_norm = (z_mag_global > self.thresholds.z_mag_malicious) and (
            (norm_ratio_global > 5.0) if has_ratio else True
        )
        if is_exploded_norm:
            is_malicious = True
            flagged_reasons.append(
                f"Global norm explosion: Z_mag={z_mag_global:.2f} > {self.thresholds.z_mag_malicious:.2f}" +
                (f" (ratio={norm_ratio_global:.2f} > 5.0)" if has_ratio else "")
            )
        elif (max_layer_z > self.thresholds.max_layer_z_malicious) and (
            (delta_loss_total > self.thresholds.delta_loss_malicious and not tradeoff_protected and ema_acc > self.thresholds.ema_accumulator_suspicious)
        ):
            is_malicious = True
            flagged_reasons.append(
                f"Corroborated layer-wise norm explosion: max_layer_Z={max_layer_z:.2f} > {self.thresholds.max_layer_z_malicious:.2f} with uncompensated harm"
            )

        # Condition C: Corroborated Persistent Harm (Validation Degradation + High EMA Accumulator)
        # Requires that validation degradation is UNCOMPENSATED (no clinical trade-off)
        if not is_malicious:
            is_toxic_loss = (delta_loss_total > self.thresholds.delta_loss_malicious) and not tradeoff_protected
            is_persistent = (ema_acc > self.thresholds.ema_accumulator_malicious)
            if is_toxic_loss and is_persistent:
                is_malicious = True
                flagged_reasons.append(
                    f"Corroborated persistent harm: uncompensated delta_loss={delta_loss_total:.4f} > {self.thresholds.delta_loss_malicious:.4f} "
                    f"with high EMA accumulator={ema_acc:.4f}"
                )

        # 4. Suspicious Checks (Borderline Anomaly or Low Cosine without Harm)
        # Note: Low positive cosine similarity alone NEVER triggers MALICIOUS, only SUSPICIOUS.
        # Note: Uncorroborated negative cosine or isolated layer-wise norm anomaly produces SUSPICIOUS.
        if not is_malicious:
            # Low positive or uncorroborated negative cosine similarity (frequent in Non-IID, flagged for tracking)
            if cos_global < self.thresholds.cos_suspicious:
                is_suspicious = True
                flagged_reasons.append(
                    f"Low/negative reference cosine: cos_global={cos_global:.4f} < {self.thresholds.cos_suspicious:.4f} "
                    f"(monitored for Non-IID variance)"
                )

            # Borderline magnitude deviation
            if z_mag_global > self.thresholds.z_mag_suspicious:
                is_suspicious = True
                flagged_reasons.append(
                    f"Borderline global norm: Z_mag={z_mag_global:.2f} > {self.thresholds.z_mag_suspicious:.2f}"
                )
            elif max_layer_z > self.thresholds.max_layer_z_suspicious:
                is_suspicious = True
                flagged_reasons.append(
                    f"Borderline layer norm: max_layer_Z={max_layer_z:.2f} > {self.thresholds.max_layer_z_suspicious:.2f}"
                )

            # Borderline validation impact (uncompensated)
            if (delta_loss_total > self.thresholds.delta_loss_suspicious) and not tradeoff_protected:
                is_suspicious = True
                flagged_reasons.append(
                    f"Uncompensated validation degradation: delta_loss={delta_loss_total:.4f} > {self.thresholds.delta_loss_suspicious:.4f}"
                )

            # Borderline trajectory accumulator
            if ema_acc > self.thresholds.ema_accumulator_suspicious:
                is_suspicious = True
                flagged_reasons.append(
                    f"Elevated trajectory accumulator: EMA={ema_acc:.4f} > {self.thresholds.ema_accumulator_suspicious:.4f}"
                )

        # 5. Composite Continuous Anomaly Score (in [0.0, 1.0] for AUROC / AUPRC calculation)
        # Weighting: Direction (0.35) + Magnitude (0.35) + Impact (0.15) + Trajectory (0.15)
        score_dir = 1.0 - max(0.0, min(1.0, (cos_global + 1.0) / 2.0))
        score_mag = min(1.0, max(0.0, z_mag_global / 5.0))
        score_imp = 0.0
        if delta_loss_total > 0 and not tradeoff_protected:
            score_imp = min(1.0, delta_loss_total / 0.20)
        score_hist = min(1.0, ema_acc)

        composite_anomaly_score = float(
            0.35 * score_dir + 0.35 * score_mag + 0.15 * score_imp + 0.15 * score_hist
        )

        # 6. Final State Decision
        if is_malicious:
            decision = DetectionDecision.MALICIOUS
        elif is_suspicious:
            decision = DetectionDecision.SUSPICIOUS
        else:
            decision = DetectionDecision.NORMAL

        return ClientEvaluationResult(
            client_id=str(cid),
            round=rnd,
            decision=decision,
            anomaly_score=round(composite_anomaly_score, 4),
            is_malicious=is_malicious,
            is_suspicious=is_suspicious,
            flagged_reasons=flagged_reasons,
            symmetrical_tradeoff_protected=tradeoff_protected,
            features=feature_record,
        )

    def calibrate_from_honest_observations(
        self,
        honest_feature_records: List[Dict[str, Any]],
        target_malicious_fpr: float = 0.01,
        target_suspicious_fpr: float = 0.05,
    ) -> DetectorThresholds:
        """
        Derive empirical quantile thresholds from honest calibration records.
        Applies Bonferroni multiple testing correction for layer-wise checks.

        Args:
            honest_feature_records: List of feature dictionaries from honest calibration runs.
            target_malicious_fpr: Desired FPR for MALICIOUS classification (e.g. 0.01 -> 99th percentile).
            target_suspicious_fpr: Desired FPR for SUSPICIOUS classification (e.g. 0.05 -> 95th percentile).

        Returns:
            Updated DetectorThresholds object.
        """
        if not honest_feature_records:
            return self.thresholds

        z_mags = np.array([r.get("z_mag_global", 0.0) for r in honest_feature_records])
        layer_zs = np.array([r.get("max_layer_z_mag", 0.0) for r in honest_feature_records])
        cosines = np.array([r.get("cos_ref_global", 1.0) for r in honest_feature_records])
        delta_losses = np.array([
            r.get("delta_loss_total", 0.0) for r in honest_feature_records
            # Filter to uncompensated records to see natural uncompensated loss fluctuation
            if not ((r.get("delta_rec_normal", 0.0) > 0.0) or (r.get("delta_rec_pneumonia", 0.0) > 0.0))
        ])
        if len(delta_losses) == 0:
            delta_losses = np.array([r.get("delta_loss_total", 0.0) for r in honest_feature_records])
        ema_accs = np.array([r.get("ema_anomaly_accumulator", 0.0) for r in honest_feature_records])

        # Quantile percentiles
        p_mal_upper = (1.0 - target_malicious_fpr) * 100.0  # 99.0
        p_sus_upper = (1.0 - target_suspicious_fpr) * 100.0  # 95.0
        # Bonferroni multiple testing correction for 4 layer-wise checks
        p_layer_mal = (1.0 - (target_malicious_fpr / 4.0)) * 100.0  # 99.75
        p_layer_sus = (1.0 - (target_suspicious_fpr / 4.0)) * 100.0  # 98.75

        p_sus_lower = target_suspicious_fpr * 100.0          # 5.0

        # Magnitude upper quantiles
        z_mag_susp = float(np.percentile(z_mags, p_sus_upper))
        z_mag_mal = float(np.percentile(z_mags, p_mal_upper))
        max_layer_susp = float(np.percentile(layer_zs, p_layer_sus))
        max_layer_mal = float(np.percentile(layer_zs, p_layer_mal))

        # Direction lower quantiles
        cos_susp = float(np.percentile(cosines, p_sus_lower))

        # Validation loss upper quantiles
        loss_susp = float(np.percentile(delta_losses, p_sus_upper))
        loss_mal = float(np.percentile(delta_losses, p_mal_upper))

        # Trajectory accumulator upper quantiles
        ema_susp = float(np.percentile(ema_accs, p_sus_upper))
        ema_mal = float(np.percentile(ema_accs, p_mal_upper))

        self.thresholds = DetectorThresholds(
            z_mag_suspicious=round(z_mag_susp, 2),
            z_mag_malicious=round(z_mag_mal, 2),
            max_layer_z_suspicious=round(max_layer_susp, 2),
            max_layer_z_malicious=round(max_layer_mal, 2),
            cos_suspicious=round(cos_susp, 4),
            cos_strong_negative=-0.75,
            cos_malicious_negative=0.00,
            delta_loss_suspicious=round(loss_susp, 4),
            delta_loss_malicious=round(loss_mal, 4),
            ema_accumulator_suspicious=round(ema_susp, 4),
            ema_accumulator_malicious=round(ema_mal, 4),
        )

        return self.thresholds
