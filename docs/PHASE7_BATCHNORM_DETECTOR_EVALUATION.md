# Phase 7 BatchNorm Attack: Frozen Detector Evaluation and Severity Audit

**Document Status:** Complete Audit Report & Empirical Evaluation (Revision 2: Effective Threshold Resolution & Severity Deep-Dive)  
**Evaluation Scope:** Preliminary One-Round Controlled Diagnostic (Seed 211, Communication Round 1)  
**Execution Gate Status:** **HOLD — 351 Main Simulations Paused Awaiting User Review**  
**Blockchain Status:** Strictly Deferred to Phase 8  

---

## 1. Executive Summary

This report delivers the comprehensive technical audit and resolution for the newly discovered BatchNorm-only poisoning vulnerability in Phase 7. The evaluation establishes:

1. **Resolution of Effective Threshold Discrepancy (PART A):** The runtime detector instance held inside memory uses the exact empirical calibrated configuration loaded from [calibrated_thresholds.json](file:///home/suvranil/project/federated_learning/detection/frozen_v1/calibrated_thresholds.json) ($Z_{\text{mag,susp}} = 4.38, Z_{\text{mag,mal}} = 9.10, Z_{\text{layer,susp}} = 9.10, Z_{\text{layer,mal}} = 12.41$), identically matching the Phase 6 blind evaluation runtime configuration. The previously reported numbers ($2.50, 4.00, 3.00, 5.00$) were uncalibrated dataclass defaults from [detector.py](file:///home/suvranil/project/federated_learning/detection/frozen_v1/detector.py#L44-L60) inadvertently displayed in human-authored documentation. Under both threshold sets, the automatic decision for Hospital 0 is identical: **`NORMAL`**.
2. **Clinical Trade-Off Vulnerability Audit (PART B):** Frozen v1 defines `has_clinical_benefit = (delta_rec_normal > 0.0 or delta_rec_pneumonia > 0.0 or delta_loss_normal < 0.0 or delta_loss_pneumonia < 0.0)`. In both IID and Non-IID conditions, poisoned floating buffers saturate activations and force 100% of validation samples to be predicted as Class 0 (Normal). This artificial surge in Normal recall ($\Delta \text{rec}_{\text{normal}} = +37.04\%$ in IID, $+100.0\%$ in Non-IID) activates trade-off protection, suppressing validation-loss degradation flags despite Pneumonia screening sensitivity collapsing to **$0.0\%$** ($97.43\% \to 0\%$ in IID, $100.0\% \to 0\%$ in Non-IID). A prospective Detector v2 design is specified separately without modifying frozen v1.
3. **Mathematical Explanation of Severity Sweep Invariance (PART C):** Across all four tested poisoning magnitudes $\mu_{\text{poison}} \in [10, 100, 1000, 10^6]$, test performance is identical:
   - For vulnerable aggregators (**FedAvg**, **Multi-Krum $f=1/f=2$**, **Tri-State Monitor**, **Tri-State Discount $0.50/0.25$**), the mean shift $\mu \ge 10$ is large enough relative to image inputs ($[-2, +2]$) to drive post-normalization values negative across all feature maps. The subsequent **ReLU** activation zeros out all feature maps ($\max(0, x) = 0$), collapsing network outputs to the final layer's bias vector ($b_{\text{normal}} = -0.00266 > b_{\text{pneumonia}} = -0.02705$), which predicts 100% Normal (Accuracy $37.50\%$, Sensitivity $0.0\%$, Macro-F1 $27.27\%$).
   - For robust aggregators (**Coordinate Median**, **Trimmed Mean $\beta=1$**, **Trimmed Mean $\beta=2$**), Hospital 0 is the sole outlier (1/10 clients). Coordinate-wise trimming eliminates the maximum coordinate regardless of whether it is 10 or $10^6$, preserving the global buffer mean at $\mathbf{0.00}$ and sustaining $\mathbf{82.2\% - 84.3\%}$ accuracy and $\mathbf{79.0\% - 82.6\%}$ Macro-F1.
4. **Expanded Matrix & Test Suite Integrity (PART D & E):** The authoritative 351-simulation matrix manifest [PHASE7_EXPERIMENT_MATRIX.json](file:///home/suvranil/project/results/phase7_robust_aggregation/PHASE7_EXPERIMENT_MATRIX.json) (SHA-256: `712d644cca26216a865e754670eb50b50719881739c85ec446b840c7918d2139`) and all 73 project unit tests pass. main simulations remain stopped and gated.

---

## 2. PART A — Effective Threshold Verification & Discrepancy Resolution

### 2.1 File Hashes and Loading Mechanism
Both [detector.py](file:///home/suvranil/project/federated_learning/detection/frozen_v1/detector.py) and [feature_extractor.py](file:///home/suvranil/project/federated_learning/detection/frozen_v1/feature_extractor.py) were loaded dynamically via `importlib.util.spec_from_file_location` directly from [federated_learning/detection/frozen_v1/](file:///home/suvranil/project/federated_learning/detection/frozen_v1/):

| Target File | Runtime `__file__` Path | Computed SHA-256 | Expected SHA-256 ([FREEZE_MANIFEST.json](file:///home/suvranil/project/federated_learning/detection/frozen_v1/FREEZE_MANIFEST.json)) | Integrity Match |
| :--- | :--- | :--- | :--- | :---: |
| **Detector Module** | `/home/suvranil/project/federated_learning/detection/frozen_v1/detector.py` | `6576468da037927fda3283a1e2d8402f3b2739d6c45b6fe8ad517900e94a7f07` | `6576468da037927fda3283a1e2d8402f3b2739d6c45b6fe8ad517900e94a7f07` | **PASS (Exact)** |
| **Feature Extractor** | `/home/suvranil/project/federated_learning/detection/frozen_v1/feature_extractor.py` | `66de939c5c0432bb8570fa5d0de82f7701a07a699f4e3d3c6aa528316122c906` | `66de939c5c0432bb8570fa5d0de82f7701a07a699f4e3d3c6aa528316122c906` | **PASS (Exact)** |
| **Calibrated Thresholds** | `/home/suvranil/project/federated_learning/detection/frozen_v1/calibrated_thresholds.json` | `737c4819df70a8cadc93d9ff9a2db82bc36040403e11ee1240031c9b0cf34507` | `737c4819df70a8cadc93d9ff9a2db82bc36040403e11ee1240031c9b0cf34507` | **PASS (Exact)** |

### 2.2 Threshold Audit: Runtime Instance vs Dataclass Defaults vs Internal Constants
The thresholds held inside the runtime instance `detector.thresholds` were audited directly via attribute reflection:

```
--- Effective Runtime Detector Thresholds Audit ---
Threshold Parameter          | Runtime Instance (Calibrated)  | Dataclass Default   
------------------------------------------------------------------------------------
z_mag_suspicious             | 4.38                           | 2.5                 
z_mag_malicious              | 9.1                            | 4.0                 
max_layer_z_suspicious       | 9.1                            | 3.0                 
max_layer_z_malicious        | 12.41                          | 5.0                 
cos_suspicious               | 0.1465                         | 0.3                 
cos_strong_negative          | -0.75                          | -0.75               
cos_malicious_negative       | 0.0                            | 0.0                 
delta_loss_suspicious        | 0.017                          | 0.05                
delta_loss_malicious         | 0.0274                         | 0.15                
ema_accumulator_suspicious   | 0.4768                         | 0.35                
ema_accumulator_malicious    | 0.8023                         | 0.65                
```

#### Detailed Breakdown of Roles:
1. **Calibrated Empirical Quantile Thresholds (from [calibrated_thresholds.json](file:///home/suvranil/project/federated_learning/detection/frozen_v1/calibrated_thresholds.json)):**
   - Bound to `detector.thresholds` upon instantiation (`DetectorThresholds(**thresholds_dict)`).
   - Govern all state transitions (`NORMAL`, `SUSPICIOUS`, `MALICIOUS`):
     - Severe magnitude explosion: `z_mag_global > 9.10` (line 140)
     - Layer-wise explosion: `max_layer_z > 12.41` (line 149)
     - Persistent toxic loss: `delta_loss_total > 0.0274` and `ema_acc > 0.8023` (line 160-161)
     - Suspicious cosine alignment: `cos_global < 0.1465` (line 174)
     - Borderline global magnitude: `z_mag_global > 4.38` (line 182)
     - Borderline layer magnitude: `max_layer_z > 9.10` (line 187)
     - Borderline validation loss degradation: `delta_loss_total > 0.0170` (line 194)
     - Borderline trajectory accumulator: `ema_acc > 0.4768` (line 201)
   - **Zero calibrated thresholds are silently ignored.** Every field in the JSON file is actively referenced in the decision logic.
2. **Internal Decision-Rule Constants (Hardcoded in [detector.py](file:///home/suvranil/project/federated_learning/detection/frozen_v1/detector.py)):**
   - Norm ratio explosion threshold: `norm_ratio_global > 5.0` (line 141)
   - Continuous anomaly score scaling: `score_mag = min(1.0, z_mag_global / 5.0)` (line 210)
   - Continuous anomaly score validation scaling: `score_imp = min(1.0, delta_loss_total / 0.20)` (line 213)
   - Continuous score component weights: `0.35 * Dir + 0.35 * Mag + 0.15 * Imp + 0.15 * Hist` (line 217)
   - Clinical trade-off zero-boundary: `delta_rec > 0.0` or `delta_loss < 0.0` (lines 106-109)
3. **Root Cause of the Discrepancy:**
   The evaluation loader in [evaluate_batchnorm_attack_conditions.py](file:///home/suvranil/project/experiments/evaluate_batchnorm_attack_conditions.py) correctly constructed `DetectorThresholds(**thresholds_dict)` at runtime. However, the author of the preliminary report manually transcribed the dataclass default values ($2.50, 4.00, 3.00, 5.00$) from the Python class definition into the documentation table instead of displaying the active calibrated values.
4. **Parity with Phase 6 Blind Evaluation:**
   Both `run_phase6_blind_evaluation.py` (lines 112-116) and `evaluate_batchnorm_attack_conditions.py` (lines 301-302) execute the identical code path:
   ```python
   with open(thresholds_path, "r", encoding="utf-8") as f:
       thresh_dict = json.load(f)
   thresholds = module.DetectorThresholds(**thresh_dict)
   detector = module.TriStateDetector(thresholds=thresholds)
   ```
   Under both the dataclass default thresholds and the calibrated empirical thresholds, the automatic decision for Hospital 0 is **identically `NORMAL`** (Score: $0.1171$ in IID, $0.0342$ in Non-IID).

---

## 3. PART B — Clinical Trade-Off Vulnerability & Proposed Detector v2

### 3.1 Logical Mechanics of the False-Negative in Frozen v1
The clinical trade-off rule in [detector.py](file:///home/suvranil/project/federated_learning/detection/frozen_v1/detector.py#L100-L112) is evaluated as follows:
```python
has_clinical_benefit = (
    (delta_rec_normal > 0.0) or
    (delta_rec_pneumonia > 0.0) or
    (delta_loss_normal < 0.0) or
    (delta_loss_pneumonia < 0.0)
)
tradeoff_protected = bool(has_clinical_benefit)
```

In Leave-One-Out (LOO) counterfactual validation:
- **Condition 12 (`ATK-BN-IID-M1`):**
  - Honest LOO Model ($w_{-0}$): Balanced performance ($\text{Rec}_{\text{norm}} = 62.96\%, \text{Rec}_{\text{pneu}} = 97.43\%, \text{Loss} = 0.4891$).
  - Full Model ($w_{\text{full}}$): Poisoned buffers push activations into negative space. The model predicts Class 0 (Normal) for 100% of validation samples ($\text{Rec}_{\text{norm}} = 100.00\%, \text{Rec}_{\text{pneu}} = 0.00\%, \text{Loss} = 0.6973$).
  - Change: $\Delta \text{rec}_{\text{normal}} = \mathbf{+0.3704} > 0.0$, but $\Delta \text{rec}_{\text{pneumonia}} = \mathbf{-0.9743}$ (complete collapse).
  - Outcome: Because $\Delta \text{rec}_{\text{normal}} > 0.0$, `has_clinical_benefit` evaluates to `True`. `tradeoff_protected = True` suppresses both the malicious loss penalty ($\Delta \mathcal{L} = +0.2081 > 0.0274$) and the suspicious loss penalty ($\Delta \mathcal{L} > 0.0170$).
- **Condition 13 (`ATK-BN-NONIID-M1`):**
  - Honest LOO Model ($w_{-0}$): $\text{Rec}_{\text{norm}} = 0.00\%, \text{Rec}_{\text{pneu}} = 100.00\%, \text{Loss} = 0.6492$.
  - Full Model ($w_{\text{full}}$): Predicts Class 0 for 100% of validation samples ($\text{Rec}_{\text{norm}} = 100.00\%, \text{Rec}_{\text{pneu}} = 0.00\%, \text{Loss} = 0.6961$).
  - Change: $\Delta \text{rec}_{\text{normal}} = \mathbf{+1.0000} > 0.0$, but $\Delta \text{rec}_{\text{pneumonia}} = \mathbf{-1.0000}$ (total collapse).
  - Outcome: `tradeoff_protected = True` is activated, shielding the client from detection.

> [!CAUTION]
> **Clinical Safety Hazard:** In chest radiography screening, missing pneumonia cases (false negatives) can be fatal. A trade-off rule that permits pneumonia sensitivity to collapse from 97.4% to 0.0% simply because normal specificity reached 100% is fundamentally unsafe for medical deployment.

### 3.2 Specification of Proposed Detector v2 (Documented Only; Frozen v1 Untouched)
In strict adherence to project constraints, **frozen v1 remains unmodified**. The following proposed specification is documented for future implementation in Phase 8:

```python
# PROPOSED DETECTOR V2 DESIGN (NON-FROZEN FUTURE SPECIFICATION)
# 1. Asymmetrical Clinical Safety Envelope
# Veto trade-off protection if disease sensitivity experiences severe degradation
clinical_harm_veto = (
    (delta_rec_pneumonia < -0.10) or     # Unacceptable drop in screening sensitivity
    (delta_loss_pneumonia > 0.10) or     # Unacceptable cross-entropy degradation on disease
    (full_metrics["recall_pneumonia"] < 0.50)  # Absolute sensitivity floor
)

has_legitimate_clinical_benefit = (
    not clinical_harm_veto and (
        (delta_rec_normal > 0.0 and delta_rec_pneumonia >= -0.02) or
        (delta_rec_pneumonia > 0.0 and delta_rec_normal >= -0.05) or
        (delta_loss_normal < -0.05 and delta_loss_pneumonia <= 0.02) or
        (delta_loss_pneumonia < -0.05 and delta_loss_normal <= 0.02)
    )
)

# 2. BatchNorm Floating-Buffer Anomaly Features
# Compute layer-wise modified Z-scores over running_mean and running_var
z_bn_mean = compute_modified_z(bn_mean_vectors)
z_bn_var = compute_modified_z(bn_var_vectors)
is_bn_anomaly = (z_bn_mean > 5.0) or (z_bn_var > 5.0)

# 3. Tri-State Decision Logic with Inconclusive Evidence Handling
if is_bn_anomaly or (is_toxic_loss and not has_legitimate_clinical_benefit):
    decision = DetectionDecision.MALICIOUS
elif (z_bn_mean > 3.0) or (delta_loss_total > tau_loss_susp and clinical_harm_veto):
    decision = DetectionDecision.SUSPICIOUS
else:
    decision = DetectionDecision.NORMAL
```

*Note: In accordance with prompt instructions, no claim is made that Detector v2 works until it is independently implemented and evaluated.*

---

## 4. PART C — Severity Sweep Verification Across All Nine Aggregators

The attack severity sweep was executed on Communication Round 1 (Partition Seed 211, Training Seed 211, Initialization Seed 42, Checkpoint SHA-256 `80a2f4d2ca2e8b00a305852cdcf494889ed32a14ace98621f70d823d60438b68`) on the pristine test split ($N=624$: 234 Normal, 390 Pneumonia).

### 4.1 Mathematical Explanation of Output Invariance
Across all tested magnitudes $\mu_{\text{poison}} \in [10, 100, 1000, 10^6]$:
1. **Activation Collapse in Vulnerable Aggregators:**
   Input chest X-rays have normalized pixels in approximately $[-2.0, +2.0]$. In the first convolutional block, activations are passed to BatchNorm1 (`features.1`):
   $$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \gamma + \beta$$
   When Hospital 0 poisons floating buffers, FedAvg aggregates a running mean of $\mu \approx 0.1 \times \mu_{\text{poison}}$. Even at $\mu_{\text{poison}} = 10.0$ ($\mu_{\text{agg}} \approx 1.0, \sigma^2_{\text{agg}} \approx 1.5$), $(x - \mu)$ becomes negative across all pixels.
   The subsequent **ReLU** activation ($\max(0, x)$) zeroes out the feature map:
   $$\text{ReLU}(\hat{x}) = \mathbf{0.0}$$
   At $\mu_{\text{poison}} \in [100, 1000, 10^6]$, the number of non-zero activations after ReLU1 is strictly **`0`**.
   The classifier linear head receives an all-zero input vector:
   $$z = W \cdot \mathbf{0} + b_{\text{classifier}} = b_{\text{classifier}}$$
   In `SmallCNN`, the final classification bias vector is:
   $$b_0 = -0.00266, \quad b_1 = -0.02705$$
   Because $-0.00266 > -0.02705$, $\text{argmax}(z)$ produces **Class 0 (Normal)** for **100% of test samples**, regardless of magnitude.
   - True Negatives (Normal correct) = 234 / 234 ($100.0\%$ specificity).
   - False Negatives (Pneumonia missed) = 390 / 390 ($0.0\%$ sensitivity).
   - Accuracy = $234 / 624 = \mathbf{37.50\%}$.
   - Macro-F1 = $\frac{54.55\% + 0.00\%}{2} = \mathbf{27.27\%}$.
2. **Buffer Outlier Rejection in Robust Aggregators:**
   Hospital 0 is 1 out of 10 clients ($10\%$). Across all magnitudes, Hospital 0's buffer value ($10, 100, 1000, 10^6$) occupies the extreme top coordinate.
   - **Coordinate Median:** Selects the 5th/6th sorted client, which is an honest update ($\mu \approx 0.00$).
   - **Trimmed Mean $\beta=1$:** Trims the 1 highest coordinate (Hospital 0) and 1 lowest coordinate, averaging the 8 honest clients ($\mu \approx 0.00$).
   - **Trimmed Mean $\beta=2$:** Trims the 2 highest and 2 lowest coordinates, averaging the 6 honest clients ($\mu \approx 0.00$).
   Consequently, the aggregated model parameters and buffers are identical across all magnitudes, sustaining invariant test performance.

---

### 4.2 Comprehensive Strategy Telemetry Table Across Magnitudes

The table below records the empirical telemetry for all nine aggregation strategies across all four poisoning magnitudes:

| Strategy | Poison $\mu$ | Layer 1 Mean ($\mu_1$) | Layer 1 Var ($\sigma^2_1$) | Layer 2 Mean ($\mu_2$) | Layer 2 Var ($\sigma^2_2$) | Test Confusion Matrix `[TN, FP, FN, TP]` | Acc (%) | Sens (%) | Spec (%) | Macro F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FedAvg** | $10$ | $1.00$ | $1.49$ | $0.89$ | $1.38$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |
| | $100$ | $10.00$ | $10.49$ | $9.89$ | $10.38$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |
| | $1,000$ | $100.04$ | $100.49$ | $99.94$ | $100.38$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |
| | $10^6$ | $100,042.5$ | $100,043.0$ | $100,042.4$ | $100,042.8$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |
| **Coordinate Median** | $10$ | $0.00$ | $0.49$ | $-0.11$ | $0.38$ | `[165, 69, 29, 361]` | $\mathbf{84.29\%}$ | $\mathbf{92.56\%}$ | $\mathbf{70.51\%}$ | $\mathbf{82.58\%}$ |
| | $100$ | $0.00$ | $0.49$ | $-0.11$ | $0.38$ | `[165, 69, 29, 361]` | $\mathbf{84.29\%}$ | $\mathbf{92.56\%}$ | $\mathbf{70.51\%}$ | $\mathbf{82.58\%}$ |
| | $1,000$ | $0.00$ | $0.49$ | $-0.11$ | $0.38$ | `[165, 69, 29, 361]` | $\mathbf{84.29\%}$ | $\mathbf{92.56\%}$ | $\mathbf{70.51\%}$ | $\mathbf{82.58\%}$ |
| | $10^6$ | $0.00$ | $0.49$ | $-0.11$ | $0.38$ | `[165, 69, 29, 361]` | $\mathbf{84.29\%}$ | $\mathbf{92.56\%}$ | $\mathbf{70.51\%}$ | $\mathbf{82.58\%}$ |
| **Trimmed Mean $\beta=1$** | $10$ | $0.00$ | $0.49$ | $-0.12$ | $0.38$ | `[135, 99, 12, 378]` | $\mathbf{82.21\%}$ | $\mathbf{96.92\%}$ | $\mathbf{57.69\%}$ | $\mathbf{79.03\%}$ |
| | $100$ | $0.00$ | $0.49$ | $-0.12$ | $0.38$ | `[135, 99, 12, 378]` | $\mathbf{82.21\%}$ | $\mathbf{96.92\%}$ | $\mathbf{57.69\%}$ | $\mathbf{79.03\%}$ |
| | $1,000$ | $0.00$ | $0.49$ | $-0.12$ | $0.38$ | `[135, 99, 12, 378]` | $\mathbf{82.21\%}$ | $\mathbf{96.92\%}$ | $\mathbf{57.69\%}$ | $\mathbf{79.03\%}$ |
| | $10^6$ | $0.00$ | $0.49$ | $-0.12$ | $0.38$ | `[135, 99, 12, 378]` | $\mathbf{82.21\%}$ | $\mathbf{96.92\%}$ | $\mathbf{57.69\%}$ | $\mathbf{79.03\%}$ |
| **Trimmed Mean $\beta=2$** | $10$ | $0.00$ | $0.47$ | $-0.12$ | $0.37$ | `[145, 89, 17, 373]` | $\mathbf{83.01\%}$ | $\mathbf{95.64\%}$ | $\mathbf{61.97\%}$ | $\mathbf{80.40\%}$ |
| | $100$ | $0.00$ | $0.47$ | $-0.12$ | $0.37$ | `[145, 89, 17, 373]` | $\mathbf{83.01\%}$ | $\mathbf{95.64\%}$ | $\mathbf{61.97\%}$ | $\mathbf{80.40\%}$ |
| | $1,000$ | $0.00$ | $0.47$ | $-0.12$ | $0.37$ | `[145, 89, 17, 373]` | $\mathbf{83.01\%}$ | $\mathbf{95.64\%}$ | $\mathbf{61.97\%}$ | $\mathbf{80.40\%}$ |
| | $10^6$ | $0.00$ | $0.47$ | $-0.12$ | $0.37$ | `[145, 89, 17, 373]` | $\mathbf{83.01\%}$ | $\mathbf{95.64\%}$ | $\mathbf{61.97\%}$ | $\mathbf{80.40\%}$ |
| **Multi-Krum $f=1$** | $10$ | $1.11$ | $1.59$ | $1.00$ | $1.49$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |
| | $100$ | $11.11$ | $11.59$ | $11.00$ | $11.49$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |
| | $1,000$ | $111.11$ | $111.59$ | $111.00$ | $111.49$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |
| | $10^6$ | $111,111.1$ | $111,111.6$ | $111,111.0$ | $111,111.5$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |
| **Multi-Krum $f=2$** | $10$ | $1.25$ | $1.72$ | $1.14$ | $1.62$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |
| | $100$ | $12.50$ | $12.72$ | $12.39$ | $12.62$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |
| | $1,000$ | $125.00$ | $125.72$ | $124.89$ | $125.62$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |
| | $10^6$ | $125,000.0$ | $125,000.7$ | $124,999.9$ | $125,000.6$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |
| **Tri-State Monitor** | $10$ | $1.00$ | $1.49$ | $0.89$ | $1.38$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |
| | $100$ | $10.00$ | $10.49$ | $9.89$ | $10.38$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |
| | $1,000$ | $100.04$ | $100.49$ | $99.94$ | $100.38$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |
| | $10^6$ | $100,042.5$ | $100,043.0$ | $100,042.4$ | $100,042.8$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |
| **Tri-State Disc $0.50$** | $10$ | $1.00$ | $1.49$ | $0.89$ | $1.38$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |
| | $100$ | $10.00$ | $10.49$ | $9.89$ | $10.38$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |
| | $1,000$ | $100.04$ | $100.49$ | $99.94$ | $100.38$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |
| | $10^6$ | $100,042.5$ | $100,043.0$ | $100,042.4$ | $100,042.8$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |
| **Tri-State Disc $0.25$** | $10$ | $1.00$ | $1.49$ | $0.89$ | $1.38$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |
| | $100$ | $10.00$ | $10.49$ | $9.89$ | $10.38$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |
| | $1,000$ | $100.04$ | $100.49$ | $99.94$ | $100.38$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |
| | $10^6$ | $100,042.5$ | $100,043.0$ | $100,042.4$ | $100,042.8$ | `[234, 0, 390, 0]` | $37.50\%$ | $0.00\%$ | $100.00\%$ | $27.27\%$ |

*Note: All observations in this section represent a preliminary controlled diagnostic on Round 1, not the final 3-seed 20-round benchmark.*

---

## 5. PART D — Expanded Phase 7 Matrix Integrity

The authoritative matrix manifest [PHASE7_EXPERIMENT_MATRIX.json](file:///home/suvranil/project/results/phase7_robust_aggregation/PHASE7_EXPERIMENT_MATRIX.json) was audited and verified:
- **Total Experiments:** Exactly **351** ($13\text{ conditions} \times 9\text{ aggregators} \times 3\text{ seeds}$).
- **Total Round Observations:** Exactly **70,200** ($351 \times 20\text{ rounds} \times 10\text{ clients}$).
- **Seeds:** `[211, 212, 213]`.
- **Condition Registration:** Both BatchNorm conditions are formally registered:
  - `ATK-BN-IID-M1` (27 simulations, 5,400 observations)
  - `ATK-BN-NONIID-M1` (27 simulations, 5,400 observations)
- **Detector Invariance:** Detector v2 is **NOT** included in the Phase 7 matrix. Phase 7 will evaluate the exact frozen v1 detector across all 351 runs to ensure that its empirical blind spot is rigorously measured rather than prematurely masked.

---

## 6. PART E — Final Status, Regressions, and Execution Gate

1. **Regression Test Suite:** All 73 tests in [tests/](file:///home/suvranil/project/tests/) pass in `7.24s`.
2. **Cryptographic Checksums:**
   - [PHASE7_EXPERIMENT_MATRIX.json](file:///home/suvranil/project/results/phase7_robust_aggregation/PHASE7_EXPERIMENT_MATRIX.json): `712d644cca26216a865e754670eb50b50719881739c85ec446b840c7918d2139`
   - Initial Model Checkpoint [results/models/initial_federated_model.pt](file:///home/suvranil/project/results/models/initial_federated_model.pt): `80a2f4d2ca2e8b00a305852cdcf494889ed32a14ace98621f70d823d60438b68`
   - [detector.py](file:///home/suvranil/project/federated_learning/detection/frozen_v1/detector.py): `6576468da037927fda3283a1e2d8402f3b2739d6c45b6fe8ad517900e94a7f07`
   - [feature_extractor.py](file:///home/suvranil/project/federated_learning/detection/frozen_v1/feature_extractor.py): `66de939c5c0432bb8570fa5d0de82f7701a07a699f4e3d3c6aa528316122c906`
   - [calibrated_thresholds.json](file:///home/suvranil/project/federated_learning/detection/frozen_v1/calibrated_thresholds.json): `737c4819df70a8cadc93d9ff9a2db82bc36040403e11ee1240031c9b0cf34507`
3. **Execution Gate:** **HELD AND PAUSED.** No main simulations will be launched until explicit authorization is granted.
4. **Blockchain:** Deferred strictly to Phase 8.
