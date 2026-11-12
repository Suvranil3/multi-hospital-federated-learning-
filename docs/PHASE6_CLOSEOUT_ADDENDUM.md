# Phase 6 Close-Out Addendum: Runtime Verification, Historical Reproduction & Medical Evaluation Audit

**Date:** 2026-11-01  
**Status:** FORMAL CLOSE-OUT ACCEPTED & AUDITED  
**Subject:** Rigorous Technical Audit of Phase 6 Blind Evaluation (Seeds 111, 112, 113)  

---

## 1. Runtime Detector Import & Cryptographic Integrity (Part A1)

To confirm that the blind evaluation runner ([`experiments/run_phase6_blind_evaluation.py`](file:///home/suvranil/project/experiments/run_phase6_blind_evaluation.py)) strictly loaded the frozen detector rather than mutable development files, dynamic import inspection was performed:

- **Runtime `__file__` Path:**  
  `/home/suvranil/project/federated_learning/detection/frozen_v1/detector.py`
- **Runtime Loaded Source SHA-256:**  
  `6576468da037927fda3283a1e2d8402f3b2739d6c45b6fe8ad517900e94a7f07`
- **Expected Manifest Source SHA-256:**  
  `6576468da037927fda3283a1e2d8402f3b2739d6c45b6fe8ad517900e94a7f07`
  *(Bit-for-bit identity: **TRUE**)*
- **Runtime Loaded Thresholds Path:**  
  `/home/suvranil/project/federated_learning/detection/frozen_v1/calibrated_thresholds.json`
- **Runtime Loaded Thresholds SHA-256:**  
  `737c4819df70a8cadc93d9ff9a2db82bc36040403e11ee1240031c9b0cf34507`
- **Expected Manifest Thresholds SHA-256:**  
  `737c4819df70a8cadc93d9ff9a2db82bc36040403e11ee1240031c9b0cf34507`
  *(Bit-for-bit identity: **TRUE**)*
- **Freeze Manifest Version:** `v1.0-phase6-development-benchmark`
- **Filesystem Permissions:** Read-only (`-r--r--r--`, `chmod 444`) on all frozen components throughout execution.

**Conclusion:** The evaluation exclusively executed the cryptographically sealed detector with zero runtime deviation or parameter mutability.

---

## 2. Reproduction of Historical Hospital 6 Observation (Part A2)

We evaluated the historical development record from `results/phase6_detector/instrumentation/exp-med-honest-a01-s45_features.jsonl` corresponding to:
**Experiment `EXP-MED-HONEST-A01-S45`, Client ID `6`, Round `10`**

### 2.1 Recorded Feature Vector
- `norm_global`: 5.0962
- `z_mag_global`: 58.4676
- `norm_ratio_global`: 2.8869
- `max_layer_z_mag`: 67.7457 (located in `dense_projection`)
- `layer_z_mag`:
  - `conv_block_1`: -3.8846
  - `conv_block_2`: +1.5332
  - `dense_projection`: **+67.7457**
  - `classification_head`: +0.0295
- `cos_ref_global`: +0.3974
- `min_layer_cos`: -0.5707 (in `classification_head`)
- `is_sign_inverted`: False
- `delta_loss_total`: +0.050662
- `delta_loss_normal`: +0.004998
- `delta_loss_pneumonia`: +0.066510
- `delta_acc_total`: 0.0%

### 2.2 Frozen Detector Evaluation
- **Final Decision:** `DetectionDecision.MALICIOUS` (`MALICIOUS`)
- **Anomaly Score:** 0.629
- **Flagged Reason:**  
  `"Corroborated layer-wise norm explosion: max_layer_Z=67.75 > 12.41 with uncompensated harm"`
- **Condition Analysis**:
  1. *Global Magnitude Rule*: Required `norm_ratio_global > 5.0` AND `z_mag_global > 9.10`. Since `norm_ratio_global` was 2.887 (< 5.0), the global magnitude rule did NOT trip.
  2. *Layer-Wise Magnitude Rule*: Layer threshold is `layer_z_mag = 12.41`. The measured `max_layer_z_mag` was **67.75** (> 12.41), tripping the layer-wise explosion flag.
  3. *Cosine Rule*: Global cosine was positive (+0.3974); minimum layer cosine was -0.5707 in classification head.
  4. *Validation-Loss Trade-Off Conditions*: Both classes exhibited positive validation loss increases ($\Delta \mathcal{L}_{\text{normal}} = +0.0050 > 0$, $\Delta \mathcal{L}_{\text{pneumonia}} = +0.0665 > 0$). Because neither class improved, Symmetrical Clinical Trade-Off Protection did NOT engage (`symmetrical_tradeoff_protected = False`).
  5. *Exact Triggering Rule*:
     `if max_layer_z > thresh.layer_z_mag and not symmetrical_protected and delta_loss_total > 0:` $\to$ **`MALICIOUS`**.

### 2.3 Scientific Explanation of Historical Agreement
In the historical development audit (`EXP-MED-018-FORENSIC-AUDIT`), this exact event was reported as the single hard false positive across 3,200 exploratory honest observations. The frozen detector produces **the exact same result**. The detector was not retroactively modified or softened to force a cosmetic zero on historical seed 45, validating that the frozen detector remains completely unmodified.

---

## 3. Comprehensive Clinical Evaluation Reporting (Part A3)

The official test split ($N=624$ samples: 234 Normal, 390 Pneumonia) was evaluated strictly post-training using the best validation checkpoint for each of the 33 blind simulations. No test set data was used during training or checkpoint selection.

### 3.1 Medical Scorecard: Condition Means and Standard Deviations ($N=33$ Runs)

| Experimental Condition | Clean Test Accuracy (%) | Pneumonia Sensitivity (%) | Normal Specificity (%) | Macro-Averaged F1 | Mean False Negatives | Mean False Positives |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Honest IID** | 84.83 ± 0.92 | 97.95 ± 0.44 | 62.96 ± 3.00 | 0.8232 ± 0.0127 | 8.0 | 86.7 |
| **Honest Non-IID $\alpha=1.0$** | 80.18 ± 2.30 | 98.72 ± 0.68 | 49.29 ± 7.27 | 0.7552 ± 0.0373 | 5.0 | 118.7 |
| **Honest Non-IID $\alpha=0.5$** | 75.80 ± 1.15 | 99.57 ± 0.39 | 36.18 ± 3.64 | 0.6826 ± 0.0217 | 1.7 | 149.3 |
| **Honest Non-IID $\alpha=0.1$** | 63.89 ± 2.41 | 99.74 ± 0.44 | 4.13 ± 7.15 | 0.4241 ± 0.0684 | 1.0 | 224.3 |
| **Label Flipping IID $M=1$** | 88.78 ± 0.64 | 95.38 ± 0.26 | 77.78 ± 1.54 | 0.8763 ± 0.0075 | 18.0 | 52.0 |
| **Label Flipping IID $M=2$** | 88.67 ± 0.79 | 88.55 ± 1.50 | 88.89 ± 0.43 | 0.8810 ± 0.0077 | 44.7 | 26.0 |
| **Label Flipping Non-IID $\alpha=0.5$ $M=1$** | 80.45 ± 2.49 | 98.63 ± 1.04 | 50.14 ± 8.34 | 0.7592 ± 0.0401 | 5.3 | 116.7 |
| **Model Inversion IID ($\gamma=-1.0$)** | 86.27 ± 0.88 | 96.67 ± 0.26 | 68.95 ± 2.02 | 0.8440 ± 0.0108 | 13.0 | 72.7 |
| **Model Scaling IID ($\gamma=+10.0$)** | 79.76 ± 1.21 | 99.32 ± 0.15 | 47.15 ± 3.48 | 0.7476 ± 0.0192 | 2.7 | 123.7 |
| **Backdoor IID ($p=0.50$)** | 86.17 ± 1.20 | 97.61 ± 0.39 | 67.09 ± 3.73 | 0.8411 ± 0.0160 | 9.3 | 77.0 |
| **Backdoor Non-IID $\alpha=0.5$ ($p=0.50$)** | 79.49 ± 1.67 | 98.55 ± 0.15 | 47.72 ± 4.61 | 0.7461 ± 0.0252 | 5.7 | 122.3 |

### 3.2 Clinical Analysis
- **Pneumonia Sensitivity Preservation:** Across honest and attack runs, sensitivity for the high-risk Pneumonia class was consistently high ($88.55\% - 99.74\%$), ensuring few false negatives (averaging $\le 8$ in IID honest, down to $1-2$ in Non-IID).
- **Heterogeneity-Induced Specificity Degradation:** Under unweighted FedAvg, increasing Dirichlet skew causes Normal specificity to decline from $62.96\%$ in IID to $4.13\%$ in extreme Non-IID $\alpha=0.1$. Because Pneumonia cases constitute $74.2\%$ of the training data, unweighted aggregation biases updates toward Pneumonia predictions unless specialized minority hospitals are preserved.
- **Label Flipping Dynamic:** Label Flipping ($M=2$, flipping Pneumonia to Normal) forced the model toward predicting Normal, artificially raising Normal specificity to $88.89\%$ while depressing Pneumonia sensitivity to $88.55\%$ (causing false negatives to jump from 8 to 44.7).

---

## 4. Backdoor Attack Evaluation & Control Baseline Audit (Part A4)

In the Phase 6 blind evaluation suite, triggered test accuracy was evaluated on all backdoor poisoning simulations using `TriggeredPneumoniaTestDataset` ($N=390$ Pneumonia test cases with $3 \times 3$ white pixel patch trigger targeting Class 0):

- **Backdoor IID ($M=1, p=0.50$):**
  - Seed 111: Clean Accuracy = 84.78%, Triggered ASR = **5.38%** (21 / 390)
  - Seed 112: Clean Accuracy = 86.86%, Triggered ASR = **8.72%** (34 / 390)
  - Seed 113: Clean Accuracy = 86.86%, Triggered ASR = **6.92%** (27 / 390)
  - **Mean Triggered ASR:** **7.01% ± 1.67%**
- **Backdoor Non-IID $\alpha=0.5$ ($M=1, p=0.50$):**
  - Seed 111: Clean Accuracy = 81.41%, Triggered ASR = **1.79%** (7 / 390)
  - Seed 112: Clean Accuracy = 78.69%, Triggered ASR = **3.85%** (15 / 390)
  - Seed 113: Clean Accuracy = 78.37%, Triggered ASR = **8.97%** (35 / 390)
  - **Mean Triggered ASR:** **4.87% ± 3.73%**

### Matched Control Baseline Audit:
The blind evaluation runner [`run_phase6_blind_evaluation.py`](file:///home/suvranil/project/experiments/run_phase6_blind_evaluation.py) evaluated triggered test sets only when `attack_type == "backdoor"`. Consequently, for fresh seeds 111, 112, 113, **matched honest-control triggered ASR runs were not recorded**.

*Formal Disclosure:* While historical Phase 5 honest baseline models (seed 42) exhibited an empirical baseline trigger misclassification rate of $0.26\% - 0.51\%$, the Phase 6 blind suite report measures **absolute triggered test ASR** and does not establish a causal, matched-control percentage-point delta for seeds 111–113. In Phase 7, matched-control triggered evaluation will be enforced across all honest control checkpoints to measure precise attack-induced ASR elevation.

---

## 5. Statistical Interpretation & Scope Limitations (Part A5)

### 5.1 Observation vs Simulation Denominators
- **Client-Round Observations:** $N = 6,600$ total (6,120 honest, 480 malicious).
- **Independent Simulations:** $M = 33$ total (12 honest control runs, 21 attack runs).
- **Independent Seed Realizations:** 3 seeds (`111, 112, 113`).

### 5.2 Confidence Intervals & Rule of Three
- Observed Hard Malicious False Positives: $k = 0$ out of $N = 6,120$ honest client-rounds.
- By the Hanley-Lipman-Hand Rule of Three, the 95% upper confidence bound on the client-round false positive probability is:
  $$p_{95\%} \le \frac{3}{N} = \frac{3}{6,120} \approx 0.049\% \quad (4.9 \times 10^{-4})$$
- *Critical Caveat:* Client-round observations within a single simulation share the same data split and global parameter trajectory and are not identically and independently distributed.
- Across independent simulations ($N_{sim} = 12$ pure honest controls), the observed simulation-level false exclusion rate is $0 / 12$ ($p_{95\%} \le 25.0\%$).

### 5.3 Limitations of Three Seeds
Evaluating across three seeds establishes multi-seed reproducibility and bounds typical variance under Dirichlet sampling. However, it cannot guarantee the absence of rare partition edge cases (such as the low-MAD artifact found in historical seed 45) that occur with low probability under Dirichlet concentration $\alpha=0.1$.

---

## 6. Project Status Updates

The Phase 6 deliverables have been accepted and cryptographically archived:
1. `results/phase6_blind_evaluation/` contains all 33 simulation feature streams and checkpoints.
2. `results/phase6_preflight_exploratory_seed101/` contains all preserved exploratory runs.
3. `docs/PHASE6_BLIND_EVALUATION_REPORT.md` provides the primary benchmark scorecard.
4. `EXPERIMENT_LOG.md` records entry `[EXP-MED-020-BLIND-EVALUATION]`.
5. `PROJECT_STATUS.md` and `docs/CURRENT_RESULTS.md` have been updated.

Phase 6 is formally closed. Phase 7 implementation authorization is granted below.
