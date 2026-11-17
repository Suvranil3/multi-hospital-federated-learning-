# Formal Statement of Research Contributions

**Project**: Multi-Hospital Federated Learning for Chest X-Ray Classification with Malicious Participant Detection and Blockchain Audit  
**Document**: `docs/RESEARCH_CONTRIBUTIONS.md`  
**Date**: November 17, 2026  
**Status**: Authoritative Definition of Problem, Contributions, Empirical Discoveries, and Limitations  

---

## 1. Research Problem

Federated Learning (FL) enables cross-institutional collaboration among healthcare providers without sharing confidential patient radiographs, satisfying institutional data sovereignty and privacy mandates (HIPAA, GDPR). However, when deployed across a distributed hospital network, two fundamental phenomena collide:

1. **Clinical Data Heterogeneity (Non-IID Case-Mix Skew)**: Real-world healthcare facilities treat demographically and epidemiologically distinct patient populations. Specialized pulmonary clinics treat a high proportion of severe pathologies, whereas community checkup clinics predominantly encounter healthy individuals. Gradient updates from specialized institutions naturally diverge in parameter space from the collective consensus.
2. **Adversarial Update Poisoning**: Dishonest, compromised, or misconfigured participants can inject poisoned model updates (such as label-flipping perturbations, stealthy trojan backdoors, or corrupted batch normalization buffers) designed to compromise global diagnostic accuracy or create targeted vulnerabilities.

The core research question is:
> **How can a federated aggregation server distinguish between an honest specialized hospital (whose gradient updates diverge due to legitimate clinical case-mix heterogeneity) and a malicious participant (whose gradient updates diverge due to deliberate adversarial poisoning), while maintaining regulatory accountability and clinical diagnostic performance?**

---

## 2. Identified Research Gaps in the Literature

Based on our comprehensive literature review ([`docs/LITERATURE_REVIEW.md`](file:///home/suvranil/project/docs/LITERATURE_REVIEW.md)), we identify four critical, unaddressed gaps in existing federated learning research:

1. **The Homogeneity Assumption in Classical Byzantine Defenses**:
   Foundational Byzantine-robust aggregation rules (such as Krum, Multi-Krum, and coordinate-wise trimming) were developed under the assumption that benign client data is identically distributed (IID). When evaluated under non-IID conditions, geometric Euclidean filters cannot distinguish between benign specialization and malicious deviation, treating legitimate clinical diversity as an adversarial attack.
2. **The High False Exclusion Cost of Binary Defenses**:
   Existing defenses enforce binary filtering: updates are either accepted at full weight or discarded entirely. In healthcare, dropping an honest specialized hospital completely discards irreplaceable minority domain knowledge, inducing severe diagnostic blindness on rare disease subtypes.
3. **The Parameter vs. Buffer Blind Spot**:
   Existing Byzantine literature focuses almost exclusively on trainable parameter tensors (`TRAINABLE_INDICES`). Non-trainable persistent model state buffers—specifically Batch Normalization running statistics (`running_mean` and `running_var`)—are omitted from detector screening and linearly averaged by standard aggregation engines, leaving a catastrophic zero-day vulnerability in federated vision pipelines.
4. **The Lack of Verifiable Regulatory Non-Repudiation**:
   Prior medical FL studies (such as EXAM and BraTS) deploy unweighted FedAvg without security monitoring or tamper-evident governance. If an excluded hospital disputes its exclusion, or if a poisoned model causes a diagnostic failure, traditional servers lack an immutable, cryptographically verifiable record proving why an update was accepted, discounted, or rejected.

---

## 3. Core Research Contributions

This research provides five concrete contributions closing these gaps:

### Contribution 1: Multi-Factor Clinical Update Screening Architecture
Rather than relying on unidimensional geometric distance, we design and implement an intelligent screening layer evaluating five orthogonal anomaly criteria:
- **Global & Layer-Wise Update Magnitude**: Computes the global $L_2$ parameter norm and layer-wise modified Z-scores (median absolute deviation) across convolutional blocks, dense projection, and classification heads to detect scaling anomalies.
- **Directional Consensus**: Measures directional cosine similarity against a robust coordinate-wise median reference model update ($\cos(\Delta \theta_k, \Delta \theta_{\text{median}})$) to detect adversarial sign inversion and trajectory deflection.
- **Class-Disentangled Counterfactual Validation Impact**: Evaluates candidate updates on a balanced server-side benchmark proxy, measuring leave-one-out (LOO) performance shifts independently for Normal and Pneumonia classes ($\Delta \mathcal{L}_{\text{normal}}$, $\Delta \mathcal{L}_{\text{pneumonia}}$, $\Delta \text{Recall}_{\text{normal}}$, $\Delta \text{Recall}_{\text{pneumonia}}$).
- **Symmetrical Clinical Trade-Off Protection**: Protects specialized hospitals exhibiting negative cosine similarity when their updates provide genuine diagnostic improvements on the minority class, preventing false malicious classification.
- **Historical Multi-Round Trajectory Consistency**: Tracks client velocity and updates an exponential moving average (EMA) anomaly accumulator to differentiate stable clinical specialization from erratic adversarial spikes.

### Contribution 2: Calibrated Tri-State Soft-Discount Aggregation
To resolve the binary exclusion dilemma, we implement a calibrated tri-state consensus mechanism:
- `NORMAL` ($w_k = 1.0$): Update is accepted at full sample-proportional weight.
- `SUSPICIOUS` ($w_k = \lambda = 0.50$): Update is discounted by a soft attenuation factor $\lambda \in (0, 1)$, incorporating useful gradient directions while bounding potential adversarial contamination.
- `MALICIOUS` ($w_k = 0.0$): Update is completely purged from aggregation.

### Contribution 3: Empirical Demarcation of Non-IID Label-Flipping Defense
We execute an exhaustive 72-cell balanced primary matrix comparing FedAvg, Multi-Krum ($f=1$), Tri-State Monitor, and Tri-State Discount ($\lambda=0.50$) across 9 experimental regimes and 2 distinct seeds (Seeds 211 and 212):
- Under moderate clinical heterogeneity ($\alpha=0.5$) with a label-flipping adversary ($M=1$), Tri-State Discounting achieves **83.09% mean accuracy and 58.98% specificity**, outperforming standard FedAvg (**+9.13% accuracy / +28.21% specificity**) and Multi-Krum (**+11.37% accuracy / +33.98% specificity**).
- Telemetry audits prove that the detector attenuated the adversary's cumulative weight by **27.92% (Seed 211)** and **36.00% (Seed 212)**, rescuing the global model from single-class diagnostic collapse.

### Contribution 4: Empirical Discovery of Multi-Krum's Minority Rejection Failure
Through granular feature telemetry, we uncover and mathematically explain the exact mechanism behind Multi-Krum's failure under clinical heterogeneity:
- In honest non-IID conditions ($\alpha=0.5$), Multi-Krum suffers a catastrophic specificity collapse down to **23.08%** (vs. FedAvg's 54.48%).
- We prove through client selection logs that Multi-Krum systematically purged the primary Normal-specialized client node (**Client 2 in Seed 211 [97.9% Normal], rejected in 18/20 rounds; Client 3 in Seed 212 [98.3% Normal], rejected in 19/20 rounds**), demonstrating that Euclidean distance-based aggregators induce severe diagnostic bias against minority clinical classes.

### Contribution 5: Discovery of the Batch Normalization Buffer Poisoning Vulnerability
We identify, implement, and analyze a critical vulnerability in federated neural networks:
- We prove that parameter-space anomaly detectors and distance aggregators are structurally blind to non-trainable floating-point buffers (`BN_FLOAT_INDICES`), classifying an adversarial client injecting $10^6$ into running statistics as `NORMAL` in **100% of communication rounds**.
- We show that linear buffer averaging causes a **100% universal diagnostic collapse across all 16 evaluated simulations**, formally distinguishing and cataloging All-Normal (37.50% Acc) and All-Pneumonia (62.50% Acc) collapse states, and formulating concrete non-linear buffer defenses (coordinate-wise median buffers).

### Contribution 6: Cryptographically Audited Architecture with Zero PHI
We demonstrate an immutable, privacy-preserving audit trail by logging 2,134 blocks on a backwards-chained SHA-256 JSONL ledger. Every local model update, detector decision, and performance metric is fingerprinted on-chain without exposing raw patient scans or heavy neural network weights, ensuring complete regulatory non-repudiation.

---

## 4. Summary of Supported Empirical Findings

| Empirical Finding | Metric / Evidence | Supported Scope |
|---|---|---|
| **Non-IID Label-Flipping Defense** | Tri-State Discount: 83.09% Acc / 58.98% Spec vs. FedAvg 73.96% Acc / 30.77% Spec | Moderate skew ($\alpha=0.5$), $M=1$ attacker, Seeds 211 & 212. |
| **Multi-Krum IID Backdoor Defense** | Multi-Krum Backdoor ASR: 2.43% vs. FedAvg 8.46% and Tri-State 5.00% | Clean IID partition, $3 \times 3$ watermark trigger, Seeds 211 & 212. |
| **Multi-Krum Non-IID Rejection** | Multi-Krum Spec: 23.08% (dropped primary Normal client in 90%–95% of rounds) | Honest $\alpha=0.5$ skew, Seeds 211 & 212. |
| **Extreme Skew Class Starvation** | Tri-State Discount collapsed to All-Pneumonia (62.50% Acc / 0.00% Spec) | Extreme skew ($\alpha=0.1$); FedAvg was most resilient (78.77% Acc / 70.72% Spec). |
| **BatchNorm Poisoning Collapse** | 100% failure rate across all 16 runs; Client 0 classified `NORMAL` in 100% of rounds | IID and Non-IID regimes, all 4 aggregators, Seeds 211 & 212. |
| **Blockchain Cryptographic Integrity** | Total 2,134 blocks verified valid (`is_valid = True`, 0 broken links) | Complete Phase 7 execution lifecycle (Tip Block #2133). |

---

## 5. Explicit Limitations & Boundaries

1. **Exploratory Two-Seed Benchmark ($N=2$)**:
   All primary experimental results are evaluated across Seeds 211 and 212. While directional patterns and telemetry proofs are definitive, they represent exploratory benchmark comparisons and **are not claimed as hypothesis-tested statistical significance**.
2. **Synthetic 2D Benchmark vs. Clinical Realities**:
   The evaluation is conducted on PneumoniaMNIST ($28 \times 28$ grayscale images) under synthetic Dirichlet partitioning. Real-world clinical hospital deployments involve 3D DICOM scans, variable spatial resolutions, scanner vendor domain shifts, and complex multi-pathology co-morbidities.
3. **White-Box Adaptive Adversaries**:
   The evaluated attacks (label flipping, static watermark backdoors, extreme buffer scaling) follow established literature threat models. We do not evaluate adaptive white-box adversaries with full knowledge of the detector's multi-factor threshold boundaries.
4. **Extreme Skew Sensitivity**:
   The Phase 6 detector is currently vulnerable to false suspicious classifications under severe class imbalance ($\alpha \le 0.1$), indicating that fixed cosine similarity thresholds must be replaced with adaptive, imbalance-aware thresholding in future work.
