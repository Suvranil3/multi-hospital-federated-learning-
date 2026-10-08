# The Hospital Scenario: Multi-Institutional Federated Healthcare

## 1. Clinical Overview

```text
Hospital 1  (Pediatric Pulmonary Center) ─┐
Hospital 2  (Metropolitan General)       ─┤
Hospital 3  (Regional Community Clinic)   ─┤
Hospital 4  (University Academic Center)  ─┤
Hospital 5  (Suburban Health Network)    ─┤
Hospital 6  (Infectious Disease Institute)─┤── Federated Orchestration ── Global Model
Hospital 7  (Rural Healthcare System)    ─┤   (Flower + FedAvg)         (Pneumonia Detection)
Hospital 8  (Pediatric Urgent Care)      ─┤
Hospital 9  (Specialized Lung Hospital)  ─┤
Hospital 10 (Diagnostic Imaging Center)  ─┘
```

In healthcare, no single hospital possesses sufficient patient diversity to train an AI diagnostic system that generalizes universally. Cross-institutional collaboration is essential to expose machine learning models to diverse patient demographics, imaging equipment variations, and pathology severities.

However, clinical collaboration faces a severe privacy barrier:
- **Patient Privacy Laws**: HIPAA, GDPR, and hospital ethics boards strictly prohibit sharing raw patient imaging scans across institutional borders.
- **Data Silos**: Each hospital operates as an isolated data silo, holding confidential chest X-rays behind hospital firewalls.

**Federated Learning (FL)** bridges this gap by decoupling machine learning training from data centralisation.

---

## 2. The Federated Training Protocol in Healthcare

1. **Local Model Initialization**: The central FL coordinator broadcasts the initial [`SmallCNN`](file:///home/suvranil/project/federated_learning/model.py#L21-L71) global model parameters $\mathbf{w}_0$ to all 10 participating hospitals.
2. **On-Premise Local Training**: Each hospital trains the model for local epochs using its own private chest X-rays. **Patient images never leave the local hospital firewall.**
3. **Weight Delta Computation**: Each hospital computes its parameter update:
   $$\Delta \mathbf{w}_i^{(t)} = \mathbf{w}_i^{(t)} - \mathbf{w}_{\text{global}}^{(t)}$$
4. **Update Transmission**: Hospitals upload only the parameter deltas $\Delta \mathbf{w}_i^{(t)}$ to the central aggregation server.
5. **Secure Aggregation**: The central server aggregates updates (via FedAvg or Byzantine-robust rules) to produce the improved global model $\mathbf{w}_{\text{global}}^{(t+1)}$.
6. **Model Broadcast**: The updated global model is sent back to all hospitals, improving diagnostic performance for all participants.

---

## 3. The Core Research Problem: Benign Skew vs. Malicious Attack

In subsequent research phases, we study a fundamental dilemma in healthcare FL:

### Case A: The Honest Specialized Hospital (Non-IID Skew)
- Consider **Hospital 1** (a specialized pediatric pulmonary clinic) versus **Hospital 3** (a general community checkup center).
- Hospital 1 treats severe pneumonia cases referred from across the region (high pneumonia prevalence, distinct clinical patterns).
- Hospital 3 performs routine screenings where 90% of scans are normal.
- Because their patient populations differ drastically (**non-IID clinical skew**), Hospital 1's legitimate gradient updates will point in a direction different from the majority consensus.
- **The Problem**: Conventional Byzantine filters (e.g., Krum, Trimmed Mean) look only at distance from the average. They mistakenly classify Hospital 1 as an adversary, discarding its valuable clinical knowledge (**high False Positive Rate**).

### Case B: The Malicious or Compromised Hospital (Poisoning Attack)
- Suppose **Hospital 9** is compromised by an adversary or has a malfunctioning radiomics software pipeline.
- The adversary mounts a **label flipping attack** (flipping pneumonia images to normal) or a **gradient scaling attack** designed to degrade the model's sensitivity.
- If accepted, this malicious update damages the global model, causing it to miss life-threatening pneumonia infections in clinical practice.

### The Research Question
> **"How can the central server distinguish an honest specialized hospital (which has naturally divergent clinical data) from a malicious hospital (which is submitting a deliberately harmful update)?"**

---

## 4. Multi-Factor Detection & Tri-State Consensus Policy

To resolve this dilemma, our planned security layer (Phase 6) moves beyond simplistic geometric distance by evaluating multiple orthogonal indicators:
1. **Directional Cosine Similarity**: Evaluates gradient angle relative to historical momentum.
2. **Local Validation Impact**: Tests candidate updates on a balanced benchmark validation proxy to detect sudden diagnostic accuracy drops.
3. **Historical Trajectory Consistency**: Tracks whether a hospital's divergence is consistent with its long-term clinical profile, or whether it exhibits sudden erratic shifts characteristic of an attack.
4. **Clinical Heterogeneity Context**: Accounts for hospital case-mix variance before applying outlier penalties.

Based on this evaluation, updates receive a **Tri-State Decision**:
- **Normal**: Full acceptance ($w = 1.0$).
- **Suspicious**: Down-weighted ($0 < w < 1.0$), preventing model disruption while preserving minority domain knowledge.
- **Malicious**: Complete rejection ($w = 0.0$), shielding the medical model from corruption.

---

## 5. Blockchain as a Healthcare Audit Trail

In clinical AI, accountability and regulatory oversight are paramount. If an AI model contributes to an erroneous diagnosis, hospital authorities require a tamper-evident audit trail:
- **What is logged on-chain**: Communication round, hospital ID, SHA-256 hash of the submitted model update, detector classification (`Normal`, `Suspicious`, `Malicious`), and action weight.
- **What is NEVER logged on-chain**: Raw chest X-rays, patient protected health information (PHI), or heavy neural network weights.
- **Benefit**: Provides non-repudiation, tamper-evident governance, and transparent compliance for hospital ethics committees and healthcare regulators.
