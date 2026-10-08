# Research Gap: Byzantine Robustness in Clinical Federated Learning

## 1. The Conflict Between Clinical Diversity and Byzantine Security

Federated Learning (FL) is recognized as the primary privacy-preserving paradigm for cross-institutional healthcare AI. However, deploying FL across real-world hospital networks exposes a fundamental conflict between **clinical data heterogeneity** and **adversarial defense mechanisms**.

```text
                                  THE HEALTHCARE DILEMMA
                                             │
                   ┌─────────────────────────┴─────────────────────────┐
                   ▼                                                   ▼
       CLINICAL REALITY (Non-IID)                             ADVERSARIAL REALITY
       Hospitals have naturally distinct                      Compromised hospitals can upload
       patient demographics & case mixes.                    poisoned updates to degrade AI.
                   │                                                   │
                   ▼                                                   ▼
       Produces divergent gradient updates                    Produces divergent gradient updates
                   │                                                   │
                   └─────────────────────────┬─────────────────────────┘
                                             │
                                             ▼
                       CONVENTIONAL BYZANTINE FILTERS
                            (Krum, Trimmed Mean)
                                             │
                       Assume ALL divergent updates are attacks!
                                             │
                   ┌─────────────────────────┴─────────────────────────┐
                   ▼                                                   ▼
          FALSE POSITIVE CRISIS                                ATTACK BLIND SPOT
          Excludes honest specialized                          Attacker disguises poisoned
          hospitals, losing clinical expertise.                updates as "natural non-IID skew".
```

---

## 2. Failure Modes of Existing Aggregators in Healthcare

### 2.1. Distance-Based Filters (Krum, Multi-Krum)
- **Mechanism**: Computes pairwise Euclidean distances between client updates and selects updates minimizing distance to the cluster centroid.
- **Why It Fails in Hospitals**:
  - A specialized pediatric hospital treating severe pneumonia produces legitimate weight updates that lie far from the majority consensus of general outpatient clinics.
  - Multi-Krum measures only geometric Euclidean distance. It interprets the honest specialized hospital as an extreme geometric outlier and discards its update entirely.
  - **Consequence**: The global model never learns to recognize the specialized pathology patterns treated by that hospital (**high False Positive Rate against honest hospitals**), harming patients who visit specialized clinics.

### 2.2. Coordinate-Wise Trimming (Trimmed Mean, Coordinate-Wise Median)
- **Mechanism**: Sorts parameter values across clients for each coordinate independently and discards the highest and lowest $\beta$ percentage of values.
- **Why It Fails in Hospitals**:
  - If a hospital encounters a unique clinical presentation or demographic variation, the convolutional filters detecting those specific visual features will undergo substantial gradient adjustments.
  - Fixed-percentage trimming automatically truncates these high-magnitude activations, assuming they are poisoned outliers.
  - **Consequence**: Legitimate clinical discoveries and domain-specific knowledge are systematically trimmed out of the global model.

---

## 3. The "Non-IID Excuse" Attack Vector

In conventional FL research, adversaries exploit the fact that aggregation servers expect some degree of variance:
- An attacker compromised at a hospital node submits a subtle poisoning update (e.g., small scaling or targeted backdoor) and claims the update's divergence is simply due to "unique local patient demographics."
- Without historical tracking or clinical context, the server cannot verify whether the divergence is genuine medical specialization or a stealthy attack.

---

## 4. The Research Gap Addressed by This Project

Most existing literature focuses on synthetic IID datasets or generic image classification (CIFAR-10, MNIST). Very few studies investigate **the intersection of realistic medical non-IID distributions with Byzantine threat modeling**.

Our research specifically closes this gap through three innovations:

### 1. Multi-Factor Contextual Detection
Rather than relying on geometric distance alone, our detector cross-examines updates across five orthogonal dimensions:
- **Gradient Deviation ($L_2$ Norm)**: Is the update magnitude statistically abnormal?
- **Directional Cosine Similarity**: Does the update angle oppose the global consensus trajectory?
- **Validation Loss / Diagnostic Impact**: Does applying this candidate update degrade sensitivity or accuracy on a balanced benchmark validation subset?
- **Historical Trajectory Consistency**: Is this hospital's divergence consistent with its historical multi-round behavior, or does it represent a sudden anomalous shift?
- **Heterogeneity Normalization**: Are outlier thresholds dynamically adjusted based on the hospital's documented clinical case-mix variance?

### 2. Tri-State Action Policy
Existing Byzantine defenses impose binary exclusion (accept or drop). In healthcare, dropping an honest hospital loses valuable clinical insights. Our tri-state policy:
- `Normal`: Fully aggregated ($w = 1.0$).
- `Suspicious`: Down-weighted ($0 < w < 1.0$), providing a soft margin that incorporates useful gradients while bounding potential harm.
- `Malicious`: Completely dropped ($w = 0.0$).

### 3. Regulatory Non-Repudiation via Blockchain
Healthcare institutions demand formal accountability. By logging cryptographic hashes of updates and detection decisions to an immutable ledger, hospital administrators can verify whether an excluded update was rightfully flagged, preventing malicious attacks while ensuring ethical fairness in collaborative AI.
