# Research Gap & Literature Context

## 1. Background: Byzantine Robustness in Federated Learning
Federated Learning (FL) relies on decentralized contributions from edge participants. In an adversarial setting where a fraction of participants may be compromised, classical aggregation (such as McMahan et al.'s FedAvg) is vulnerable because a single unconstrained model update can steer the global parameter vector arbitrarily.

To mitigate this vulnerability, several Byzantine-robust aggregation mechanisms have been proposed:
- **Distance / Selection-Based**: Krum, Multi-Krum (Blanchard et al., 2017)
- **Coordinate-wise Statistics**: Trimmed Mean, Coordinate-wise Median (Yin et al., 2018)
- **Hybrid Schemes**: Bulyan (Mhamdi et al., 2018)
- **Clustering / Similarity**: FoolsGold (Fung et al., 2020), FLTrust (Cao et al., 2021)

---

## 2. Research Questions Regarding Existing Aggregators Under Non-IID Skew

While Byzantine-robust aggregators demonstrate theoretical and empirical resilience under Independent and Identically Distributed (IID) or mildly heterogeneous assumptions, critical research questions emerge when deploying them in realistic, highly non-IID edge scenarios:

### Investigation 1: How do distance-based filters perform when honest updates naturally diverge?
Distance-based schemes (e.g., Krum, Multi-Krum) select updates minimizing Euclidean distance to their neighbors under the assumption that benign updates cluster together while malicious updates act as geometric outliers.
- **Research Question**: In non-IID environments where honest clients possess specialized local class distributions (e.g., Dirichlet $\alpha = 0.5$ or $\alpha = 0.1$), do benign gradients naturally form disparate clusters or extreme individual outliers?
- **Hypothesized Vulnerability to Test**: If honest specialized clients appear as geometric outliers, does Multi-Krum disproportionately exclude valid, minority-class updates, leading to high False Positive Rates (FPR) against honest clients and degraded global generalization on under-represented classes?

### Investigation 2: What are the limits of coordinate-wise trimming under extreme heterogeneity?
Coordinate-wise Trimmed Mean and Median eliminate or attenuate extreme parameter values independently per coordinate.
- **Research Question**: When data heterogeneity causes honest model weights across disparate clients to possess high variance along specific feature coordinates, does fixed-percentage trimming systematically discard legitimate domain knowledge from honest non-IID participants?
- **Trade-off to Investigate**: Does tuning trimming thresholds to preserve honest non-IID variance inadvertently widen the attack surface for subtle, low-magnitude poisoning attacks (e.g., stealthy model scaling or targeted backdoors)?

### Investigation 3: Can historical behavioral consistency differentiate non-IID variance from adversarial shifts?
- **Observation to Explore**: An honest non-IID client typically produces updates that, while divergent from the global average, remain consistent with its own historical trajectory and local distribution profile across communication rounds. Conversely, an adversary launching model poisoning or sudden backdoor injection may exhibit sharp behavioral shifts or unnatural cross-round dynamics.
- **Research Question**: Can tracking historical client behavioral metrics alongside update similarity and validation loss impact provide a reliable discriminative signal that reduces false positives without compromising attack detection recall?

---

## 3. The Distinction from Generic "Trust-Score" Approaches
A common pattern in recent literature is proposing generic "trust scores" or reputation systems coupled with blockchain. However:
- Generic trust score systems often lack formal grounding in FL gradient dynamics and fail to distinguish benign distribution shift from intentional poisoning.
- Storing entire models or computing complex reputation matrices directly on smart contracts introduces prohibitive gas costs and high transaction latency.

### Our Focused Contribution:
Rather than claiming a generic trust architecture, our project explicitly investigates:
1. **Multi-Factor Heterogeneity-Aware Detection**: A modular detector combining update deviation, directional cosine similarity, local validation impact, client historical consistency, and data heterogeneity indicators.
2. **Tri-State Action Policy**: Classifying participants into `Normal` (full weight), `Suspicious` (down-weighted), and `Malicious` (rejected), allowing the system to handle borderline ambiguous updates softly rather than imposing binary exclusion.
3. **Lightweight Decoupled Blockchain Audit Trail**: Using blockchain strictly as an append-only, tamper-evident cryptographic log of model update hashes and detector decisions—ensuring transparent auditability without burdening the ledger with ML computations or raw model weights.
