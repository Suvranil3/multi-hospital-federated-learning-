# Research Gap & Literature Context

## 1. Background: Byzantine Robustness in Clinical Federated Learning
Federated Learning (FL) is uniquely suited for multi-hospital collaboration because hospital patient data (e.g., chest X-rays) cannot leave institutional firewalls due to medical privacy laws. However, if any participating hospital or client node is compromised by an adversary, classical aggregation (such as FedAvg) is vulnerable to toxic model updates.

To counter malicious updates, various Byzantine-robust aggregation mechanisms exist:
- **Distance / Selection-Based**: Krum, Multi-Krum (Blanchard et al., 2017)
- **Coordinate-wise Statistics**: Trimmed Mean, Coordinate-wise Median (Yin et al., 2018)
- **Hybrid Schemes**: Bulyan (Mhamdi et al., 2018)
- **Clustering / Cosine Similarity**: FoolsGold (Fung et al., 2020), FLTrust (Cao et al., 2021)

---

## 2. Research Questions in a Multi-Hospital Setting Under Non-IID Skew

While Byzantine-robust aggregators perform adequately under Independent and Identically Distributed (IID) assumptions, critical vulnerabilities arise in real-world healthcare networks:

### Investigation 1: How do geometric filters perform when honest hospitals treat specialized patient populations?
Distance-based schemes (e.g., Krum, Multi-Krum) assume benign updates cluster tightly in parameter space, while malicious updates act as geometric outliers.
- **Clinical Reality**: In a multi-hospital collaborative network, Hospital A may be a specialized pediatric hospital, while Hospital B is an adult pulmonary clinic. Their case mixes and pathology prevalence differ drastically (non-IID clinical skew).
- **Research Question**: Do legitimate gradients from honest specialized hospitals appear as geometric outliers, causing Multi-Krum to discard valid diagnostic insights and produce high False Positive Rates (FPR) against honest hospitals?

### Investigation 2: What are the limits of coordinate-wise trimming under medical case-mix skew?
Coordinate-wise Trimmed Mean and Median eliminate extreme coordinate values independently.
- **Research Question**: When natural clinical diversity across hospitals causes model weights to display high variance along specific pathology-related feature maps, does trimming systematically discard crucial domain knowledge from specialized hospitals?
- **Trade-off to Investigate**: Does loosening trimming thresholds to preserve clinical diversity inadvertently widen the attack surface for stealthy poisoning or targeted backdoors?

### Investigation 3: Can historical behavioral consistency differentiate benign clinical variance from adversarial attacks?
- **Observation**: An honest specialized hospital generates updates that, while divergent from the majority consensus, remain consistent with its own historical clinical trajectory across rounds. Conversely, an adversary launching model poisoning or a backdoor introduces sharp, unnatural shifts in gradient dynamics.
- **Research Question**: Can tracking historical hospital update consistency alongside cosine similarity, validation loss impact, and case-mix context reliably separate benign clinical diversity from malicious attacks?

---

## 3. Our Focused Contribution
Rather than proposing generic "trust scores" or computationally unfeasible on-chain AI, our research investigates:
1. **Multi-Factor Clinical Heterogeneity-Aware Detection**: A modular detector combining update deviation ($L_2$), directional cosine similarity, validation performance impact, historical behavioral trajectory, and clinical heterogeneity context.
2. **Tri-State Action Policy**: Classifying updates into `Normal` (full weight), `Suspicious` (down-weighted), and `Malicious` (rejected), preventing harsh binary exclusion of specialized hospitals.
3. **Decoupled Blockchain Medical Audit Trail**: Using an Ethereum-compatible ledger strictly as an append-only, tamper-evident log of update hashes, hospital IDs, and detector classifications for clinical governance—zero patient data and zero heavy model weights on-chain.
