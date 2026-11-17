# 🎓 Minor Project Live Demonstration & Faculty Defense Guide

**Project Title:** Multi-Hospital Federated Learning for Chest X-Ray Classification with Malicious Participant Detection and Tamper-Evident Blockchain Audit  
**Author:** Suvranil Ghatak  
**Presentation Target:** Faculty Evaluation Committee / Academic Examiners  
**Estimated Duration:** 10–15 Minutes (plus Q&A)  
**Interactive Tool:** Streamlit Live Demonstration Dashboard (`dashboard/app.py`)

---

## ⏱️ Recommended Timing Breakdown (12 Minutes Total)

| Slide | Topic | Focus | Duration |
| :---: | :--- | :--- | :---: |
| **1** | Clinical Dilemma & Regulatory Gap | HIPAA/GDPR data silos & adversarial threat | 1.0 min |
| **2** | Medical Dataset (PneumoniaMNIST) | Official splits, class imbalance, clinical sensitivity | 1.0 min |
| **3** | Four-Tier System Architecture | Data, FL, Defense, Blockchain interaction | 1.0 min |
| **4** | 10-Hospital Federated Learning | FedAvg consensus, +5.13% gain over centralized | 1.0 min |
| **5** | Clinical Heterogeneity (Non-IID) | Dirichlet case-mix, negative cosine discovery | 1.5 min |
| **6** | Adversarial Attack Suite (Phase 5) | Missed pneumonia (+757%), backdoor amplification | 1.0 min |
| **7** | Non-IID-Aware Detector (Phase 6) | 0.9814 AUROC, 0.00% False Malicious, Clinical Shield | 1.5 min |
| **8** | Robust Aggregation (Phase 7) | Specificity rescued (+28%), Multi-Krum minority failure | 2.0 min |
| **9** | Cryptographic Blockchain Audit | 2,134 blocks, SHA-256 hash chaining, FDA compliance | 1.0 min |
| **10** | Final Conclusions & Roadmap | Completed deliverables, 89/89 tests, future scope | 1.0 min |

---

## 🗣️ Slide-by-Slide Verbatim Speaking Script

### Slide 1: Problem Statement & Clinical Dilemma
> **"Good morning, respected faculty members and examiners.**
> 
> Today, I present our completed minor project: *Multi-Hospital Federated Learning for Chest X-Ray Classification with Malicious Participant Detection and Tamper-Evident Blockchain Audit*.
> 
> In healthcare AI, patient medical images cannot leave hospital firewalls due to strict legal frameworks like HIPAA and GDPR. Federated Learning solves this by allowing hospitals to train a shared global model collaboratively while keeping patient scans strictly local.
> 
> However, federated healthcare networks introduce two compounding challenges:
> 1. Malicious or compromised hospitals can inject Byzantine poisoning attacks that compromise diagnostic safety.
> 2. Real-world hospitals treat naturally distinct patient demographics (e.g., pediatric vs. adult clinics). Classical Byzantine-robust defenses mistake this legitimate clinical specialization for adversarial attacks, severely penalizing honest minority clinics.
> 
> Our research specifically solves this fundamental conflict."

---

### Slide 2: Medical Dataset (PneumoniaMNIST)
> **"To benchmark our framework, we use PneumoniaMNIST from MedMNIST v3, comprising 5,856 pediatric chest X-ray images.**
> 
> We preserve the official benchmark splits: 4,708 training, 524 validation, and 624 untouched test images. The dataset features a realistic 74% pneumonia class imbalance, mirroring acute respiratory intake cohorts.
> 
> In pediatric clinical triage, missing an active pneumonia infection can be fatal. Therefore, our clinical evaluation prioritizes **Sensitivity and Specificity** rather than looking solely at raw accuracy."

---

### Slide 3: Four-Tier System Architecture
> **"Our solution operates across four tightly integrated tiers:**
> 
> - **Tier 1 (Hospital Data Layer):** 10 simulated clinical hospital partitions holding private scans.
> - **Tier 2 (Federated Learning Engine):** Powered by Flower, training a lightweight 420,000-parameter SmallCNN across 20 communication rounds, sharing only parameter updates ($\Delta w$).
> - **Tier 3 (Defense & Consensus Subsystem):** Our 5-factor feature extractor screens client updates, applies our Symmetrical Clinical Trade-Off Shield, and assigns a tri-state classification: Normal, Suspicious, or Malicious.
> - **Tier 4 (Blockchain Audit Layer):** An immutable permissioned JSONL ledger anchoring every round, hash, and detector flag via SHA-256 cryptographic block chaining."

---

### Slide 4: 10-Hospital IID Federated Learning Baseline
> **"In Phase 3, we evaluated an honest 10-hospital IID federation against a centralized baseline.**
> 
> Our federated model achieved **86.38% test accuracy**, outperforming centralized training (81.25%) by **+5.13%**. Even more dramatically, Normal specificity surged from **50.43% to 70.94% (+20.51%)**.
> 
> Frequent federated averaging acts as an implicit regularizer across diverse client batches, preventing the model from overfitting to the dominant pneumonia class. Crucially, zero patient images ever leave hospital nodes."

---

### Slide 5: Clinical Heterogeneity & Non-IID Skew
> **"In Phase 4, we modeled real-world demographic specialization using constrained Dirichlet distributions ($\alpha \in \{1.0, 0.5, 0.1\}$).**
> 
> Under moderate specialization ($\alpha=0.5$), some hospitals treat predominantly pneumonia while others treat predominantly normal scans.
> 
> **Here is our key empirical discovery:** Under severe skew ($\alpha=0.1$), specialized clinics naturally exhibit **negative cosine alignment ($\cos \theta$ down to $-0.8411$)** with the global model. Classical anomaly detectors assume honest gradients always align positively; consequently, they falsely disqualify honest specialized clinics. Our project is designed specifically to prevent this failure."

---

### Slide 6: Adversarial Attack Suite (Phase 5)
> **"In Phase 5, we benchmarked four attack families against unweighted FedAvg:**
> 
> 1. Targeted Label Flipping ($1 \to 0$: pneumonia to normal).
> 2. Update Inversion ($\gamma = -1.0$).
> 3. Update Scaling ($\gamma = +10.0$).
> 4. Backdoor Watermark Trigger Injection.
> 
> When two hospitals flip labels, missed pneumonia cases surge from 7 to 60—a **+757% increase in life-threatening false negatives**. Furthermore, clinical non-IID skew amplifies backdoor Attack Success Rate by **32.5 times** ($0.26\% \to 8.46\%$), proving that unweighted federated learning is fundamentally unsafe for clinical deployment."

---

### Slide 7: Non-IID-Aware Detector & Clinical Shield (Phase 6)
> **"To defend the federation, Phase 6 introduced our multi-factor detector, evaluated out-of-sample on 1,800 gradient vectors.**
> 
> The detector achieves **0.9814 AUROC** and **0.8801 AUPRC**. Crucially, it achieves a **0.00% False Malicious Rate** across all Dirichlet skew levels.
> 
> This is enabled by our **Symmetrical Clinical Trade-Off Shield**: when an update improves performance on one medical class while slightly degrading another, it is verified as a legitimate specialized clinic and protected from false malicious categorization. In our audit, **100% of negative cosine events (199 / 199)** were successfully protected."

---

### Slide 8: Byzantine-Robust Aggregation Comparison (Phase 7)
> **"In Phase 7, we evaluated four aggregators across 72 primary runs (Seeds 211 and 212). This yielded three major empirical findings:**
> 
> 1. **Tri-State Soft Discounting ($\lambda=0.50$) Rescues Specificity:** Under non-IID label flipping, it achieves **58.98% specificity**, outperforming standard FedAvg by +28.21% and Multi-Krum by +33.98%.
> 2. **Multi-Krum's Minority Rejection Failure:** In honest non-IID conditions ($\alpha=0.5$), Multi-Krum collapsed Normal specificity down to **23.08%**. Telemetry proves Multi-Krum rejected the primary normal-treating hospital in **90% to 95% of rounds** because its updates had the largest Euclidean distance!
> 3. **The BatchNorm Poisoning Blind Spot:** We discovered that an attacker poisoning running statistics with $10^6$ evades weight detectors in 100% of rounds, causing **100% universal model collapse** across all classical aggregators."

---

### Slide 9: Cryptographic Blockchain Audit Verification
> **"In Tier 4, we implemented a permissioned blockchain audit layer.**
> 
> As demonstrated live on the dashboard, our verification script cryptographically validates all **2,134 blocks** from Genesis (#0) to Tip (#2133) with zero broken links.
> 
> Every round, model hash, and detector flag is anchored chronologically. If a model update causes patient harm, the ledger provides immutable proof for FDA AI/ML post-market surveillance and GDPR traceability, with zero patient data exposed."

---

### Slide 10: Final Conclusions & Academic Roadmap
> **"In conclusion, our minor project has achieved:**
> - A complete 8-phase experimental framework.
> - Empirical validation of Tri-State Soft Discounting for heterogeneous medical federations.
> - Identification of the geometric minority rejection failure in Multi-Krum and the BatchNorm buffer vulnerability.
> - A 19-section academic paper draft (`docs/RESEARCH_PAPER_DRAFT.md`) and comprehensive literature review.
> - 89 out of 89 automated PyTest tests passing with 100% success rate.
> 
> In future Phase 9 work, we plan to implement coordinate-wise median buffer aggregation to eliminate the BatchNorm vulnerability and expand to $N \ge 5$ seeds.
> 
> Thank you for your time, and I invite your questions."

---

## ❓ Frequently Asked Faculty Questions & Model Answers

### Q1: Why did you use PneumoniaMNIST instead of full-resolution hospital DICOM scans?
> **Answer:** "PneumoniaMNIST is an established medical benchmark from MedMNIST v3 derived from actual clinical pediatric radiographs. Using $28 \times 28$ grayscale images allows us to simulate an entire 10-hospital federated network over 20 rounds across dozens of adversarial configurations within academic computational limits. Full-resolution 1024x1024 DICOM scans would require multi-node cluster clusters for training without altering the mathematical dynamics of gradient poisoning, Dirichlet skew, or Byzantine aggregation."

### Q2: Why did Federated Learning achieve higher accuracy (86.38%) than Centralized learning (81.25%)?
> **Answer:** "In our centralized baseline, SmallCNN achieved 99.74% pneumonia recall but only 50.43% specificity due to the 74% class imbalance. In federated learning, each client optimizes locally and their updates are averaged centrally. This distributed consensus acts as an implicit variance regularizer, preventing the optimizer from overfitting to majority pneumonia features and lifting Normal specificity to 70.94% (+20.51%)."

### Q3: How does your Clinical Trade-Off Shield distinguish an honest specialized hospital from an attacker?
> **Answer:** "An attacker performing label flipping pushes updates that degrade classification performance across the board or cause diagnostic trade-offs that do not correspond to medical sub-features. Our detector measures the counterfactual validation loss on held-out proxies separately for Normal and Pneumonia. If an update improves loss on one class while experiencing a minor trade-off on another, it demonstrates genuine feature specialization and is classified as `SUSPICIOUS` for soft discounting rather than `MALICIOUS` for quarantine."

### Q4: Why did Multi-Krum collapse Normal specificity to 23.08% under honest non-IID conditions?
> **Answer:** "Multi-Krum computes pairwise Euclidean distances between client updates and selects the updates with minimum distance sums to the consensus. Under Dirichlet $\alpha=0.5$, 9 out of 10 hospitals treat mostly pneumonia, while one hospital (Client 2 in Seed 211, Client 3 in Seed 212) treats 98% Normal cases. Because that hospital's updates point in a different direction, its Euclidean distance to all peers is the largest in the network. Multi-Krum discarded that client in 90%–95% of rounds, causing the global model to forget how to identify normal lungs."

### Q5: What is the BatchNorm buffer poisoning vulnerability and how can it be fixed?
> **Answer:** "PyTorch's `nn.BatchNorm2d` maintains non-trainable running mean and variance buffers that update during the forward pass. Classical Byzantine defenses (and all parameter-only anomaly detectors) examine only trainable weight tensors ($\Delta w$), while PyTorch's default federated strategies average buffer tensors linearly. An attacker injecting $10^6$ into buffers bypasses weight detectors completely, corrupting the global running stats to $\approx 10^5$ and causing 100% single-class collapse. In Phase 9, we plan to implement coordinate-wise median buffer aggregation to cure this vulnerability."

### Q6: Why do you report results on only two seeds (211 and 212)?
> **Answer:** "Phase 7 evaluated 72 primary simulations across 4 aggregators and 9 conditions using two independent random seeds. We explicitly disclose this as an exploratory benchmark ($N=2$) and refrain from making claims of formal hypothesis-tested statistical significance. For formal journal publication, expanding to $N \ge 5$ seeds is outlined as our immediate Phase 9 objective."

### Q7: Why use a blockchain instead of a standard PostgreSQL database?
> **Answer:** "In a multi-institution consortium with competing hospital networks, no single hospital or central authority should be trusted to modify past audit logs unilaterally. A standard database allows administrators to alter or delete logs retrospectively. Our permissioned blockchain utilizes cryptographic SHA-256 block hash chaining, where any modified byte invalidates all subsequent block hashes, providing mathematical proof of audit trail integrity for FDA and GDPR regulatory compliance."
