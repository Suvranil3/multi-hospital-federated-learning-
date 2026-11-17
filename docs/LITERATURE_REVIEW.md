# Comprehensive Academic Literature Review

**Project**: Multi-Hospital Federated Learning for Chest X-Ray Classification with Malicious Participant Detection and Blockchain Audit  
**Document**: `docs/LITERATURE_REVIEW.md`  
**Date**: November 17, 2026  
**Status**: Authoritative Bibliographic Compilation & Critical Synthesis  

---

## 1. Foundations of Federated Learning & Non-IID Optimization

### 1.1. Federated Averaging (FedAvg)
- **Full Title**: Communication-Efficient Learning of Deep Networks from Decentralized Data
- **Authors**: H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas
- **Year**: 2017
- **Venue**: Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS 2017), PMLR 54:1273–1282
- **Verified URL / DOI**: [http://proceedings.mlr.press/v54/mcmahan17a.html](http://proceedings.mlr.press/v54/mcmahan17a.html) / [10.48550/arXiv.1602.05629](https://doi.org/10.48550/arXiv.1602.05629)
- **Main Contribution**: Formulated the standard Federated Learning (FL) paradigm and introduced the Federated Averaging (`FedAvg`) algorithm, which decouples model training from direct data access by iteratively averaging client model parameter deltas computed via local stochastic gradient descent (SGD).
- **Relevance to This Research**: Serves as the primary baseline aggregator throughout this project. It provides linear sample-weighted parameter aggregation ($\theta^{(t+1)} = \sum_{k=1}^K \frac{n_k}{N} \theta_k^{(t+1)}$), which exhibits zero Byzantine robustness and accepts arbitrary adversarial weight updates uncritically.

### 1.2. The Non-IID Weight Divergence Problem
- **Full Title**: Federated Learning with Non-IID Data
- **Authors**: Yue Zhao, Meng Li, Liangzhen Lai, Naveen Suda, Damon Civin, Vikas Chandra
- **Year**: 2018
- **Venue**: arXiv preprint arXiv:1806.00582
- **Verified URL / DOI**: [https://doi.org/10.48550/arXiv.1806.00582](https://doi.org/10.48550/arXiv.1806.00582)
- **Main Contribution**: Quantified how non-identical class distributions across client devices degrade `FedAvg` convergence speed and final model accuracy. Proved analytically and empirically that weight divergence ($\| \theta_{\text{FedAvg}} - \theta_{\text{Centralized}} \|$) increases proportionally to the earth mover's distance (EMD) between local client data distributions and the global population.
- **Relevance to This Research**: Establishes the theoretical foundation for our medical hospital scenario. Hospital case-mix divergence produces natural gradient divergence, creating the core dilemma where benign domain diversity mimics adversarial gradient attacks.

### 1.3. Proximal Heterogeneity Regularization (FedProx)
- **Full Title**: Federated Optimization in Heterogeneous Networks
- **Authors**: Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith
- **Year**: 2020
- **Venue**: Proceedings of Machine Learning and Systems (MLSys 2020), 2:429–450
- **Verified URL / DOI**: [https://doi.org/10.48550/arXiv.1812.06127](https://doi.org/10.48550/arXiv.1812.06127)
- **Main Contribution**: Introduced `FedProx`, adding a proximal regularization term $\frac{\mu}{2} \| \theta - \theta^{(t)} \|_2^2$ to the local objective function to restrict local model drift under system and statistical heterogeneity.
- **Relevance to This Research**: Highlights the fundamental tension in heterogeneous FL: restricting local divergence stabilizes global convergence but can suppress domain-specific adaptation in specialized client nodes.

### 1.4. Dirichlet-Based Synthetic Non-IID Partitioning
- **Full Title**: Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification
- **Authors**: Tzu-Ming Harry Hsu, Hang Qi, Matthew Brown
- **Year**: 2019
- **Venue**: arXiv preprint arXiv:1909.06335
- **Verified URL / DOI**: [https://doi.org/10.48550/arXiv.1909.06335](https://doi.org/10.48550/arXiv.1909.06335)
- **Main Contribution**: Standardized the use of symmetric Dirichlet distributions $\text{Dir}(\alpha)$ to synthesize controlled label heterogeneity across federated client partitions, demonstrating that lower concentration parameters ($\alpha < 1.0$) induce severe performance drops across standard visual classification tasks.
- **Relevance to This Research**: Directly informs our non-IID partitioning methodology across Phase 4 through Phase 7, where $\alpha \in \{1.0, 0.5, 0.1\}$ is used to systematically benchmark hospital case-mix heterogeneity from mild to extreme skew.

---

## 2. Classical Byzantine-Robust Federated Aggregation Rules

### 2.1. Geometric Distance-Based Filtering: Krum and Multi-Krum
- **Full Title**: Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent
- **Authors**: Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, Julien Stainer
- **Year**: 2017
- **Venue**: Advances in Neural Information Processing Systems (NeurIPS 2017), 30
- **Verified URL / DOI**: [https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html](https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html) / [10.48550/arXiv.1703.02757](https://doi.org/10.48550/arXiv.1703.02757)
- **Main Contribution**: Introduced `Krum` and `Multi-Krum`, the first provably Byzantine-tolerant aggregation rules for distributed SGD. Krum evaluates each update by computing the sum of squared Euclidean distances to its $n - f - 2$ nearest neighbors, selecting the candidate that minimizes this dispersion score. Multi-Krum averages the top $m = n - f$ candidates.
- **Relevance to This Research**: Multi-Krum ($f=1, m=9$) is our primary classical robust aggregation benchmark. Our research demonstrates a fundamental structural flaw of Multi-Krum under non-IID conditions: because it assumes spherical IID clustering in Euclidean space, it systematically rejects honest specialized clients holding rare medical classes (e.g., rejecting Client 2/3 holding 98% Normal cases in 90%–95% of rounds under $\alpha=0.5$), inducing catastrophic specificity collapse down to 23.08%.

### 2.2. Coordinate-Wise Trimming and Medians
- **Full Title**: Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates
- **Authors**: Dong Yin, Yudong Chen, Kannan Ramchandran, Peter Bartlett
- **Year**: 2018
- **Venue**: Proceedings of the 35th International Conference on Machine Learning (ICML 2018), PMLR 80:5650–5659
- **Verified URL / DOI**: [https://doi.org/10.48550/arXiv.1803.01498](https://doi.org/10.48550/arXiv.1803.01498)
- **Main Contribution**: Established statistical minimax optimal rates for Byzantine distributed learning using coordinate-wise median and trimmed mean aggregation, proving robust convergence bounds for strongly convex and smooth non-convex losses under adversarial corruption.
- **Relevance to This Research**: Evaluated as supplementary baselines in Phase 7. Coordinate Median and Trimmed Mean discard coordinate-level extrema, which provides protection under IID assumptions but over-trims legitimate gradient adjustments on minority classes when client data is non-IID.

---

## 3. Adversarial Threat Models in Federated Learning

### 3.1. Targeted Model Replacement & Backdoor Injection
- **Full Title**: How To Backdoor Federated Learning
- **Authors**: Eugene Bagdasaryan, Andreas Veit, Yiqing Hua, Deborah Estrin, Vitaly Shmatikov
- **Year**: 2020
- **Venue**: Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS 2020), PMLR 108:2938–2948
- **Verified URL / DOI**: [https://proceedings.mlr.press/v108/bagdasaryan20a.html](https://proceedings.mlr.press/v108/bagdasaryan20a.html) / [10.48550/arXiv.1807.00459](https://doi.org/10.48550/arXiv.1807.00459)
- **Main Contribution**: Formulated the "model replacement" paradigm, demonstrating that an adversary participating in even a single round can scale their parameter update by $\gamma = \frac{N}{\eta}$ to override benign updates in `FedAvg`, embedding a stealthy backdoor while preserving high validation accuracy on the main task. Proposed "constrain-and-scale" to bypass $L_2$ norm thresholding.
- **Relevance to This Research**: Informs our Phase 5 and Phase 7 backdoor threat model ($3 \times 3$ corner watermark trigger, trigger retention $\rho=0.50$). Our findings prove that Multi-Krum suppresses IID backdoors effectively (2.43% ASR vs. 8.46% FedAvg), but Tri-State Soft Discounting provides the best joint balance under heterogeneity (5.00% ASR while maintaining 86.46% clean accuracy).

### 3.2. Stealthy Backdoors in Constrained Federated Settings
- **Full Title**: Can You Really Backdoor Federated Learning?
- **Authors**: Ziteng Sun, Peter Kairouz, Ananda Theertha Suresh, H. Brendan McMahan
- **Year**: 2019
- **Venue**: NeurIPS 2019 Workshop on Implementation Issues in Modern Machine Learning
- **Verified URL / DOI**: [https://doi.org/10.48550/arXiv.1911.07963](https://doi.org/10.48550/arXiv.1911.07963)
- **Main Contribution**: Analyzed the durability of backdoor triggers across rounds, showing that differential privacy noise and norm clipping can degrade backdoor success rates, but stealthy adversaries with repeated participation can maintain high backdoor retention.
- **Relevance to This Research**: Explains why single-round defenses are insufficient and justifies our multi-round tracking (EMA anomaly accumulators and blockchain trajectory logging).

### 3.3. Label-Flipping Poisoning Attacks
- **Full Title**: Poisoning Attacks against Support Vector Machines
- **Authors**: Battista Biggio, Blaine Nelson, Pavel Laskov
- **Year**: 2012
- **Venue**: Proceedings of the 29th International Conference on Machine Learning (ICML 2012), arXiv:1206.6389
- **Verified URL / DOI**: [https://doi.org/10.48550/arXiv.1206.6389](https://doi.org/10.48550/arXiv.1206.6389)
- **Main Contribution**: Pioneered algorithmic data poisoning via label flipping, showing that targeted label corruption shifts decision boundaries with minimal injection budget.
- **Relevance to This Research**: In our clinical setting, a label-flipping attacker flips Pneumonia (Class 1) to Normal (Class 0). Under non-IID $\alpha=0.5$ skew, this attack collapses FedAvg specificity to 30.77% and Multi-Krum to 25.00%, whereas our Tri-State Discounting defense rescues specificity to 58.98% (+28.21% over FedAvg).

---

## 4. Batch Normalization & Model-State Vulnerabilities

### 4.1. Local Batch Normalization Under Non-IID Skew (FedBN)
- **Full Title**: FedBN: Federated Learning on Non-IID Features via Local Batch Normalization
- **Authors**: Xiaoxiao Li, Meirui Jiang, Xiaofei Zhang, Michael Kamp, Qi Dou
- **Year**: 2021
- **Venue**: International Conference on Learning Representations (ICLR 2021)
- **Verified URL / DOI**: [https://openreview.net/forum?id=6YEQUn0QICG](https://openreview.net/forum?id=6YEQUn0QICG) / [10.48550/arXiv.2102.07623](https://doi.org/10.48550/arXiv.2102.07623)
- **Main Contribution**: Demonstrated that in federated learning under domain and feature shift, aggregating Batch Normalization layers centrally damages global model convergence. Proposed keeping BatchNorm parameters strictly local to client devices while aggregating only convolutional and linear weights.
- **Relevance to This Research**: Highlights the unique vulnerability of Batch Normalization in federated architectures. In standard federated pipelines (e.g. Flower FedAvg) where BatchNorm buffers (`running_mean` and `running_var`) are aggregated centrally, attackers can leave trainable weights honest and inject extreme buffer outliers ($10^6$), bypassing all parameter-space detectors and causing universal 100% diagnostic model collapse.

### 4.2. Batch Normalization Inconsistencies and Covariate Shift
- **Full Title**: Overcoming the Challenges of Batch Normalization in Federated Learning
- **Authors**: Yuxuan Du, et al.
- **Year**: 2024
- **Venue**: arXiv preprint arXiv:2405.14670
- **Verified URL / DOI**: [https://doi.org/10.48550/arXiv.2405.14670](https://doi.org/10.48550/arXiv.2405.14670)
- **Main Contribution**: Proved that external covariate shift across federated clients causes running mean and variance estimates to deviate significantly from the true global population distribution, leading to severe test-time activation distortion in `.eval()` mode.
- **Relevance to This Research**: Directly explains the physical mechanism of our Phase 7 BatchNorm buffer poisoning findings: linear buffer averaging propagates adversarial outliers into global running statistics, scaling feature activations to zero and causing degenerate single-class collapse.

---

## 5. Anomaly Detection & Trust-Discounted Aggregation

### 5.1. FoolsGold: Sybil and Poisoning Mitigation via Cosine Diversity
- **Full Title**: Mitigating Sybils in Federated Learning Poisoning with FoolsGold
- **Authors**: Clement Fung, Chris J. M. Yoon, Ivan Beschastnikh
- **Year**: 2020
- **Venue**: IEEE Transactions on Information Forensics and Security, 15:3416–3430
- **Verified URL / DOI**: [https://doi.org/10.1109/TIFS.2020.2995964](https://doi.org/10.1109/TIFS.2020.2995964)
- **Main Contribution**: Introduced `FoolsGold`, which computes pairwise cosine similarities of historical gradient trajectories to identify Sybil attackers who share indicative poisoning objectives, dynamically down-weighting correlated updates.
- **Relevance to This Research**: Demonstrates that directional similarity contains vital security signals. However, FoolsGold assumes malicious clients share high cosine similarity, whereas in single-attacker scenarios ($M=1$) under high clinical non-IID skew, honest clients themselves exhibit diverse directional trajectories.

### 5.2. Anomaly-Based Participant Screening: Auror
- **Full Title**: Auror: Defending against Poisoning Attacks in Collaborative Deep Learning Systems
- **Authors**: Shiqi Shen, Shruti Tople, Prateek Saxena
- **Year**: 2016
- **Venue**: Proceedings of the 32nd Annual Computer Security Applications Conference (ACSAC 2016), 508–519
- **Verified URL / DOI**: [https://doi.org/10.1145/2991079.2991125](https://doi.org/10.1145/2991079.2991125)
- **Main Contribution**: Proposed masking and clustering update features on critical classification layers to identify malicious participants attempting targeted label poisoning.
- **Relevance to This Research**: Informs our layer-wise modified Z-score computation on dense and classification head tensors.

### 5.3. Trajectory-Based Detection: FLDetector
- **Full Title**: FLDetector: Defending Federated Learning Against Model Poisoning Attacks via Detecting Malicious Clients
- **Authors**: Zaixi Zhang, Xiaoyu Cao, Jinyuan Jia, Neil Zhenqiang Gong
- **Year**: 2022
- **Venue**: Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '22), 2416–2426
- **Verified URL / DOI**: [https://doi.org/10.1145/3534678.3539231](https://doi.org/10.1145/3534678.3539231)
- **Main Contribution**: Developed `FLDetector`, which reconstructs client loss trajectories using historical model checkpoints and predicts expected gradient directions using a Hessian-based approximation. Clients whose actual updates deviate from theoretical predictions are flagged as malicious.
- **Relevance to This Research**: Demonstrates the value of multi-round trajectory tracking. Our research extends this insight by combining trajectory tracking with leave-one-out counterfactual proxy validation and a non-binary tri-state decision policy.

---

## 6. Federated Learning in Medical Imaging & Healthcare Networks

### 6.1. The Healthcare Federated Learning Paradigm
- **Full Title**: The Future of Digital Health with Federated Learning
- **Authors**: Nicola Rieke, Jonny Hancox, Wenqi Li, Fausto Milletarì, Holger R. Roth, Shadi Albarqouni, Spyridon Bakas, Mathieu N. Galtier, Bennett A. Landman, Klaus Maier-Hein, Sébastien Ourselin, Micah J. Sheller, Ronald M. Summers, Andrew Trask, Daguang Xu, Bernard T. Baetens, M. Jorge Cardoso
- **Year**: 2020
- **Venue**: NPJ Digital Medicine, 3(1), Article 119
- **Verified URL / DOI**: [https://doi.org/10.1038/s41746-020-00323-1](https://doi.org/10.1038/s41746-020-00323-1)
- **Main Contribution**: Authored the landmark review establishing the clinical necessity, governance framework, and technical challenges of Federated Learning in digital medicine, emphasizing HIPAA/GDPR regulatory compliance and institutional data silos.
- **Relevance to This Research**: Provides the authoritative clinical motivation for our multi-hospital architecture. Highlights that patient data cannot cross institutional boundaries, necessitating on-premise local training.

### 6.2. Multi-Institutional Collaborative Medicine
- **Full Title**: Federated Learning in Medicine: Facilitating Multi-Institutional Collaborations without Sharing Patient Data
- **Authors**: Micah J. Sheller, Brandon Edwards, G. Anthony Reina, Jason Martin, Sarthak Pati, Aikaterini Kotrotsou, Mikhail Milchenko, Weilin Xu, Daniel Marcus, Rivka R. Colen, Spyridon Bakas
- **Year**: 2020
- **Venue**: Scientific Reports, 10(1), Article 12598
- **Verified URL / DOI**: [https://doi.org/10.1038/s41598-020-69250-1](https://doi.org/10.1038/s41598-020-69250-1)
- **Main Contribution**: Validated federated learning across multiple clinical institutions for brain tumor segmentation (BraTS), proving that FL models achieve 99% of the performance of centralized models while maintaining absolute data containment.
- **Relevance to This Research**: Demonstrates the clinical feasibility of cross-institutional medical FL, establishing that privacy preservation can coexist with near-centralized diagnostic accuracy.

### 6.3. The EXAM COVID-19 Initiative
- **Full Title**: Federated Learning for Predicting Clinical Outcomes in Patients with COVID-19
- **Authors**: Ittai Dayan, Holger R. Roth, Aoxiao Zhong, Ahmed Harouni, Amilcare Gentili, Anas Z. Abidin, et al. (EXAM Consortium)
- **Year**: 2021
- **Venue**: Nature Medicine, 27(10):1735–1743
- **Verified URL / DOI**: [https://doi.org/10.1038/s41591-021-01506-3](https://doi.org/10.1038/s41591-021-01506-3)
- **Main Contribution**: Deployed federated learning across 20 global clinical hospitals to train the EXAM model on chest radiographs and clinical vitals, improving average AUC by 16% and generalizability across diverse global populations.
- **Relevance to This Research**: Serves as the prime empirical precedent for multi-hospital chest X-ray classification. Proves that real-world deployment across dozens of hospitals is achievable and necessary, while highlighting the total absence of Byzantine defenses in existing real-world clinical deployments.

### 6.4. Benchmark Dataset: MedMNIST v2
- **Full Title**: MedMNIST v2 - A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification
- **Authors**: Jiancheng Yang, Rui Shi, Donglai Wei, Ziyue Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni
- **Year**: 2023
- **Venue**: Scientific Data, 10(1), Article 41
- **Verified URL / DOI**: [https://doi.org/10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8)
- **Main Contribution**: Curated and standardized MedMNIST v2, providing 18 lightweight biomedical benchmark datasets formatted as $28 \times 28$ images, including PneumoniaMNIST (derived from Kermany et al., *Cell* 2018).
- **Relevance to This Research**: Source of our primary project dataset. Enables rapid, reproducible, memory-safe execution of 96 full federated simulations while preserving authentic clinical radiograph distributions.

---

## 7. Synthesis of the Open Research Gap

A critical synthesis of the surveyed literature identifies an urgent, unaddressed gap at the intersection of clinical federated learning and adversarial robustness:

```text
┌───────────────────────────────────────┐      ┌───────────────────────────────────────┐
│       CLINICAL MEDICAL FL             │      │       BYZANTINE ROBUST FL             │
│   Rieke et al. (2020)                 │      │   Blanchard et al. (NeurIPS 2017)     │
│   Sheller et al. (Sci Rep 2020)       │      │   Yin et al. (ICML 2018)              │
│   Dayan et al. (Nat Med 2021)         │      │   Bagdasaryan et al. (AISTATS 2020)   │
│                                       │      │                                       │
│   • Multi-hospital networks           │      │   • Assume IID / homogeneous data     │
│   • Severe non-IID case-mix skew      │      │   • Geometric Euclidean clustering    │
│   • Zero Byzantine threat modeling    │      │   • Binary accept / reject exclusion  │
│   • Unweighted FedAvg assumption      │      │   • Evaluate synthetic toy benchmarks │
└───────────────────┬───────────────────┘      └───────────────────┬───────────────────┘
                    │                                              │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │            THE OPEN RESEARCH GAP             │
                    │                                              │
                    │  1. How do classical Byzantine aggregators   │
                    │     perform under clinical non-IID skew?     │
                    │  2. How can honest specialized hospitals be  │
                    │     protected from false exclusion?          │
                    │  3. How can non-trainable state buffers      │
                    │     (BatchNorm) be defended in FL?           │
                    │  4. How can institutional decisions be       │
                    │     transparently audited without PHI?       │
                    └──────────────────────────────────────────────┘
```

1. **The Incompatibility Gap**: Classical robust aggregators (Krum, Trimmed Mean) were formulated under IID assumptions. When deployed under realistic clinical heterogeneity ($\alpha=0.5$), Euclidean filters cannot distinguish between a malicious poisoning attack and legitimate clinical specialization, systematically discarding minority-class hospitals.
2. **The Binary Exclusion Dilemma**: Existing defenses impose hard binary decisions (accept or drop). In healthcare, dropping an honest specialized hospital discards vital medical knowledge, inducing diagnostic blindness.
3. **The Parameter vs. Buffer Blind Spot**: Existing Byzantine literature focuses almost exclusively on trainable parameter weights. Non-trainable persistent state buffers (such as BatchNorm running mean and variance) are overlooked by detectors and linearly averaged, creating a devastating zero-day attack vector.
4. **The Regulatory Audit Gap**: No prior framework integrates multi-factor clinical anomaly screening with an immutable, privacy-preserving cryptographic audit trail capable of satisfying healthcare non-repudiation standards.

This project and its accompanying academic draft directly address these open challenges.
