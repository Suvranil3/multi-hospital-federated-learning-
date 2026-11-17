# Academic Paper Reproducibility Checklist & Artifact Verification

**Project**: Multi-Hospital Federated Learning for Chest X-Ray Classification with Malicious Participant Detection and Blockchain Audit  
**Document**: `docs/PAPER_REPRODUCIBILITY_CHECKLIST.md`  
**Date**: November 17, 2026  
**Status**: Authoritative Experimental Verification & Audit Standards  

---

## 1. Dataset & Preprocessing Specifications

- **Dataset Source**: PneumoniaMNIST benchmark from the official MedMNIST v2 collection ([Yang et al., 2023](https://doi.org/10.1038/s41597-022-01721-8)), derived from pediatric chest radiographs curated by Kermany et al. (*Cell*, 2018).
- **Local Raw File**: [`data/raw/pneumoniamnist.npz`](file:///home/suvranil/project/data/raw/pneumoniamnist.npz)
- **File SHA-256**: `e1792d3f03751cb101e99f19a63b3c1941436c988665f47853417b05be250cd8`
- **Total Images**: 5,856 verified anterior-posterior radiographs.
- **Resolution**: $28 \times 28$ pixels, single-channel grayscale (`shape: (1, 28, 28)`).
- **Official Benchmark Splits**:
  - **Training Split**: $4,708$ images ($1,214$ Normal [25.79%], $3,494$ Pneumonia [74.21%]).
  - **Validation Split**: $524$ images ($135$ Normal [25.76%], $389$ Pneumonia [74.24%]).
  - **Untouched Benchmark Test Split**: $N = 624$ images ($234$ Normal [37.50%], $390$ Pneumonia [62.50%]).
- **Test Set Isolation**: The 624-sample benchmark test split was strictly quarantined and used exclusively for post-hoc final evaluation. It was never accessed during local client training, never used for server-side proxy validation, and never influenced detector threshold calibration or aggregation weights.
- **Normalization Standard**: Standardized using empirical training set statistics:
  - Empirical Normalized Mean: `0.57192`
  - Empirical Normalized Standard Deviation: `0.16835`
  - Standardization Equation: $\hat{x} = \frac{x - 0.57192}{0.16835}$

---

## 2. Model Architecture & Parameter Index Partitioning

- **Neural Network Class**: `SmallCNN` defined in [`federated_learning/model.py`](file:///home/suvranil/project/federated_learning/model.py)
- **Total Parameters**: 420,802 parameters (1.7 MB uncompressed PyTorch checkpoint size).
- **Layer Partitioning**:
  - `conv1`: `Conv2d(1, 32, kernel_size=3, padding=1)` $\to$ `BatchNorm2d(32)` $\to$ `ReLU()` $\to$ `MaxPool2d(2, 2)`
  - `conv2`: `Conv2d(32, 64, kernel_size=3, padding=1)` $\to$ `BatchNorm2d(64)` $\to$ `ReLU()` $\to$ `MaxPool2d(2, 2)`
  - `fc1`: `Flatten()` $\to$ `Linear(3136, 128)` $\to$ `ReLU()` $\to$ `Dropout(p=0.5)`
  - `fc2`: `Linear(128, 2)`
- **Tensor Index Partitioning**:
  - **Trainable Parameters (`TRAINABLE_INDICES`)**: Indices 0, 1, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17 (12 weight and bias tensors, 420,610 parameters).
  - **BatchNorm Floating-Point Buffers (`BN_FLOAT_INDICES`)**: Indices 2, 3, 6, 7 (`conv1.bn.running_mean`, `conv1.bn.running_var`, `conv2.bn.running_mean`, `conv2.bn.running_var`, 192 float32 parameters).
  - **BatchNorm Integer Buffers (`BN_INT_INDICES`)**: Indices 12, 13 (`conv1.bn.num_batches_tracked`, `conv2.bn.num_batches_tracked`, 2 int64 step counters).

---

## 3. Client Federation & Partitioning Methodology

- **Number of Clients**: $K = 10$ simulated client nodes.
- **Participation Rate**: Full participation ($C = 1.0$, all 10 clients participate in every round).
- **Communication Rounds**: $T = 20$ global rounds per experiment.
- **Local Epochs**: $E = 1$ epoch per round.
- **Batch Size**: $B = 32$.
- **Optimizer**: Adam optimizer with learning rate $\eta = 0.001$, weight decay $\lambda_{\text{wd}} = 0.0001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.
- **Loss Function**: PyTorch `CrossEntropyLoss()` computed over raw logits.
- **Non-IID Partitioning Method**: Symmetric Dirichlet distribution $\text{Dir}(\alpha)$ implemented in [`federated_learning/dataset.py`](file:///home/suvranil/project/federated_learning/dataset.py):
  - **IID**: Uniform random partition ($\alpha = \infty$).
  - **Mild Skew**: $\alpha = 1.0$.
  - **Moderate Skew**: $\alpha = 0.5$.
  - **Extreme Skew**: $\alpha = 0.1$.
- **Partition Seeds**: Deterministically tied to experiment seed (Seed 211 uses partition seed 211; Seed 212 uses partition seed 212).

---

## 4. Adversarial Attack Configurations

- **Malicious Fraction**: $M = 1$ client out of $K = 10$ ($10\%$ adversarial participant ratio). Malicious participant is deterministically assigned to Client ID `0`.
- **Attack Vector 1 — Targeted Label Flipping (`ATK-LF`)**:
  - Implementation: [`federated_learning/attacks/label_flipping.py`](file:///home/suvranil/project/federated_learning/attacks/label_flipping.py)
  - Target: Flips positive Pneumonia cases ($y=1$) to Normal ($y=0$): $\tilde{y} = 1 - y$ for $y=1$. Normal cases ($y=0$) remain unchanged.
- **Attack Vector 2 — Watermark Backdoor Injection (`ATK-BD`)**:
  - Implementation: [`federated_learning/attacks/backdoor.py`](file:///home/suvranil/project/federated_learning/attacks/backdoor.py)
  - Trigger Pattern: $3 \times 3$ solid white watermark (pixel intensity $1.0$, pixel coordinates rows 24–26, cols 24–26).
  - Target Label: Normal ($y = 0$).
  - Trigger Retention Rate: $\rho = 0.50$ (injected into 50% of local Pneumonia training samples).
  - Evaluation Split: `TriggeredPneumoniaTestDataset` ($N=390$ true Pneumonia test cases with $3 \times 3$ watermark).
- **Attack Vector 3 — Batch Normalization Buffer Poisoning (`ATK-BN`)**:
  - Implementation: [`federated_learning/attacks/batchnorm_poisoning.py`](file:///home/suvranil/project/federated_learning/attacks/batchnorm_poisoning.py)
  - Attack Mechanism: Client executes genuine local training on trainable parameters $\theta_0$, but directly injects an extreme positive outlier magnitude ($1.0 \times 10^6$) into floating-point running statistics (`running_mean` and `running_var`).

---

## 5. Security Screening & Malicious Detector Configuration

- **Detector Implementation**: [`federated_learning/detection/detector.py`](file:///home/suvranil/project/federated_learning/detection/detector.py)
- **Source File SHA-256**: `6576468da037927fda3283a1e2d8402f3b2739d6c45b6fe8ad517900e94a7f07`
- **Detector Status**: Permanently frozen as of Phase 6 closeout (October 31, 2026).
- **Calibrated Decision Thresholds**:
  - Global Modified Z-Score Threshold: $\tau_Z = 3.5$
  - Layer-Wise Modified Z-Score Threshold: $\tau_{Z,\text{layer}} = 3.5$
  - Directional Cosine Similarity Reference Threshold: $\tau_{\cos} = 0.0$ (evaluated against coordinate-wise median reference update)
  - Counterfactual Proxy Normal Loss Delta Threshold: $\tau_{\Delta \mathcal{L}} = +0.50$
  - Counterfactual Proxy Pneumonia Loss Delta Threshold: $\tau_{\Delta \mathcal{L},\text{pneu}} = +0.50$
  - Exponential Moving Average (EMA) Anomaly Factor: $\beta_{\text{EMA}} = 0.70$
  - Composite Anomaly Score Threshold (Suspicious Tier): $\tau_{\text{suspicious}} = 0.35$
  - Composite Anomaly Score Threshold (Malicious Tier): $\tau_{\text{malicious}} = 0.70$
- **Balanced Benchmark Proxy Set**: $N = 200$ images (100 Normal, 100 Pneumonia) extracted from the validation split.
- **Symmetrical Clinical Trade-Off Protection**: If $\cos(\Delta \theta_k, \Delta \theta_{\text{ref}}) < 0$, but counterfactual minority recall improves ($\Delta \text{Recall}_k^{\text{normal}} > 0$) without catastrophic loss spike, the update is shielded from `MALICIOUS` classification and designated as `SUSPICIOUS`.

---

## 6. Aggregation Strategy Configurations

- **Standard FedAvg**: Sample-weighted linear parameter and buffer averaging ($w_k = n_k / N$).
- **Multi-Krum ($f=1, q=7, m=9$)**:
  - Implementation: [`federated_learning/aggregation/krum.py`](file:///home/suvranil/project/federated_learning/aggregation/krum.py)
  - Evaluates pairwise Euclidean distances over flattened trainable parameter vectors (`TRAINABLE_INDICES`).
  - Score for client $i$: $S(i) = \sum_{j \in \mathcal{N}_7(i)} \| \theta_i - \theta_j \|_2^2$, where $|\mathcal{N}_7(i)| = K - f - 2 = 10 - 1 - 2 = 7$.
  - Selects $m = K - f = 10 - 1 = 9$ lowest-scoring clients and averages them uniformly.
- **Tri-State Telemetry Monitor**: Identical to FedAvg with full telemetry and blockchain logging.
- **Soft Tri-State Trust Discounting ($\lambda = 0.50$)**:
  - Implementation: [`federated_learning/aggregation/tristate.py`](file:///home/suvranil/project/federated_learning/aggregation/tristate.py)
  - Raw weights: $w_k = 1.0$ if `NORMAL`, $w_k = 0.50$ if `SUSPICIOUS`, $w_k = 0.0$ if `MALICIOUS`.
  - Normalized weights: $\tilde{w}_k = \frac{w_k n_k}{\sum_j w_j n_j}$.

---

## 7. Experiment Registry, Manifest & Checkpoint Provenance

- **Primary Matrix Manifest**: [`results/phase7_robust_aggregation/PHASE7_REDUCED_EXPERIMENT_MATRIX_96.json`](file:///home/suvranil/project/results/phase7_robust_aggregation/PHASE7_REDUCED_EXPERIMENT_MATRIX_96.json) (SHA-256: `b640bd74e47eac8bea19afcdf21193b8e478275e1e17b2c7db27da884ecb07b1`, 72 primary experiment specifications).
- **Primary Execution Registry**: [`results/phase7_robust_aggregation/execution_registry_reduced_96.json`](file:///home/suvranil/project/results/phase7_robust_aggregation/execution_registry_reduced_96.json) (SHA-256: `239f7f0088de4530fd8373791189972a764960d1ec45e188e5b890afed5a9d88`, 72 completed simulation records).
- **Historical Execution Registry (Preserved)**: [`results/phase7_robust_aggregation/execution_registry.json`](file:///home/suvranil/project/results/phase7_robust_aggregation/execution_registry.json) (43 records, preserving 24 supplementary historical experiments).
- **Physical Model Checkpoints**: Exactly 144 `.pt` files (72 `best_*.pt` and 72 `final_*.pt`) located in [`results/phase7_robust_aggregation/checkpoints/`](file:///home/suvranil/project/results/phase7_robust_aggregation/checkpoints/).
- **Evaluation Reports**: Exactly 72 `*_results.json` files located in [`results/phase7_robust_aggregation/evaluation/`](file:///home/suvranil/project/results/phase7_robust_aggregation/evaluation/).
- **Granular Telemetry Logs**: Exactly 72 `*_features.jsonl` files located in [`results/phase7_robust_aggregation/instrumentation/`](file:///home/suvranil/project/results/phase7_robust_aggregation/instrumentation/).
- **Batch 2 Recovery Audit Documentation**: [`docs/PHASE7_BATCH2_RECOVERY_AUDIT.md`](file:///home/suvranil/project/docs/PHASE7_BATCH2_RECOVERY_AUDIT.md).

---

## 8. Blockchain Cryptographic Audit Verification

- **Ledger File**: [`results/blockchain_audit/audit_ledger.jsonl`](file:///home/suvranil/project/results/blockchain_audit/audit_ledger.jsonl)
- **Genesis Block**: Block #0 (Index 0, Previous Hash: `0000000000000000000000000000000000000000000000000000000000000000`).
- **Final Tip Block**: Block #2133 (Previous Hash: `ff354a107ea82fc606b0d91d6402444855bc45722f9ca29e001b7dfb36bb57ba`).
- **Total Blocks Verified**: Exactly 2,134 blocks.
- **Verification Tool**: Standalone verification via [`blockchain_audit.verification.verify_ledger()`](file:///home/suvranil/project/blockchain_audit/verification.py).
- **Verification Status**: **`is_valid = True`, 0 broken backwards hash links, 0 genesis discrepancies, 100% cryptographic match against on-disk artifact digests**.

---

## 9. Software Dependencies & Runtime Environment

- **Operating System**: Linux Ubuntu (kernel `7.0.0-42-generic-x86_64`, `glibc 2.43`).
- **Python Version**: `3.14.4` (in dedicated `.venv`).
- **PyTorch Version**: `2.14.1+cu130` (CPU mode enforced for Phase 7 execution safety).
- **Flower Federated Framework**: `flwr 1.39.0`.
- **MedMNIST Collection**: `medmnist 3.0.2`.
- **NumPy Version**: `numpy 2.5.3`.
- **Runner Script**: [`experiments/run_phase7_memory_safe.py`](file:///home/suvranil/project/experiments/run_phase7_memory_safe.py) (SHA-256: `7c97f3e0abb98b3f0080f998859b99cf200c5b9246c92f8b0082991111f5b614`).
