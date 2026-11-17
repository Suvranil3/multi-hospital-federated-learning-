# 🏥 Multi-Hospital Federated Learning: Faculty Demonstration Dashboard

This interactive Streamlit application provides a complete, faculty-ready demonstration interface for the final minor project:
**"Multi-Hospital Federated Learning for Chest X-Ray Classification with Malicious Participant Detection and Tamper-Evident Blockchain Audit"**

---

## 🚀 Quick Start & Launch Instructions

Activate the project virtual environment and launch the Streamlit dashboard:

```bash
# 1. Activate project virtual environment
source .venv/bin/activate

# 2. Launch the Streamlit application
streamlit run dashboard/app.py
```

By default, the dashboard opens in your browser at:
**[http://localhost:8501](http://localhost:8501)**

*Headless / Custom Port Launch:*
```bash
streamlit run dashboard/app.py --server.port 8501 --server.headless true
```

---

## 🧭 Dashboard Operating Modes

The dashboard features four dedicated operating modes selectable from the sidebar:

### 1. 🎓 Guided Faculty Presentation Mode (Default)
Optimized for a structured 10–15 minute defense before faculty examiners:
- **10 Sequential Slides:** Walks through the problem statement, dataset, architecture, IID federated learning, non-IID challenges, attack suite, detection mechanism, robust aggregation, blockchain verification, and conclusions.
- **Embedded Speaker Notes:** Each slide features a dedicated **"Speaker Script"** providing verbatim presentation lines and key talking points to emphasize.
- **Fast Navigation:** Intuitive `Next ➡` and `⬅ Previous` buttons alongside a slide jump selector.

### 2. 🧭 Interactive Project Explorer Mode
Enables examiners to freely inspect any of the 10 core research sections:
1. **Project Overview & Architecture:** HIPAA/GDPR clinical dilemma, 4-tier system architecture, and milestone verification table.
2. **PneumoniaMNIST Dataset Explorer:** Official splits (4,708 train, 524 val, 624 test), 74% class imbalance, and sample X-ray grid.
3. **Centralized CNN Baseline:** SmallCNN architecture (420k params), 81.25% test accuracy, 99.74% pneumonia sensitivity, and confusion matrix.
4. **10-Hospital IID Federated Learning:** Flower virtual engine, 20 rounds, 86.38% test accuracy (+5.13% over centralized), and privacy mechanics.
5. **Non-IID Clinical Heterogeneity:** Dirichlet case-mix modeling ($\alpha \in \{1.0, 0.5, 0.1\}$) and the negative cosine alignment phenomenon ($\cos \theta < 0$).
6. **Adversarial Attack Suite:** Targeted label flipping (+757% missed pneumonia surge), update scaling/inversion, and backdoor watermark triggers (32.5× ASR amplification).
7. **Malicious Participant Detection:** 5-factor feature architecture, 0.9814 AUROC, 0.00% False Malicious Rate across all Dirichlet skews, and Symmetrical Clinical Trade-Off Shield (199/199 events protected).
8. **Byzantine-Robust Aggregation Comparison:** Complete evaluation of 4 aggregators (FedAvg, Multi-Krum, Tri-State Monitor, Tri-State Discount 0.50) across 9 conditions and 2 seeds ($N=2$); rescues Normal specificity to 58.98%; exposes Multi-Krum minority rejection and the BatchNorm buffer poisoning blind spot.
9. **Blockchain Audit Verification:** Cryptographic SHA-256 verification of all 2,134 blocks in `audit_ledger.jsonl`; interactive block height browser.
10. **Final Experimental Scorecard:** Master 72-cell primary scorecard table with interactive filters by condition, strategy, and seed; links to academic research papers.

### 3. 🔍 Technical Artifact Inspector Mode
Provides full scientific transparency by loading raw, unedited JSON logs, checkpoint records, and hash tables:
- `pneumonia_mnist_verification_report.json`
- `medical_centralized_results.json`
- `federated_iid_results.json`
- `phase4_heterogeneity_comparison.json`
- `phase5_attack_comparison.json`
- `detection_results.json` & `calibrated_thresholds.json`
- `execution_registry_reduced_96.json` (72 completed primary cells)
- `audit_ledger.jsonl` verification dictionary

### 4. 📦 Archived Proof-of-Concept Mode
Preserves complete historical evidence of the initial exploratory CIFAR-10 phase (76.94% accuracy) alongside documentation explaining why the project migrated to pediatric chest X-rays (PneumoniaMNIST) to address genuine clinical hospital requirements.

---

## 🔒 Reliability, Safety & Architectural Principles

- **Zero Accidental Training:** The dashboard is purely a demonstration and analytics tool. It contains no execution buttons that trigger expensive training runs or modify model weights.
- **Pure Source of Truth:** All displayed metrics, figures, and tables are loaded directly from verified experimental artifacts in `results/`. No metrics are fabricated or hardcoded.
- **Fast Cached Navigation:** Data loading operations utilize `@st.cache_data` for sub-second page transitions.
- **Defensive Error Handling:** Missing or corrupt files are handled gracefully with informative status messages.

---

## 🧪 Automated Test & Verification Status

- **Unit & Integration Tests:** **89 / 89 Passed (100%)** via PyTest.
- **Blockchain Ledger:** **2,134 Blocks Verified (`is_valid = True`)** via SHA-256 hash chaining.
- **Primary Research Matrix:** **72 / 72 Cells Completed** across Seeds 211 & 212.
- **Supplementary Matrix:** **24 Historical Runs Preserved** across Seeds 101, 102, 103.
