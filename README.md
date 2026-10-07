# Blockchain-Assisted Non-IID-Aware Malicious Participant Detection in Federated Learning

## Final-Year Research Project

### Research Question
> **"Can a non-IID-aware malicious participant detection mechanism reduce false positives while still detecting poisoning attacks in Federated Learning?"**

---

## 1. Project Overview
Federated Learning (FL) enables collaborative model training across decentralized clients while preserving data privacy. However, in adversarial settings, malicious participants can mount poisoning attacks (label flipping, gradient scaling, backdoor injection) to compromise the global model.

Conventional Byzantine-robust aggregation mechanisms (such as Krum, Trimmed Mean, Coordinate-wise Median) rely heavily on distance and consensus heuristics. In non-IID settings where benign edge clients hold naturally skewed, specialized data distributions, their legitimate model updates deviate significantly from the central consensus. Consequently, existing defenses suffer from high False Positive Rates (FPR), unfairly penalizing honest specialized clients.

This research project develops and evaluates a **Non-IID-Aware Malicious Participant Detector** that incorporates:
1. Update deviation ($L_2$ norm)
2. Directional cosine similarity
3. Local validation / performance impact
4. Client historical behavioral trajectory
5. Client data heterogeneity context

The detector classifies client updates into a tri-state decision:
- **Normal**: Full acceptance into global aggregation ($w = 1.0$).
- **Suspicious**: Down-weighted based on anomaly severity ($0 < w < 1.0$).
- **Malicious**: Complete rejection ($w = 0.0$).

Accepted and weighted updates are aggregated into the global model via robust aggregation. An Ethereum-compatible **blockchain ledger** serves strictly as an append-only, tamper-evident audit layer to log cryptographic hashes of model updates, detector classifications, and actions taken—without storing weights or running detection on-chain.

---

## 2. Repository Structure
```
Blockchain_FL_Project/
│
├── README.md                 # Project overview and setup
├── PROJECT_STATUS.md         # Active development phase and status
├── RESEARCH_OBJECTIVE.md     # Research questions, hypotheses, and scope
├── RESEARCH_GAP.md           # Analysis of Byzantine-robust aggregation limits under non-IID
├── ARCHITECTURE.md           # 4-tier system design and security model
├── DECISIONS.md              # Architectural Decision Records (ADRs)
├── CHANGELOG.md              # Chronological change log
├── EXPERIMENT_LOG.md         # Empirical experiment record
├── requirements.txt          # Python dependencies
├── .gitignore                # Version control ignore rules
│
├── data/
│   ├── README.md             # Dataset documentation and partitioning
│   ├── raw/                  # Downloaded raw CIFAR-10 archives
│   ├── processed/            # Serialized client partitions
│   └── verify_dataset.py     # CIFAR-10 verification and stats script
│
├── notebooks/                # Exploratory analysis
├── federated_learning/       # Flower FL clients, server, and strategies
├── detection/                # Non-IID-aware detector modules
├── attacks/                  # Label flipping, model poisoning, backdoor
├── blockchain/               # Smart contract audit trail & Web3 connector
├── experiments/
│   └── config.py             # Central reproducibility configuration & seed 42
├── results/
│   ├── figures/              # Generated visualizations and plots
│   ├── tables/               # Formatted results and evaluation tables
│   └── logs/                 # Run logs and machine-readable JSON summaries
└── tests/
    └── test_data_loading.py  # Pytest suite for data integrity
```

---

## 3. Development Phases
- [x] **Phase 1: Dataset Verification & Repository Setup**
- [ ] **Phase 2: Baseline Centralized CNN**
- [ ] **Phase 3: Basic Federated Learning (FedAvg, IID, 10 Clients)**
- [ ] **Phase 4: Non-IID Federated Learning (Dirichlet $\alpha \in \{1.0, 0.5, 0.1\}$)**
- [ ] **Phase 5: Attack Simulation (Label Flipping, Model Poisoning, Backdoor)**
- [ ] **Phase 6: Multi-Factor Non-IID-Aware Detector**
- [ ] **Phase 7: Tri-State Aggregation & Robust Baselines (Multi-Krum, Trimmed Mean)**
- [ ] **Phase 8: Blockchain Tamper-Evident Audit Trail**
- [ ] **Phase 9: Comprehensive Empirical Evaluation & Ablation**
- [ ] **Phase 10: Research Analysis & Publication Artifacts**

---

## 4. Setup & Quickstart

### Prerequisites
- Python 3.14 (or Python 3.10+)
- Git

### Installation
```bash
# Clone repository
git clone <repo-url>
cd project

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Phase 1 Dataset Verification
```bash
# Run CIFAR-10 verification and generate statistics & plots
python data/verify_dataset.py

# Run unit tests
pytest tests/test_data_loading.py
```
