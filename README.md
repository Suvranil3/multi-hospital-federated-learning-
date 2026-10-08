# Multi-Hospital Federated Learning for Chest X-Ray Classification with Malicious Participant Detection

## Final-Year Research Project

### Research Question
> **"Can a non-IID-aware malicious participant detection mechanism reduce false positives while still detecting poisoning attacks in a multi-hospital Federated Learning setting?"**

---

## 1. Project Scenario & Medical Context

### The Hospital Problem Scenario
```text
Hospital 1  (Pediatric)   ─┐
Hospital 2  (General)     ─┤
Hospital 3  (Pulmonary)   ─┤
Hospital 4  (Community)   ─┤
Hospital 5  (Academic)    ─┤
...                       ├── Federated Learning ── Global Medical Model
Hospital 10 (Regional)    ┘
```

In healthcare, patient medical images are strictly confidential and governed by HIPAA/GDPR regulations. Hospitals cannot centralize patient scans in a single external database.

**Federated Learning (FL)** solves this by allowing 10 simulated hospitals to train the same Convolutional Neural Network locally on their respective chest X-rays. Hospitals share model weight updates with an aggregation server rather than raw patient data.

### Core Research Problem
- **Benign Clinical Heterogeneity (Non-IID)**: Honest hospitals treat different patient populations (e.g., pediatric clinics vs. specialized pulmonary centers), resulting in naturally divergent model weight updates.
- **Malicious Attacks**: A compromised hospital can mount poisoning attacks (label flipping, gradient manipulation) to degrade or backdoor the global model.
- **Goal**: Develop a multi-factor detector that distinguishes honest clinical diversity from deliberate malicious attacks, reducing false positives against honest specialized hospitals.
- **Blockchain Audit**: An immutable cryptographic ledger records update hashes and detector decisions for healthcare auditability and accountability.

---

## 2. Core Concepts in Simple Terms

### Why This Dataset?
> **"We use chest X-ray images because the final project is designed around a hospital scenario. The dataset gives us a real medical-image classification problem that can later be distributed among simulated hospitals."**

### Why This Model?
> **"We use a small CNN because it is lightweight, easy to train, and suitable for running the same model across multiple simulated hospitals."**

### Why This Matters for Federated Learning?
> **"Each hospital can train on its own local medical images without directly sharing the raw patient images. The hospitals can contribute model updates to a shared global model."**

---

## 3. Primary Medical Dataset: PneumoniaMNIST

The primary project dataset is **PneumoniaMNIST** from the official MedMNIST collection (Zenodo / Yang et al.):
- **Task**: Binary classification of pediatric chest X-rays (**0: Normal**, **1: Pneumonia**).
- **Format**: Grayscale images of dimension **1 × 28 × 28**.
- **Official Splits (Untouched)**:
  - **Train split**: 4,708 samples (1,214 Normal, 3,494 Pneumonia)
  - **Validation split**: 524 samples (135 Normal, 389 Pneumonia)
  - **Test split**: 624 samples (234 Normal, 390 Pneumonia, kept untouched until final evaluation)
  - **Total**: 5,856 images

*(The previous generic proof-of-concept work on CIFAR-10 is archived in [`results/archive/cifar10/`](file:///home/suvranil/project/results/archive/cifar10/README.md).)*

---

## 4. Current Milestone: Centralized Medical Baseline

- **Model**: `SmallCNN` (2 Conv blocks with BatchNorm and MaxPool, Flatten, Dense 128, 2-class output, **420,802 parameters**).
- **Centralized Test Performance** (Evaluated on untouched test split):
  - **Test Accuracy**: **81.25%** (507 / 624 correct)
  - **Pneumonia Sensitivity / Recall**: **99.74%** (389 / 390 pneumonia cases correctly detected)
  - **Pneumonia Precision**: **77.03%**
  - **Pneumonia F1-Score**: **0.8693**
  - **Macro F1-Score**: **0.7689**
  - **Confusion Matrix**: TN=118, FP=116, FN=1, TP=389
- **Medical Note**: *These are research benchmark scores evaluated on 28×28 images for federated learning simulation; they do not represent clinical diagnostic deployment.*

---

## 5. Repository Structure
```text
project/
├── README.md                 # Project overview and hospital scenario
├── PROJECT_STATUS.md         # Active development phase and status
├── RESEARCH_OBJECTIVE.md     # Research questions, hypotheses, and scope
├── RESEARCH_GAP.md           # Byzantine detection limits under clinical non-IID
├── ARCHITECTURE.md           # 4-tier system design and security model
├── DECISIONS.md              # Architectural Decision Records (ADRs)
├── CHANGELOG.md              # Chronological change log
├── EXPERIMENT_LOG.md         # Empirical experiment record
├── requirements.txt          # Python dependencies (includes medmnist)
│
├── data/
│   ├── README.md             # Dataset documentation and partitioning
│   ├── raw/                  # Downloaded raw PneumoniaMNIST (.npz)
│   └── verify_dataset.py     # PneumoniaMNIST verification and stats script
│
├── federated_learning/
│   └── model.py              # SmallCNN PyTorch model (420,802 params, 1x28x28 -> 2)
│
├── experiments/
│   ├── config.py             # Central reproducibility configuration & seed 42
│   ├── train_medical.py      # Centralized medical baseline training pipeline
│   └── train_centralized.py  # Centralized pipeline runner / alias
│
├── dashboard/
│   ├── app.py                # Interactive Streamlit teacher dashboard
│   ├── README.md             # Dashboard documentation and guide
│   └── demo/                 # Dashboard visual stages and stubs
│
├── results/
│   ├── figures/              # Generated medical visualizations and plots
│   ├── logs/                 # Run logs and machine-readable JSON summaries
│   ├── models/               # Best trained model checkpoints (.pt)
│   └── archive/cifar10/      # Archived previous CIFAR-10 proof-of-concept
│
└── tests/
    ├── test_data_loading.py  # Pytest suite for PneumoniaMNIST data integrity
    ├── test_model.py         # Pytest suite for SmallCNN model architecture
    └── test_training.py      # Pytest suite for training logic and medical metrics
```

---

## 6. Development Progression
- [x] **Phase MEDICAL-01: Medical Dataset Verification (PneumoniaMNIST)**
- [x] **Phase MEDICAL-02: Centralized Medical Small CNN Baseline**
- [ ] **Phase 3: 10-Hospital Federated Learning Baseline (Flower + FedAvg)**
- [ ] **Phase 4: Non-IID Hospital Data Heterogeneity Simulation**
- [ ] **Phase 5: Malicious Hospital Attack Simulation (Poisoning, Backdoor)**
- [ ] **Phase 6: Multi-Factor Non-IID-Aware Detector**
- [ ] **Phase 7: Robust Aggregation Baselines**
- [ ] **Phase 8: Blockchain Tamper-Evident Medical Audit Trail**
- [ ] **Phase 9: Comprehensive Empirical Evaluation & Ablation**
- [ ] **Phase 10: Publication Artifacts**

---

## 7. Setup & Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.14.4)
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
pip install -r requirements-dashboard.txt
```

### Running Verification & Baseline Training
```bash
# Phase MEDICAL-01: Verify PneumoniaMNIST dataset
.venv/bin/python data/verify_dataset.py

# Phase MEDICAL-02: Train centralized medical CNN baseline
.venv/bin/python experiments/train_medical.py

# Launch interactive teacher dashboard
.venv/bin/streamlit run dashboard/app.py

# Run all automated tests
.venv/bin/pytest tests/ -v
```
