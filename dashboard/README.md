# Interactive Teacher Demonstration Dashboard

This interactive Streamlit dashboard provides a visual, empirical demonstration of the completed research milestones for the project:
**"Multi-Hospital Federated Learning for Chest X-Ray Classification with Malicious Participant Detection"**

---

## 1. Prerequisites & Installation

Activate the project virtual environment and ensure dependencies are installed:

```bash
# Activate existing project virtual environment
source .venv/bin/activate

# Install dashboard requirements
pip install -r requirements-dashboard.txt
```

---

## 2. Launching the Dashboard

Launch the Streamlit app:

```bash
streamlit run dashboard/app.py
```

By default, the dashboard opens at:
**[http://localhost:8501](http://localhost:8501)**

---

## 3. Operating Modes

### Mode A: Teacher Demo (Default)
Optimized for presenting the hospital scenario directly to your teacher or advisor:
- **One-Click Presentation**: Click the prominent **`▶ START PROJECT DEMO`** button to view verified results across Medical Dataset Verification and Centralized Baseline.
- **Fast & Reliable**: Loads existing verified experimental logs (`results/logs/medical_centralized_results.json`) instantaneously.
- **Presentation Script**: Contains the professor presentation script:
  *"We use chest X-ray images because the final project is designed around a hospital scenario. The dataset gives us a real medical-image classification problem that can later be distributed among simulated hospitals. We use a small CNN because it is lightweight, easy to train, and suitable for running the same model across multiple simulated hospitals. Each hospital can train on its own local medical images without directly sharing the raw patient images. The hospitals can contribute model updates to a shared global model."*
- **One-Click Reset**: Click **`↻ Reset Demo`** in the sidebar at any time.

### Mode B: Technical Details
Provides full transparency into engineering artifacts:
- Hardware, environment specifications, and random seed 42.
- Architectural Decision Records (ADR-001 through ADR-013).
- Raw machine-readable JSON logs for `EXP-MED-000-DATA` and `EXP-MED-001-CENTRAL-CNN`.
- Future hospital phase plug-in interfaces.

### Mode C: Archived Proof-of-Concept
Displays the complete historical CIFAR-10 exploration completed during Phase 1 and Phase 2 (76.94% accuracy), with explanation of why the project migrated to PneumoniaMNIST.

---

## 4. How to Run Fresh Medical CNN Training (Optional)

In **Teacher Demo** mode under **Stage 2: Centralized Medical Small CNN Baseline**, expand the `⚙️ Optional: Run Live Training Again` section:
1. Review the execution confirmation checkbox.
2. Click **`▶ Run Fresh Training`**.
3. The dashboard executes `.venv/bin/python experiments/train_medical.py` on your compute device (GPU/CPU) and live-streams epoch progress and loss/accuracy updates.
4. When training finishes, all figures and JSON results automatically reload.

---

## 5. What the Dashboard Currently Demonstrates

1. **Phase MEDICAL-01: Medical Dataset Verification** (`EXP-MED-000-DATA`):
   - 5,856 total pediatric chest X-rays (4,708 train, 524 validation, 624 untouched test).
   - Binary diagnosis (Normal: 0, Pneumonia: 1).
   - Representative chest X-ray sample grid (`pneumonia_sample_grid.png`).
   - Class distribution chart across official splits (`pneumonia_class_distribution.png`).
2. **Phase MEDICAL-02: Centralized Medical Small CNN Baseline** (`EXP-MED-001-CENTRAL-CNN`):
   - Compact `SmallCNN` architecture (420,802 parameters, ~1.61 MB).
   - Training hyperparameters: 15 epochs, Adam, lr=0.001, batch size 64.
   - **Measured Test Accuracy: 81.25%**, Test Loss: 0.9172 on untouched test set.
   - **Pneumonia Sensitivity / Recall: 99.74%** (389 / 390 cases correctly detected).
   - **Pneumonia Precision: 77.03%** | **Pneumonia F1-Score: 0.8693**.
   - Training time: 13.35 seconds on NVIDIA RTX 3050 Laptop GPU.
   - Medical disclaimer prominently displayed.
3. **Automated Quality & Test Suite**:
   - Status: **14 / 14 PyTest tests passed (100%)**.

---

## 6. Future Phases (Strictly NOT Implemented Yet)

The dashboard maintains strict scientific boundaries. The following modules are marked as **`PLANNED / NOT IMPLEMENTED`**:
- **Phase 3**: 10-Hospital Federated Learning with Flower + FedAvg — *NEXT MILESTONE*
- **Phase 4**: Non-IID Hospital Data Heterogeneity Simulation (Dirichlet case-mix skew)
- **Phase 5**: Malicious Hospital Attacks (Poisoned updates, backdoors)
- **Phase 6**: Non-IID-Aware Malicious Hospital Detector
- **Phase 7**: Robust Tri-State Consensus Aggregation
- **Phase 8**: Tamper-Evident Blockchain Medical Audit Trail
