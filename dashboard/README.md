# Interactive Teacher Demonstration Dashboard

This interactive Streamlit dashboard provides a visual, empirical demonstration of the completed research milestones for the project:
**"Blockchain-Assisted Non-IID-Aware Malicious Participant Detection in Federated Learning"**

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

## 3. Two Operating Modes

### Mode A: Teacher Demo (Default)
Optimized for presenting directly to your teacher or advisor:
- **One-Click Presentation**: Click the large **`▶ START PROJECT DEMO`** button. The dashboard smoothly loads and displays the verified results across Phase 1 and Phase 2.
- **Fast & Reliable**: By default, uses the existing verified experimental logs (`results/logs/`) rather than forcing a 100-second retraining delay during live presentations.
- **Presentation Script**: Provides a clear explanation script tailored for your instructor with a single-click copy area.
- **One-Click Reset**: Click **`↻ Reset Demo`** in the sidebar or stop the demo at any time without altering project files.

### Mode B: Technical Details
Provides full transparency into engineering artifacts:
- Hardware, environment specifications, and random seeds.
- Architectural Decision Records (ADR-001 through ADR-008).
- Raw machine-readable JSON logs for `EXP-000-DATA` and `EXP-001-CENTRAL-CNN`.
- Future phase plug-in interfaces.

---

## 4. How to Run Fresh CNN Training (Optional)

In **Teacher Demo** mode under **Stage 2: Centralized Small CNN Baseline**, expand the `⚙️ Optional: Run Training Again (Live Execution)` section:
1. Review the execution confirmation checkbox.
2. Click **`▶ Run Fresh Training`**.
3. The dashboard executes `.venv/bin/python experiments/train_centralized.py` on your compute device (GPU/CPU) and live-streams epoch progress and loss/accuracy updates.
4. When training finishes, all figures and JSON results automatically reload.

---

## 5. What the Dashboard Currently Demonstrates

1. **Phase 1: Dataset Verification** (`EXP-000-DATA`):
   - 60,000 total CIFAR-10 images (50,000 train source, 10,000 test).
   - 10 classes with exact uniform balance (5,000 train/class, 1,000 test/class).
   - Representative class sample grid (`cifar10_sample_grid.png`).
   - Class distribution chart (`cifar10_class_distribution.png`).
2. **Phase 2: Centralized Small CNN Baseline** (`EXP-001-CENTRAL-CNN`):
   - Compact `SmallCNN` architecture (189,130 parameters).
   - Training hyperparameters: 15 epochs, Adam, lr=0.001, batch size 64.
   - **Measured Test Accuracy: 76.94%**, Test Loss: 0.6578 on untouched test set.
   - Training time: 100.97 seconds on NVIDIA RTX 3050 Laptop GPU.
   - Loss & accuracy learning curves, plus 10-class confusion matrix.
3. **Automated Quality & Test Suite**:
   - Status: **13 / 13 PyTest tests passed (100%)**.
4. **Milestone Progress**:
   - **~67% progress** toward the initial Federated Learning baseline (Phase 1 & 2 complete, Phase 3 next).
   - Clearly labeled as project-management milestone progress, not model accuracy.

---

## 6. Future Phases (Strictly NOT Implemented Yet)

The dashboard maintains strict scientific boundaries. The following modules are marked as **`PLANNED / NOT IMPLEMENTED`**:
- **Phase 3**: 10-Client Federated Learning with FedAvg (Flower simulation engine) — *NEXT MILESTONE*
- **Phase 4**: Non-IID Dirichlet Partitioning ($\alpha \in \{1.0, 0.5, 0.1\}$)
- **Phase 5**: Adversarial Attacks (Label flipping, parameter scaling, backdoors)
- **Phase 6**: Multi-Factor Non-IID-Aware Detector (Orthogonal feature extraction)
- **Phase 7**: Robust Tri-State Aggregation (`Normal`, `Suspicious`, `Malicious`)
- **Phase 8**: Tamper-Evident Blockchain Audit Trail

---

## 7. Modular Future-Proof Architecture

The dashboard code is organized into decoupled plug-ins under `dashboard/`:
```text
dashboard/
├── app.py                     # Main Streamlit application and layout
├── requirements-dashboard.txt  # Dashboard dependencies (streamlit)
├── README.md                  # Dashboard documentation
└── demo/
    ├── __init__.py
    ├── pipeline_state.py      # Session state and cached log loaders
    ├── dataset_demo.py        # Stage 1: CIFAR-10 dataset verification component
    ├── cnn_demo.py            # Stage 2: Centralized SmallCNN component & live training
    └── future_stubs.py        # Clean plug-in interfaces for Phases 3 through 8
```
As Phase 3 and subsequent phases are implemented, corresponding demo modules (`fl_demo.py`, `noniid_demo.py`, etc.) can be seamlessly plugged in without restructuring the application.

