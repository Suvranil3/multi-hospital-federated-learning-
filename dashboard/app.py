"""
Interactive Teacher Demonstration Dashboard.
Research Project: 'Blockchain-Assisted Non-IID-Aware Malicious Participant Detection in Federated Learning'

Modes:
- Teacher Demo (Default): Visual, single-click automated presentation of Phase 1 and Phase 2.
- Technical Details: Comprehensive engineering metrics, configurations, file paths, and logs.
"""

from pathlib import Path
import sys
import streamlit as st

# Setup Root Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.demo.pipeline_state import (
    init_session_state,
    reset_demo_state,
    load_cifar_report,
    load_cnn_results,
    RESULTS_LOGS_DIR,
    RESULTS_FIGURES_DIR,
)
from dashboard.demo.dataset_demo import render_dataset_stage
from dashboard.demo.cnn_demo import render_cnn_stage
from dashboard.demo.future_stubs import (
    render_pipeline_visualization,
    render_fl_baseline_demo,
    render_noniid_demo,
    render_attack_demo,
    render_detector_demo,
    render_blockchain_demo,
)

# Page Setup
st.set_page_config(
    page_title="Teacher Demo | Blockchain FL Research",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
init_session_state()

# Custom CSS for Professional Presentation
st.markdown(
    """
    <style>
    .demo-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .demo-sub {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .script-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        border-radius: 0.35rem;
        padding: 1.1rem;
        margin-top: 1rem;
        margin-bottom: 1.2rem;
    }
    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        font-size: 0.85rem;
        font-weight: 700;
        border-radius: 9999px;
    }
    .pill-complete {
        background-color: #D1FAE5;
        color: #065F46;
    }
    .pill-next {
        background-color: #FEF3C7;
        color: #92400E;
    }
    .pill-planned {
        background-color: #E5E7EB;
        color: #374151;
    }
    .big-btn {
        display: flex;
        justify-content: center;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    mode = st.radio(
        "Dashboard Mode",
        ["Teacher Demo", "Technical Details"],
        index=0,
        help="Select Teacher Demo for high-level visual presentation, or Technical Details for full engineering logs.",
    )

    st.divider()
    st.markdown("### Interactive Presentation")
    if st.button("↻ Reset Demo", use_container_width=True):
        reset_demo_state()
        st.rerun()

    st.divider()
    st.markdown("### Project Milestone Tracker")
    st.markdown(
        """
        - **Phase 1 (Dataset)**: <span class="status-pill pill-complete">COMPLETED</span><br>
        - **Phase 2 (Centralized CNN)**: <span class="status-pill pill-complete">COMPLETED</span><br>
        - **Phase 3 (10-Client FL)**: <span class="status-pill pill-next">NEXT</span><br>
        - **Phase 4 (Non-IID Skew)**: <span class="status-pill pill-planned">PLANNED</span><br>
        - **Phase 5 (Attacks)**: <span class="status-pill pill-planned">PLANNED</span><br>
        - **Phase 6 (Detector)**: <span class="status-pill pill-planned">PLANNED</span><br>
        - **Phase 8 (Blockchain)**: <span class="status-pill pill-planned">PLANNED</span>
        """,
        unsafe_allow_html=True,
    )
    st.caption("ℹ️ Progress toward initial FL baseline milestone: ~67% (project-management progress, not model accuracy).")

    st.divider()
    st.markdown("### Repository Quick Info")
    st.write("**Git Checkpoint**: `acac4f8`")
    st.write("**Unit Tests**: `13 / 13 Passing`")
    st.write("**Hardware**: `RTX 3050 GPU (CUDA)`")


# Header
st.markdown('<div class="demo-title">Blockchain-Assisted Non-IID-Aware Malicious Participant Detection in Federated Learning</div>', unsafe_allow_html=True)
st.markdown('<div class="demo-sub">Interactive Academic Demonstration & Empirical Research Progress</div>', unsafe_allow_html=True)


# =========================================================================
# MODE A: TEACHER DEMO (DEFAULT)
# =========================================================================
if mode == "Teacher Demo":
    # Prominent Action Bar
    col_demo_btn, col_reset_btn = st.columns([3, 1])
    with col_demo_btn:
        if not st.session_state.demo_started:
            if st.button("▶ START PROJECT DEMO", type="primary", use_container_width=True):
                st.session_state.demo_started = True
                st.rerun()
        else:
            st.success("✨ **Interactive Demonstration Active**: Showing verified Phase 1 & Phase 2 results.")
    with col_reset_btn:
        if st.session_state.demo_started:
            if st.button("⏹ Stop / Reset Demo", use_container_width=True):
                reset_demo_state()
                st.rerun()

    st.divider()

    # Pre-demo introductory banner if not started
    if not st.session_state.demo_started:
        st.info("👆 Click **'▶ START PROJECT DEMO'** above to begin the step-by-step presentation for your professor.")
        st.markdown(
            """
            #### Demonstration Highlights:
            1. **Dataset Integrity Verification**: 60,000 CIFAR-10 images verified across 10 balanced classes.
            2. **SmallCNN Baseline**: 189,130-parameter model achieving **76.94% accuracy** centrally.
            3. **Future Research Boundaries**: Clearly distinguishing completed baselines from upcoming Federated Learning and Byzantine detection.
            """
        )

    # When demo is started, smoothly display all stages
    if st.session_state.demo_started:
        # Step Progress Indicator
        step_col1, step_col2, step_col3 = st.columns(3)
        with step_col1:
            st.markdown("#### Step 1: Dataset Verification")
            st.progress(1.0)
            st.caption("✓ 60,000 images, 10 classes balanced")
        with step_col2:
            st.markdown("#### Step 2: SmallCNN Baseline")
            st.progress(1.0)
            st.caption("✓ 76.94% Test Accuracy achieved")
        with step_col3:
            st.markdown("#### Step 3: Federated Learning")
            st.progress(0.0)
            st.caption("→ Next upcoming milestone")

        st.divider()

        # STAGE 1: DATASET VERIFICATION
        render_dataset_stage()
        st.divider()

        # STAGE 2: CENTRALIZED CNN BASELINE
        render_cnn_stage()
        st.divider()

        # TEST STATUS CARD
        st.markdown("### 🧪 Automated Quality & Test Suite")
        t_col1, t_col2 = st.columns([2, 3])
        with t_col1:
            st.success("### ✓ 13 / 13 Tests Passed (100%)")
            st.caption("Strict regression testing across data loading, tensor validity, and model gradient descent.")
        with t_col2:
            with st.expander("🔍 Show test details"):
                st.markdown(
                    """
                    | Test File | Test Name | Status |
                    | :--- | :--- | :---: |
                    | `tests/test_data_loading.py` | `test_cifar10_raw_splits_exist` | **PASSED** |
                    | `tests/test_data_loading.py` | `test_cifar10_batch_shape_and_types` | **PASSED** |
                    | `tests/test_data_loading.py` | `test_cifar10_deterministic_shuffling` | **PASSED** |
                    | `tests/test_model.py` | `test_cnn_construction` | **PASSED** |
                    | `tests/test_model.py` | `test_cnn_output_shape [1, 4, 16, 32]` | **PASSED** |
                    | `tests/test_model.py` | `test_cnn_forward_pass_numerical_validity` | **PASSED** |
                    | `tests/test_model.py` | `test_cnn_loss_calculation` | **PASSED** |
                    | `tests/test_model.py` | `test_cnn_one_training_step` | **PASSED** |
                    | `tests/test_training.py` | `test_prepare_data_loaders_splits` | **PASSED** |
                    | `tests/test_training.py` | `test_prepare_data_loaders_batch_shapes` | **PASSED** |
                    """
                )

        st.divider()

        # RESEARCH PIPELINE PROGRESSION
        render_pipeline_visualization()
        st.divider()

        # TEACHER SCRIPT / EXPLANATION
        st.markdown("### 🗣️ What I Can Say to My Teacher")
        teacher_speech = (
            "We first verified the CIFAR-10 dataset and confirmed its structure and class balance. "
            "Then we built and validated a lightweight CNN using PyTorch. "
            "The centralized model achieved a measured test accuracy of 76.94 percent. "
            "The next stage is to use this same model in a 10-client Federated Learning setup using Flower and FedAvg. "
            "After that, we will study the effect of non-IID data and malicious participants."
        )

        st.markdown(
            f"""
            <div class="script-box">
                <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 0.5rem; color: #1F2937;">
                    <b>Presentation Script:</b><br>
                    "{teacher_speech}"
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Copyable area
        st.text_area("📋 Copyable Script:", value=teacher_speech, height=90)

        st.divider()

        # CURRENT PROJECT STRUCTURE
        st.markdown("### 📁 Current Project Structure")
        with st.expander("📂 View repository layout and active directories", expanded=False):
            st.markdown(
                """
                ```text
                Blockchain_FL_Project/
                ├── data/                 [ACTIVE] Raw dataset and verification script
                ├── federated_learning/   [ACTIVE] SmallCNN PyTorch model (Reusable)
                ├── experiments/          [ACTIVE] Central config (seed 42) & centralized training
                ├── results/              [ACTIVE] Generated figures, tables, and JSON logs
                ├── tests/                [ACTIVE] 13 automated unit tests
                ├── dashboard/            [ACTIVE] Demonstration app and modules
                ├── attacks/              [RESERVED / STUB] Phase 5 attack simulation
                ├── detection/            [RESERVED / STUB] Phase 6 non-IID-aware detector
                └── blockchain/           [RESERVED / STUB] Phase 8 audit trail smart contract
                ```
                """
            )


# =========================================================================
# MODE B: TECHNICAL DETAILS
# =========================================================================
else:
    st.subheader("🛠️ Technical Details & Engineering Specifications")

    tab_t1, tab_t2, tab_t3 = st.tabs(["⚙️ Configurations & Metadata", "📄 Raw Experiment Logs", "🔮 Future Plug-in Modules"])

    with tab_t1:
        st.markdown("#### Reproducibility & Environment")
        env_col1, env_col2 = st.columns(2)
        with env_col1:
            st.write("**Python Version**: `3.14.4`")
            st.write("**PyTorch Version**: `2.14.1+cu130`")
            st.write("**Device**: `NVIDIA GeForce RTX 3050 Laptop GPU`")
            st.write("**Master Random Seed**: `42`")
            st.write("**Git Checkpoint**: `acac4f8`")
        with env_col2:
            st.write("**Dataset Source**: Official CIFAR-10 (torchvision mirror)")
            st.write("**Train Split**: 40,000 samples (Stratified 80%)")
            st.write("**Validation Split**: 10,000 samples (Stratified 20%)")
            st.write("**Test Split**: 10,000 samples (Untouched)")
            st.write("**Batch Size**: 64 | **Epochs**: 15 | **Optimizer**: Adam")

        st.divider()
        st.markdown("#### Architectural Decision Records (ADRs)")
        st.markdown(
            r"""
            - **ADR-001**: CIFAR-10 Dataset selection.
            - **ADR-002**: SmallCNN architecture (~189k parameters) for lightweight simulation.
            - **ADR-003**: Flower framework (`flwr[simulation]`) selection.
            - **ADR-004**: Dirichlet non-IID partitioning parameterization ($\alpha \in \{1.0, 0.5, 0.1\}$).
            - **ADR-005**: Decoupled blockchain role as tamper-evident audit trail only (zero on-chain ML/weights).
            - **ADR-006**: Tri-state classification (`Normal`, `Suspicious`, `Malicious`).
            - **ADR-007**: Phased modular development strategy.
            - **ADR-008**: Centralized training protocol and stratified split.
            """
        )


    with tab_t2:
        st.markdown("#### Raw JSON Experiment Records")
        cifar_raw = load_cifar_report()
        cnn_raw = load_cnn_results()

        st.markdown("##### 1. CIFAR-10 Verification Report (`EXP-000-DATA`)")
        if cifar_raw:
            st.json(cifar_raw)
        else:
            st.warning("CIFAR-10 report not found.")

        st.markdown("##### 2. Centralized CNN Results (`EXP-001-CENTRAL-CNN`)")
        if cnn_raw:
            st.json(cnn_raw)
        else:
            st.warning("Centralized CNN report not found.")

    with tab_t3:
        st.markdown("#### Future Phase Plug-in Interfaces (Clean Stubs)")
        st.caption("These placeholders will be activated as subsequent phases are implemented.")
        render_fl_baseline_demo()
        st.divider()
        render_noniid_demo()
        st.divider()
        render_attack_demo()
        st.divider()
        render_detector_demo()
        st.divider()
        render_blockchain_demo()

# Footer
st.divider()
st.caption("Blockchain-Assisted Non-IID-Aware Malicious Participant Detection in Federated Learning | Read-Only Demonstration Dashboard | Seed: 42")
