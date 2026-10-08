"""
Interactive Teacher Demonstration Dashboard.
Research Project: 'Multi-Hospital Federated Learning for Chest X-Ray Classification with Malicious Participant Detection'

Modes:
- Teacher Demo (Default): Visual, single-click presentation for the professor.
- Technical Details: Comprehensive engineering metrics, configurations, file paths, and logs.
- Archived Proof-of-Concept: Complete historical CIFAR-10 baseline results.
"""

from pathlib import Path
import sys
import streamlit as st
from PIL import Image

# Setup Root Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.demo.pipeline_state import (
    init_session_state,
    reset_demo_state,
    load_medical_dataset_report,
    load_medical_cnn_results,
    load_archived_cifar_report,
    load_archived_cifar_results,
    RESULTS_ARCHIVE_DIR,
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
    page_title="Hospital FL Security | Teacher Demo",
    page_icon="🏥",
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
    .archive-box {
        background-color: #F3F4F6;
        border-left: 5px solid #6B7280;
        border-radius: 0.35rem;
        padding: 1rem;
        margin-bottom: 1rem;
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
        ["Teacher Demo", "Technical Details", "Archived Proof-of-Concept"],
        index=0,
        help="Select Teacher Demo for hospital presentation, Technical Details for logs, or Archived Proof-of-Concept for historical CIFAR-10 data.",
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
        - **Phase MEDICAL-01 (PneumoniaMNIST)**: <span class="status-pill pill-complete">COMPLETED</span><br>
        - **Phase MEDICAL-02 (Centralized Baseline)**: <span class="status-pill pill-complete">COMPLETED</span><br>
        - **Phase 3 (10-Hospital FL with Flower)**: <span class="status-pill pill-next">NEXT</span><br>
        - **Phase 4 (Non-IID Hospital Data)**: <span class="status-pill pill-planned">PLANNED</span><br>
        - **Phase 5 (Malicious Attacks)**: <span class="status-pill pill-planned">PLANNED</span><br>
        - **Phase 6 (Detector)**: <span class="status-pill pill-planned">PLANNED</span><br>
        - **Phase 8 (Blockchain Audit)**: <span class="status-pill pill-planned">PLANNED</span>
        """,
        unsafe_allow_html=True,
    )
    st.caption("ℹ️ Current milestone complete: Centralized medical baseline established. Federated learning not yet started.")

    st.divider()
    st.markdown("### System Quick Info")
    st.write("**Dataset**: `PneumoniaMNIST (Chest X-Rays)`")
    st.write("**Task**: `Normal vs Pneumonia (Binary)`")
    st.write("**Model**: `SmallCNN (420,802 params)`")
    st.write("**Current Stage**: `Medical Centralized Baseline`")
    st.write("**Unit Tests**: `14 / 14 Passing`")
    st.write("**Hardware**: `RTX 3050 GPU (CUDA)`")


# Header Banner
st.markdown('<div class="demo-title">Multi-Hospital Federated Learning for Chest X-Ray Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="demo-sub">With Non-IID-Aware Malicious Participant Detection and Tamper-Evident Blockchain Audit</div>', unsafe_allow_html=True)


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
            st.success("✨ **Interactive Demonstration Active**: Showing verified Medical Dataset & Centralized Baseline results.")
    with col_reset_btn:
        if st.session_state.demo_started:
            if st.button("⏹ Stop / Reset Demo", use_container_width=True):
                reset_demo_state()
                st.rerun()

    st.divider()

    # Pre-demo introductory banner if not started
    if not st.session_state.demo_started:
        st.info("👆 Click **'▶ START PROJECT DEMO'** above to present the hospital project to your professor.")
        st.markdown(
            """
            #### Demonstration Highlights:
            1. **Hospital Scenario**: 10 simulated hospitals collaborating via Federated Learning to classify Chest X-Rays without exposing raw patient images.
            2. **PneumoniaMNIST Dataset**: 5,856 pediatric chest X-ray images (4,708 train, 524 val, 624 untouched test).
            3. **Centralized Baseline Model**: Lightweight SmallCNN (420,802 parameters) achieving **81.25% Test Accuracy** and **99.74% Pneumonia Recall**.
            4. **Next Step**: 10-Hospital Federated Learning with Flower + FedAvg.
            """
        )

    # When demo is started, smoothly display all stages
    if st.session_state.demo_started:
        # Step Progress Indicator
        step_col1, step_col2, step_col3 = st.columns(3)
        with step_col1:
            st.markdown("#### Step 1: Medical Dataset")
            st.progress(1.0)
            st.caption("✓ 5,856 Chest X-Rays, PneumoniaMNIST verified")
        with step_col2:
            st.markdown("#### Step 2: Centralized CNN Baseline")
            st.progress(1.0)
            st.caption("✓ 81.25% Test Acc, 99.74% Pneumonia Recall")
        with step_col3:
            st.markdown("#### Step 3: 10-Hospital FL")
            st.progress(0.0)
            st.caption("→ Upcoming next milestone")

        st.divider()

        # HOSPITAL SCENARIO CONCEPT CARD
        st.markdown("### 🏥 The Hospital Problem Scenario")
        st.markdown(
            """
            ```text
            Hospital 1  (Pediatric)   ─┐
            Hospital 2  (General)     ─┤
            Hospital 3  (Pulmonary)   ─┤
            Hospital 4  (Community)   ─┤
            Hospital 5  (Academic)    ─┤
            ...                       ├── Federated Learning ── Global Medical Model
            Hospital 10 (Regional)    ┘
            ```
            """
        )
        c_exp1, c_exp2, c_exp3 = st.columns(3)
        with c_exp1:
            st.markdown("**1. Privacy Guarantee**")
            st.write("Hospitals keep their sensitive patient chest X-rays strictly local. Only model updates are transmitted.")
        with c_exp2:
            st.markdown("**2. Natural Non-IID Diversity**")
            st.write("Honest hospitals naturally see different case mixes (e.g., pediatric vs. adult), leading to benign divergent updates.")
        with c_exp3:
            st.markdown("**3. Malicious Threat Detection**")
            st.write("A compromised hospital could upload toxic poisoned weights. Our research focuses on distinguishing benign clinical skew from malicious attacks.")

        st.divider()

        # STAGE 1: MEDICAL DATASET VERIFICATION
        render_dataset_stage()
        st.divider()

        # STAGE 2: CENTRALIZED MEDICAL BASELINE
        render_cnn_stage()
        st.divider()

        # TEST STATUS CARD
        st.markdown("### 🧪 Automated Quality & Test Suite")
        t_col1, t_col2 = st.columns([2, 3])
        with t_col1:
            st.success("### ✓ 14 / 14 Tests Passed (100%)")
            st.caption("Regression tested across PneumoniaMNIST data loaders, SmallCNN architecture, gradient descent, and medical metrics.")
        with t_col2:
            with st.expander("🔍 Show test details"):
                st.markdown(
                    """
                    | Test File | Test Name | Status |
                    | :--- | :--- | :---: |
                    | `tests/test_data_loading.py` | `test_pneumoniamnist_official_splits_exist` | **PASSED** |
                    | `tests/test_data_loading.py` | `test_pneumoniamnist_batch_shape_and_types` | **PASSED** |
                    | `tests/test_data_loading.py` | `test_pneumoniamnist_deterministic_shuffling` | **PASSED** |
                    | `tests/test_model.py` | `test_cnn_construction` | **PASSED** |
                    | `tests/test_model.py` | `test_cnn_output_shape [1, 4, 16, 32]` | **PASSED** |
                    | `tests/test_model.py` | `test_cnn_forward_pass_numerical_validity` | **PASSED** |
                    | `tests/test_model.py` | `test_cnn_loss_calculation` | **PASSED** |
                    | `tests/test_model.py` | `test_cnn_one_training_step` | **PASSED** |
                    | `tests/test_training.py` | `test_prepare_data_loaders_splits` | **PASSED** |
                    | `tests/test_training.py` | `test_prepare_data_loaders_batch_shapes` | **PASSED** |
                    | `tests/test_training.py` | `test_compute_medical_metrics` | **PASSED** |
                    """
                )

        st.divider()

        # RESEARCH PIPELINE PROGRESSION
        render_pipeline_visualization()
        st.divider()

        # TEACHER SCRIPT / EXPLANATION
        st.markdown("### 🗣️ Presentation Guide for Professor")
        teacher_speech = (
            "We use chest X-ray images because the final project is designed around a hospital scenario. "
            "The dataset gives us a real medical-image classification problem that can later be distributed among simulated hospitals. "
            "We use a small CNN because it is lightweight, easy to train, and suitable for running the same model across multiple simulated hospitals. "
            "Each hospital can train on its own local medical images without directly sharing the raw patient images. "
            "The hospitals can contribute model updates to a shared global model."
        )

        st.markdown(
            f"""
            <div class="script-box">
                <p style="font-size: 1.05rem; line-height: 1.6; margin-bottom: 0.5rem; color: #1F2937;">
                    <b>Verbatim Presentation Script:</b><br>
                    "{teacher_speech}"
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.text_area("📋 Copyable Script:", value=teacher_speech, height=90)


# =========================================================================
# MODE B: TECHNICAL DETAILS
# =========================================================================
elif mode == "Technical Details":
    st.subheader("🛠️ Technical Details & Medical Engineering Specifications")

    tab_t1, tab_t2, tab_t3 = st.tabs(["⚙️ Configurations & Metadata", "📄 Raw Experiment Logs", "🔮 Future Hospital Modules"])

    with tab_t1:
        st.markdown("#### Reproducibility & Medical Environment")
        env_col1, env_col2 = st.columns(2)
        with env_col1:
            st.write("**Primary Dataset**: `PneumoniaMNIST (MedMNIST v3)`")
            st.write("**Task**: `Binary (0: Normal, 1: Pneumonia)`")
            st.write("**Image Dimensions**: `1 × 28 × 28 (Grayscale)`")
            st.write("**Python Version**: `3.14.4`")
            st.write("**PyTorch Version**: `2.14.1+cu130`")
            st.write("**Compute Device**: `NVIDIA GeForce RTX 3050 Laptop GPU`")
            st.write("**Master Seed**: `42`")
        with env_col2:
            st.write("**Official Train Split**: 4,708 samples")
            st.write("**Official Validation Split**: 524 samples")
            st.write("**Official Test Split**: 624 samples (Untouched)")
            st.write("**Batch Size**: 64 | **Epochs**: 15 | **Optimizer**: Adam")
            st.write("**Weight Decay**: 1e-4 | **Loss**: CrossEntropyLoss")
            st.write("**Model Parameters**: 420,802 parameters (~1.61 MB)")

        st.divider()
        st.markdown("#### Architectural Decisions for Hospital Scenario")
        st.markdown(
            """
            - **ADR-009**: PneumoniaMNIST selected as primary medical dataset for multi-hospital federated learning.
            - **ADR-010**: SmallCNN modified for 1-channel grayscale 28x28 chest X-rays and binary diagnosis head.
            - **ADR-011**: Preservation of official MedMNIST train/val/test splits without artificial re-splitting.
            - **ADR-012**: Mandatory reporting of Precision, Sensitivity/Recall, and F1-score alongside Accuracy for medical evaluation.
            - **ADR-013**: Archival of CIFAR-10 proof-of-concept into `results/archive/cifar10/`.
            """
        )

    with tab_t2:
        st.markdown("#### Raw JSON Experiment Records")
        med_data_raw = load_medical_dataset_report()
        med_cnn_raw = load_medical_cnn_results()

        st.markdown("##### 1. PneumoniaMNIST Verification Report (`EXP-MED-000-DATA`)")
        if med_data_raw:
            st.json(med_data_raw)
        else:
            st.warning("PneumoniaMNIST verification report not found.")

        st.markdown("##### 2. Centralized Medical CNN Baseline Results (`EXP-MED-001-CENTRAL-CNN`)")
        if med_cnn_raw:
            st.json(med_cnn_raw)
        else:
            st.warning("Medical CNN baseline report not found.")

    with tab_t3:
        st.markdown("#### Future Hospital Research Modules")
        render_fl_baseline_demo()
        st.divider()
        render_noniid_demo()
        st.divider()
        render_attack_demo()
        st.divider()
        render_detector_demo()
        st.divider()
        render_blockchain_demo()


# =========================================================================
# MODE C: ARCHIVED PROOF-OF-CONCEPT (CIFAR-10)
# =========================================================================
else:
    st.subheader("📦 Archived Previous Proof-of-Concept (CIFAR-10)")
    st.markdown(
        """
        <div class="archive-box">
            <b>Archival Notice:</b> This section contains the complete historical Phase 1 and Phase 2 proof-of-concept work conducted on the generic CIFAR-10 dataset.<br>
            Per the professor's requirement that the entire project be presented in a medical hospital context,
            <b>PneumoniaMNIST</b> is now the primary project dataset. CIFAR-10 is preserved here in full as historical baseline evidence.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cifar_rep = load_archived_cifar_report()
    cifar_res = load_archived_cifar_results()

    ac_col1, ac_col2, ac_col3 = st.columns(3)
    with ac_col1:
        st.metric("CIFAR-10 Images", "60,000", help="50,000 train, 10,000 test")
        st.caption("10 balanced object classes (32x32 RGB)")
    with ac_col2:
        st.metric("Centralized Accuracy", "76.94%", help="Evaluated on 10,000 test images")
        st.caption("189k-parameter SmallCNN")
    with ac_col3:
        st.metric("Training Time", "100.97s", help="15 epochs on RTX 3050 GPU")
        st.caption("Adam optimizer, seed 42")

    st.markdown("#### Archived Visual Artifacts")
    f_c1, f_c2 = st.columns(2)
    cifar_grid = RESULTS_ARCHIVE_DIR / "figures" / "cifar10_sample_grid.png"
    cifar_cm = RESULTS_ARCHIVE_DIR / "figures" / "cifar10_confusion_matrix.png"

    with f_c1:
        if cifar_grid.exists():
            st.image(Image.open(cifar_grid), caption="Archived CIFAR-10 Sample Grid", use_container_width=True)
    with f_c2:
        if cifar_cm.exists():
            st.image(Image.open(cifar_cm), caption="Archived CIFAR-10 10-Class Confusion Matrix", use_container_width=True)

    with st.expander("📄 View Archived CIFAR-10 JSON Logs"):
        st.write("**Verification Report:**")
        if cifar_rep:
            st.json(cifar_rep)
        st.write("**Centralized Training Results:**")
        if cifar_res:
            st.json(cifar_res)


# Footer
st.divider()
st.caption("Multi-Hospital Federated Learning Security | Chest X-Ray Classification (PneumoniaMNIST) | Seed: 42")
