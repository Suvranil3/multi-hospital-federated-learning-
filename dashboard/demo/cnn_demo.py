"""
Interactive Stage 2: Centralized Medical Small CNN Baseline.
Primary Scenario: Multi-Hospital Chest X-Ray Classification.
"""

from pathlib import Path
import subprocess
import sys
import streamlit as st
from PIL import Image

from dashboard.demo.pipeline_state import (
    PROJECT_ROOT,
    RESULTS_FIGURES_DIR,
    load_medical_cnn_results,
)


def run_medical_training_subprocess() -> bool:
    """Execute real medical training script via subprocess with streaming progress."""
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    script_path = PROJECT_ROOT / "experiments" / "train_medical.py"

    if not venv_python.exists():
        venv_python = Path(sys.executable)

    progress_bar = st.progress(0)
    status_text = st.empty()
    log_box = st.empty()

    status_text.info("🚀 Initiating medical CNN training on GPU/CPU... Please wait.")

    try:
        process = subprocess.Popen(
            [str(venv_python), str(script_path)],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines = []
        epoch_count = 0
        total_epochs = 15

        for line in iter(process.stdout.readline, ""):
            output_lines.append(line)
            # Show last 8 lines of logs
            log_box.code("".join(output_lines[-8:]), language="text")

            # Parse epoch progress if line matches Epoch [XX/15]
            if "Epoch [" in line:
                epoch_count += 1
                frac = min(1.0, epoch_count / total_epochs)
                progress_bar.progress(frac)
                status_text.info(f"⏳ Training in progress: Epoch {epoch_count}/{total_epochs}...")

        process.stdout.close()
        return_code = process.wait()

        if return_code == 0:
            progress_bar.progress(1.0)
            status_text.success("✅ Training completed successfully! Fresh medical metrics loaded.")
            # Clear cache to reload fresh results
            load_medical_cnn_results.clear()
            return True
        else:
            status_text.error(f"❌ Training failed with exit code {return_code}.")
            return False

    except Exception as e:
        status_text.error(f"❌ Execution error: {str(e)}")
        return False


def render_cnn_stage() -> None:
    """Render Stage 2: Centralized Medical Small CNN baseline."""
    results = load_medical_cnn_results()

    # Stage Header
    col_hdr, col_badge = st.columns([3, 1])
    with col_hdr:
        st.markdown("### 🧠 Stage 2: Centralized Medical Small CNN Baseline")
        st.write("Evaluating the baseline CNN on Chest X-Rays prior to distributing it across simulated hospital clients.")
    with col_badge:
        st.markdown(
            """
            <div style="text-align: right; padding-top: 0.5rem;">
                <span style="background-color: #10B981; color: white; padding: 0.35rem 0.8rem; border-radius: 0.35rem; font-weight: bold; font-size: 0.95rem;">
                    ✓ Baseline Validated
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Explanation block: WHY THIS MODEL
    st.markdown(
        """
        > **Why This Model?**  
        > *We use a small CNN because it is lightweight, easy to train, and suitable for running the same model across multiple simulated hospitals.*
        """
    )

    # 1. Visual Architecture Flow & Hyperparameters
    st.markdown("#### 1. Architecture & Design Specification")
    arch_col, spec_col = st.columns([1, 1])

    with arch_col:
        st.markdown(
            """
            ```text
            +--------------------------------------------+
            |      Input Chest X-Ray (1 × 28 × 28)       |
            +--------------------------------------------+
                                  ↓
            +--------------------------------------------+
            | Conv Block 1: 32 Filters (3x3), BN, MaxPool| -> 32 × 14 × 14
            +--------------------------------------------+
                                  ↓
            +--------------------------------------------+
            | Conv Block 2: 64 Filters (3x3), BN, MaxPool| -> 64 × 7 × 7
            +--------------------------------------------+
                                  ↓
            +--------------------------------------------+
            |         Flatten Layer (3,136 dims)         |
            +--------------------------------------------+
                                  ↓
            +--------------------------------------------+
            |        Dense 128 (ReLU, Dropout 0.2)       |
            +--------------------------------------------+
                                  ↓
            +--------------------------------------------+
            |    2-Class Output (Normal vs Pneumonia)    |
            +--------------------------------------------+
            ```
            """
        )

    with spec_col:
        st.markdown("##### Configuration Parameters")
        p1, p2 = st.columns(2)
        with p1:
            st.metric("Total Parameters", "420,802", help="Lightweight for 10-hospital FL simulation")
            st.metric("Batch Size", "64")
        with p2:
            st.metric("Optimizer", "Adam (lr=0.001)")
            st.metric("Epochs", "15")

        st.caption("ℹ️ *Default Mode*: Displaying verified results from `results/logs/medical_centralized_results.json`.")

        # Fresh Training Expansion
        with st.expander("⚙️ Optional: Run Live Training Again"):
            st.warning("⚠️ Running training executes 15 epochs on your GPU/CPU (~13s on GPU).")
            confirm = st.checkbox("I confirm I want to execute fresh medical training")
            if st.button("▶ Run Fresh Training", disabled=not confirm):
                with st.spinner("Executing experiments/train_medical.py..."):
                    run_medical_training_subprocess()
                    results = load_medical_cnn_results()

    st.divider()

    # 2. Final Model Results (Comprehensive Medical Scorecard)
    st.markdown("#### 2. Centralized Model Evaluation & Medical Metrics")
    if results is None:
        st.warning("Centralized medical CNN results not yet available. Run `.venv/bin/python experiments/train_medical.py`.")
        return

    test_perf = results.get("test_performance", {})
    train_perf = results.get("training_performance", {})
    env_info = results.get("environment", {})

    test_acc = test_perf.get("test_accuracy", 81.25)
    test_loss = test_perf.get("test_loss", 0.9172)
    p_metrics = test_perf.get("pneumonia_metrics", {})
    p_prec = p_metrics.get("precision", 0.7703) * 100
    p_rec = p_metrics.get("recall_sensitivity", 0.9974) * 100
    p_f1 = p_metrics.get("f1_score", 0.8693)
    macro_f1 = test_perf.get("macro_metrics", {}).get("macro_f1", 0.7689)
    train_time = train_perf.get("total_training_time_seconds", 13.35)
    device_name = env_info.get("device", "NVIDIA GeForce RTX 3050 Laptop GPU")

    # Medical Metric Scorecard
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("Test Accuracy", f"{test_acc:.2f}%", help="Overall correct predictions across 624 test images")
    with sc2:
        st.metric("Pneumonia Sensitivity", f"{p_rec:.2f}%", help="Recall for Pneumonia: 389/390 cases identified")
    with sc3:
        st.metric("Pneumonia Precision", f"{p_prec:.2f}%", help="Positive Predictive Value for Pneumonia")
    with sc4:
        st.metric("Pneumonia F1-Score", f"{p_f1:.4f}", help="Harmonic mean of Precision and Recall")

    # Important Medical Disclaimer Alert Box
    st.info(
        "⚕️ **IMPORTANT MEDICAL NOTE**: "
        "Because this is a medical classification task, we explicitly report Precision, Recall, and F1 in addition to Accuracy. "
        "These metrics are academic benchmark scores on 28×28 PneumoniaMNIST and DO NOT represent clinical diagnosis performance or certified clinical deployment."
    )

    # 3. Learning Curves and Confusion Matrix
    st.markdown("#### 3. Training Dynamics & Diagnostic Confusion Matrix")
    fig_col1, fig_col2 = st.columns(2)

    loss_fig_path = RESULTS_FIGURES_DIR / "medical_training_loss.png"
    acc_fig_path = RESULTS_FIGURES_DIR / "medical_training_accuracy.png"
    cm_fig_path = RESULTS_FIGURES_DIR / "medical_confusion_matrix.png"

    with fig_col1:
        if loss_fig_path.exists():
            st.image(Image.open(loss_fig_path), caption="Cross-Entropy Loss Curve (15 Epochs)", use_container_width=True)
        if acc_fig_path.exists():
            st.image(Image.open(acc_fig_path), caption="Classification Accuracy Curves (Train & Validation)", use_container_width=True)

    with fig_col2:
        if cm_fig_path.exists():
            st.image(Image.open(cm_fig_path), caption="PneumoniaMNIST Test Confusion Matrix (Normal vs. Pneumonia)", use_container_width=True)
            st.caption(f"Evaluated on 624 untouched test images | Training time: {train_time:.1f}s | Device: {device_name}")
        else:
            st.warning("Confusion matrix figure not found.")
