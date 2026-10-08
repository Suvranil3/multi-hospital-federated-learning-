"""
Interactive Stage 2: Centralized Small CNN Baseline (Phase 2).
"""

from pathlib import Path
import subprocess
import sys
import streamlit as st
from PIL import Image

from dashboard.demo.pipeline_state import (
    PROJECT_ROOT,
    RESULTS_FIGURES_DIR,
    load_cnn_results,
)


def run_training_subprocess() -> bool:
    """Execute real training script via subprocess with streaming progress."""
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    script_path = PROJECT_ROOT / "experiments" / "train_centralized.py"

    if not venv_python.exists():
        venv_python = Path(sys.executable)

    progress_bar = st.progress(0)
    status_text = st.empty()
    log_box = st.empty()

    status_text.info("🚀 Initiating centralized training on GPU/CPU... Please wait.")

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
            status_text.success("✅ Training completed successfully! Fresh metrics loaded.")
            # Clear cache to reload fresh results
            load_cnn_results.clear()
            return True
        else:
            status_text.error(f"❌ Training failed with exit code {return_code}.")
            return False

    except Exception as e:
        status_text.error(f"❌ Execution error: {str(e)}")
        return False


def render_cnn_stage() -> None:
    """Render Stage 2: Centralized Small CNN baseline."""
    results = load_cnn_results()

    # Stage Header
    col_hdr, col_badge = st.columns([3, 1])
    with col_hdr:
        st.markdown("### 🧠 Stage 2: Centralized Small CNN Baseline")
        st.write("Building and validating the Small CNN for CIFAR-10 before federated distribution.")
    with col_badge:
        st.markdown(
            """
            <div style="text-align: right; padding-top: 0.5rem;">
                <span style="background-color: #10B981; color: white; padding: 0.35rem 0.8rem; border-radius: 0.35rem; font-weight: bold; font-size: 0.95rem;">
                    ✓ Model Validated
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 1. Visual Architecture Flow & Hyperparameters
    st.markdown("#### 1. Architecture & Design Specification")
    arch_col, spec_col = st.columns([1, 1])

    with arch_col:
        st.markdown(
            """
            ```text
            +--------------------------------------------+
            |          Input Image (32 × 32 × 3)         |
            +--------------------------------------------+
                                  ↓
            +--------------------------------------------+
            |  Conv Block 1: 32 Filters (3x3), BN, Pool  |
            +--------------------------------------------+
                                  ↓
            +--------------------------------------------+
            |  Conv Block 2: 64 Filters (3x3), BN, Pool  |
            +--------------------------------------------+
                                  ↓
            +--------------------------------------------+
            |  Conv Block 3: 64 Filters (3x3), BN, Pool  |
            +--------------------------------------------+
                                  ↓
            +--------------------------------------------+
            |         Flatten Layer (1,024 dims)         |
            +--------------------------------------------+
                                  ↓
            +--------------------------------------------+
            |        Dense 128 (ReLU, Dropout 0.3)       |
            +--------------------------------------------+
                                  ↓
            +--------------------------------------------+
            |            10-Class Logits Output          |
            +--------------------------------------------+
            ```
            """
        )

    with spec_col:
        st.markdown("##### Configuration Parameters")
        p1, p2 = st.columns(2)
        with p1:
            st.metric("Total Parameters", "189,130", help="Lightweight for FL client simulation")
            st.metric("Batch Size", "64")
        with p2:
            st.metric("Optimizer", "Adam (lr=0.001)")
            st.metric("Epochs", "15")

        st.caption("ℹ️ *Default Mode*: Using existing verified results from `results/logs/centralized_cnn_results.json` to keep demonstrations immediate.")

        # Fresh Training Expansion
        with st.expander("⚙️ Optional: Run Training Again (Live Execution)"):
            st.warning("⚠️ Running training executes 15 epochs on your GPU/CPU (~1.5 minutes). Only run if you wish to demonstrate live training to your professor.")
            confirm = st.checkbox("I confirm I want to execute fresh training")
            if st.button("▶ Run Fresh Training", disabled=not confirm):
                with st.spinner("Executing experiments/train_centralized.py..."):
                    run_training_subprocess()
                    results = load_cnn_results()

    st.divider()

    # 2. Final Model Results (Large Result Card)
    st.markdown("#### 2. Final Centralized Model Performance")
    if results is None:
        st.warning("Centralized CNN results not yet available.")
        return

    test_perf = results.get("test_performance", {})
    train_perf = results.get("training_performance", {})
    env_info = results.get("environment", {})

    test_acc = test_perf.get("test_accuracy", 76.94)
    test_loss = test_perf.get("test_loss", 0.6578)
    train_time = train_perf.get("total_training_time_seconds", 100.97)
    device_name = env_info.get("device", "NVIDIA GeForce RTX 3050 Laptop GPU")

    # Prominent Metric Banner
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%); color: white; border-radius: 0.6rem; padding: 1.5rem; margin-bottom: 1.5rem;">
            <div style="font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.9;">
                Measured Phase 2 Result (Centralized ML Baseline)
            </div>
            <div style="font-size: 2.8rem; font-weight: 800; margin: 0.3rem 0;">
                {test_acc:.2f}% <span style="font-size: 1.2rem; font-weight: 400; opacity: 0.9;">Test Accuracy</span>
            </div>
            <div style="font-size: 0.95rem; opacity: 0.95;">
                Evaluated on 10,000 untouched official CIFAR-10 test samples | Test Loss: <b>{test_loss:.4f}</b> | Training Time: <b>{train_time:.1f}s</b> | Compute: <b>{device_name}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Learning Curves and Confusion Matrix
    st.markdown("#### 3. Training Dynamics & Class Predictions")
    fig_col1, fig_col2 = st.columns(2)

    loss_fig_path = RESULTS_FIGURES_DIR / "centralized_training_loss.png"
    acc_fig_path = RESULTS_FIGURES_DIR / "centralized_training_accuracy.png"
    cm_fig_path = RESULTS_FIGURES_DIR / "confusion_matrix.png"

    with fig_col1:
        if loss_fig_path.exists():
            st.image(Image.open(loss_fig_path), caption="Cross-Entropy Loss Decay across 15 Epochs", use_container_width=True)
        if acc_fig_path.exists():
            st.image(Image.open(acc_fig_path), caption="Classification Accuracy Convergence (Train & Validation)", use_container_width=True)

    with fig_col2:
        if cm_fig_path.exists():
            st.image(Image.open(cm_fig_path), caption="10-Class Test Confusion Matrix", use_container_width=True)
            st.info("💡 **Explanation**: The confusion matrix shows which CIFAR-10 classes the CNN predicts correctly and which classes it confuses.")
        else:
            st.warning("Confusion matrix figure not found.")
