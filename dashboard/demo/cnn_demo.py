"""
Section 3: Centralized Medical Small CNN Baseline
Presents architecture details, epoch trajectories, and diagnostic metrics on untouched test data.
"""

from pathlib import Path
import streamlit as st
from PIL import Image
from dashboard.demo.pipeline_state import (
    RESULTS_FIGURES_DIR,
    load_medical_cnn_results,
)


def render_cnn_section() -> None:
    """Render Section 3: Centralized CNN Baseline."""
    st.markdown("## 🧠 Centralized Medical Small CNN Baseline")
    st.markdown(
        r"""
        **Phase MEDICAL-02 Execution:** Centralized training on all 4,708 local chest X-rays.  
        Establishes the performance benchmark for comparison against decentralized multi-hospital federated learning.
        """
    )

    results = load_medical_cnn_results()

    # Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Test Accuracy", "81.25%", help="507 out of 624 correct diagnoses on untouched test split")
    with col2:
        st.metric("Pneumonia Recall", "99.74%", help="389 / 390 active pneumonia cases correctly flagged")
    with col3:
        st.metric("Normal Specificity", "50.43%", help="118 / 234 normal scans correctly classified")
    with col4:
        st.metric("Model Size", "420,802 Params", help="SmallCNN parameter footprint: ~1.61 MB")

    st.divider()

    # Diagnostic Performance Scorecard
    st.markdown("### 📋 1. Diagnostic Scorecard on Untouched Test Set (624 Images)")
    scorecard = [
        {"Metric": "Sensitivity / Recall", "Pneumonia (Class 1)": "99.74% (389/390)", "Normal (Class 0)": "50.43% (118/234)", "Macro Average": "75.09%"},
        {"Metric": "Precision", "Pneumonia (Class 1)": "77.03% (389/505)", "Normal (Class 0)": "99.16% (118/119)", "Macro Average": "88.09%"},
        {"Metric": "F1-Score", "Pneumonia (Class 1)": "0.8693", "Normal (Class 0)": "0.6686", "Macro Average": "0.7689"},
    ]
    st.table(scorecard)

    st.divider()

    # Confusion Matrix Breakdown
    st.markdown("### 🎯 2. Confusion Matrix & Clinical Safety Analysis")
    c_cm_img, c_cm_txt = st.columns([1, 1])

    cm_path = RESULTS_FIGURES_DIR / "medical_confusion_matrix.png"

    with c_cm_img:
        if cm_path.exists():
            st.image(Image.open(cm_path), caption="Centralized Baseline Test Confusion Matrix", use_container_width=True)
        else:
            st.info("Confusion matrix figure not found.")

    with c_cm_txt:
        st.markdown(
        r"""
            #### Clinical Safety Breakdown:
            - **True Positives (TP = 389):** Successfully diagnosed pneumonia patients.
            - **False Negatives (FN = 1):** Missed pneumonia cases. Achieving a **0.26% false negative rate** meets clinical safety screening priorities.
            - **True Negatives (TN = 118):** Correctly confirmed healthy children.
            - **False Positives (FP = 116):** Healthy children flagged as potentially sick. In clinical triage, false alarms trigger confirmatory radiologist review without risking patient life.
            """
        )
        st.info("The high pneumonia sensitivity (99.74%) reflects natural optimization towards the majority pathology without artificial cost-weighting.")

    st.divider()

    # Training Curves
    st.markdown("### 📈 3. Training & Validation Trajectory (15 Epochs)")
    c_loss, c_acc = st.columns(2)

    loss_curve_path = RESULTS_FIGURES_DIR / "medical_training_loss.png"
    acc_curve_path = RESULTS_FIGURES_DIR / "medical_training_accuracy.png"

    with c_loss:
        if loss_curve_path.exists():
            st.image(Image.open(loss_curve_path), caption="Training & Validation Loss Trajectory", use_container_width=True)
        else:
            st.info("Loss curve figure not found.")

    with c_acc:
        if acc_curve_path.exists():
            st.image(Image.open(acc_curve_path), caption="Training & Validation Accuracy Trajectory", use_container_width=True)
        else:
            st.info("Accuracy curve figure not found.")

    st.divider()

    # SmallCNN Architecture Rationale
    st.markdown("### 🏗️ 4. SmallCNN Architecture Specifications")
    st.markdown(
        r"""
        - **Input Layer:** $1 \times 28 \times 28$ grayscale normalized chest radiograph.
        - **Conv Block 1:** Conv2D (32 filters, $3\times3$) $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool ($2\times2$) $\to$ Dropout (0.25).
        - **Conv Block 2:** Conv2D (64 filters, $3\times3$) $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool ($2\times2$) $\to$ Dropout (0.25).
        - **Dense Head:** Linear (1,600 $\to$ 256) $\to$ ReLU $\to$ Dropout (0.50) $\to$ Linear (256 $\to$ 2 logits).
        - **Total Parameters:** **420,802 trainable weights** (~1.61 MB).
        - **Why SmallCNN?** Compact parameter footprint ensures rapid local training on commodity hospital hardware, minimizes communication bandwidth during federated synchronization, and avoids overfitting on $28\times28$ inputs.
        """
    )

    if results:
        with st.expander("📄 View Full JSON Results"):
            st.json(results)


# Backward-compatible alias
render_cnn_stage = render_cnn_section
