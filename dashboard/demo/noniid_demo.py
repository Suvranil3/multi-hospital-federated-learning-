"""
Section 5: Non-IID Clinical Heterogeneity & Case-Mix Skew
Explores Dirichlet distribution modeling across 10 simulated hospitals and the negative cosine alignment phenomenon.
"""

from pathlib import Path
import streamlit as st
from PIL import Image
from dashboard.demo.pipeline_state import (
    RESULTS_FIGURES_DIR,
    load_non_iid_comparison,
    load_non_iid_result,
)


def render_noniid_section() -> None:
    """Render Section 5: Non-IID Hospital Distribution."""
    st.markdown("## 📊 Non-IID Hospital Data Heterogeneity")
    st.markdown(
        r"""
        **Phase 4 Execution:** Modeling realistic clinical case-mix specialization across 10 hospitals via constrained Dirichlet distributions ($\alpha \in \{1.0, 0.5, 0.1\}$).  
        Reveals how honest clinical specialization causes directional gradient divergence ($\cos \theta < 0$).
        """
    )

    comp_res = load_non_iid_comparison()

    # Skew Selector
    st.markdown("### 🎛️ 1. Interactive Dirichlet Heterogeneity Explorer")
    alpha_choice = st.radio(
        "Select Dirichlet Concentration Parameter (α):",
        ["α = 1.0 (Mild Skew)", "α = 0.5 (Moderate Specialization)", "α = 0.1 (Extreme Specialization)"],
        index=1,
        horizontal=True,
    )

    alpha_map = {
        "α = 1.0 (Mild Skew)": ("10", "1.0", "hospital_distribution_alpha10.png", "federated_non_iid_alpha10_confusion_matrix.png"),
        "α = 0.5 (Moderate Specialization)": ("05", "0.5", "hospital_distribution_alpha05.png", "federated_non_iid_alpha05_confusion_matrix.png"),
        "α = 0.1 (Extreme Specialization)": ("01", "0.1", "hospital_distribution_alpha01.png", "federated_non_iid_alpha01_confusion_matrix.png"),
    }
    alpha_code, alpha_val, dist_img_name, cm_img_name = alpha_map[alpha_choice]
    single_res = load_non_iid_result(alpha_code)

    if single_res:
        m = single_res.get("test_metrics", {})
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test Accuracy", f"{m.get('test_accuracy', 0):.2f}%")
        with col2:
            st.metric("Sensitivity (Recall)", f"{m.get('sensitivity', 0):.2f}%")
        with col3:
            st.metric("Specificity", f"{m.get('specificity', 0):.2f}%")
        with col4:
            st.metric("Missed Cases (FN)", f"{m.get('confusion_matrix', {}).get('fn', 0)} / 390")

    st.divider()

    # Visual Artifacts for Selected Alpha
    c_dist, c_cm = st.columns([3, 2])
    dist_path = RESULTS_FIGURES_DIR / dist_img_name
    cm_path = RESULTS_FIGURES_DIR / cm_img_name

    with c_dist:
        st.markdown(f"#### Hospital Class Distribution (α = {alpha_val})")
        if dist_path.exists():
            st.image(Image.open(dist_path), caption=f"10-Hospital Case-Mix Proportions (α = {alpha_val})", use_container_width=True)
        else:
            st.info("Distribution plot not found.")

    with c_cm:
        st.markdown(f"#### Test Confusion Matrix (α = {alpha_val})")
        if cm_path.exists():
            st.image(Image.open(cm_path), caption=f"Confusion Matrix (α = {alpha_val})", use_container_width=True)
        else:
            st.info("Confusion matrix not found.")

    st.divider()

    # Heterogeneity Summary Table
    st.markdown("### 📋 2. Comprehensive Heterogeneity Comparison")
    summary_data = [
        {"Condition": "IID Uniform Baseline", "Dirichlet α": "None", "Test Acc": "86.38%", "Sensitivity": "95.64%", "Specificity": "70.94%", "Macro F1": "0.8415", "Missed Pneumonia": "17 / 390"},
        {"Condition": "Mild Heterogeneity", "Dirichlet α": "1.0", "Test Acc": "79.81%", "Sensitivity": "98.97%", "Specificity": "47.86%", "Macro F1": "0.7498", "Missed Pneumonia": "4 / 390"},
        {"Condition": "Moderate Heterogeneity", "Dirichlet α": "0.5", "Test Acc": "80.45%", "Sensitivity": "98.46%", "Specificity": "50.43%", "Macro F1": "0.7611", "Missed Pneumonia": "6 / 390"},
        {"Condition": "Extreme Heterogeneity", "Dirichlet α": "0.1", "Test Acc": "85.10%", "Sensitivity": "91.79%", "Specificity": "73.93%", "Macro F1": "0.8366", "Missed Pneumonia": "32 / 390"},
    ]
    st.table(summary_data)

    comp_chart_path = RESULTS_FIGURES_DIR / "iid_vs_non_iid_heterogeneity_comparison.png"
    if comp_chart_path.exists():
        st.image(Image.open(comp_chart_path), caption="IID vs. Non-IID Diagnostic Trajectory Comparison", use_container_width=True)

    st.divider()

    # The Negative Cosine Discovery
    st.markdown("### 🚨 3. The Negative Cosine Alignment Discovery")
    st.warning(
        r"""
        #### 📉 Why Classical Byzantine Detectors Fail in Healthcare:
        - In extreme non-IID settings ($\alpha = 0.1$), specialized clinics naturally experience negative cosine alignment ($\cos \theta < 0$, down to **$-0.8411$**).
        - **Reason:** A specialized hospital treating 95%+ Normal scans produces gradient vectors that point in the opposite direction of a general hospital treating 95%+ Pneumonia scans.
        - **Classical Mistake:** Standard Byzantine defenses (e.g., Krum, Cosine-based anomaly filters) assume honest updates always align positively with the consensus ($\cos \theta > 0$). They mistake legitimate medical specialization for a malicious attack, disenfranchising specialized centers.
        - **Our Solution:** Symmetrical Clinical Trade-Off Shield (Phase 6) that verifies counterfactual validation loss on held-out data before penalizing divergent gradients.
        """
    )
