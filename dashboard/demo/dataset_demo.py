"""
Section 2: PneumoniaMNIST Dataset Explorer
Visualizes real pediatric chest X-ray distribution, class proportions, and sample scans.
"""

from pathlib import Path
import streamlit as st
from PIL import Image
from dashboard.demo.pipeline_state import (
    RESULTS_FIGURES_DIR,
    load_medical_dataset_report,
)


def render_dataset_section() -> None:
    """Render Section 2: Dataset Explorer."""
    st.markdown("## 📊 PneumoniaMNIST Dataset Explorer")
    st.markdown(
        r"""
        **Primary Benchmark Dataset:** Pediatric Chest Radiographs from MedMNIST v3.  
        Simulating a multi-hospital consortium diagnosing pediatric pneumonia while keeping raw patient scans strictly private.
        """
    )

    data_rep = load_medical_dataset_report()

    # Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Scans", "5,856", help="All official splits combined")
    with col2:
        st.metric("Pneumonia Scans", "4,273 (72.95%)", help="Active pediatric pneumonia infections")
    with col3:
        st.metric("Normal Scans", "1,583 (27.05%)", help="Healthy pediatric chest radiographs")
    with col4:
        st.metric("Resolution", "1 × 28 × 28", help="Grayscale normalized input tensor")

    st.divider()

    # Split Distribution Table
    st.markdown("### 🗂️ 1. Official MedMNIST Benchmark Splits")
    st.caption("Preserves official benchmark splits without artificial re-splitting, ensuring strict comparability.")

    splits_data = [
        {"Split": "Train", "Total": 4708, "Normal (0)": 1214, "Pneumonia (1)": 3494, "% Pneumonia": "74.21%"},
        {"Split": "Validation", "Total": 524, "Normal (0)": 135, "Pneumonia (1)": 389, "% Pneumonia": "74.24%"},
        {"Split": "Test (Untouched)", "Total": 624, "Normal (0)": 234, "Pneumonia (1)": 390, "% Pneumonia": "62.50%"},
        {"Split": "Total Collection", "Total": 5856, "Normal (0)": 1583, "Pneumonia (1)": 4273, "% Pneumonia": "72.95%"},
    ]
    st.table(splits_data)

    st.divider()

    # Visual Artifacts
    st.markdown("### 🖼️ 2. Dataset Visual Artifacts")
    col_img1, col_img2 = st.columns(2)

    sample_grid_path = RESULTS_FIGURES_DIR / "pneumonia_sample_grid.png"
    dist_chart_path = RESULTS_FIGURES_DIR / "pneumonia_class_distribution.png"

    with col_img1:
        st.markdown("#### Representative Chest X-Ray Grid")
        if sample_grid_path.exists():
            st.image(Image.open(sample_grid_path), caption="PneumoniaMNIST Sample Grid (Normal vs Pneumonia)", use_container_width=True)
        else:
            st.info("Sample grid figure not found.")

    with col_img2:
        st.markdown("#### Class Distribution by Split")
        if dist_chart_path.exists():
            st.image(Image.open(dist_chart_path), caption="Class Distribution across Benchmark Splits", use_container_width=True)
        else:
            st.info("Class distribution figure not found.")

    st.divider()

    # Clinical Context & Medical Trade-off
    st.markdown("### 🩺 3. Clinical Diagnostic Realities")
    c1, c2 = st.columns(2)

    with c1:
        st.warning("#### ⚠️ High Clinical Cost of False Negatives")
        st.markdown(
        r"""
            - In pediatric pulmonary triage, **missing an active pneumonia infection (False Negative)** can lead to respiratory failure or septic shock.
            - Therefore, clinical machine learning prioritizes **Sensitivity / Recall $\ge 95\%$** over raw accuracy.
            - Even if an algorithm generates some false alarms (False Positives), radiologists can verify them rapidly; a missed case is fatal.
            """
        )

    with c2:
        st.info("#### 🧬 Benchmark Boundaries & Ethical Disclosure")
        st.markdown(
        r"""
            - **Benchmark Resolution:** Downsampled to $28 \times 28$ grayscale for high-throughput multi-hospital federated simulation on academic hardware.
            - **Simulated Cohort:** 10 simulated client nodes represent hospital departments, not physical clinical servers.
            - **Research Disclaimer:** Developed strictly for Byzantine-robust federated learning research; not certified for clinical diagnostic triage.
            """
        )

    if data_rep:
        with st.expander("📄 View Full JSON Verification Report"):
            st.json(data_rep)


# Backward-compatible alias
render_dataset_stage = render_dataset_section
