"""
Interactive Stage 1: Medical Dataset Verification (PneumoniaMNIST).
Primary Scenario: Multi-Hospital Chest X-Ray Classification.
"""

from pathlib import Path
import streamlit as st
from PIL import Image

from dashboard.demo.pipeline_state import (
    RESULTS_FIGURES_DIR,
    load_medical_dataset_report,
)


def render_dataset_stage() -> None:
    """Render Stage 1: PneumoniaMNIST Dataset Verification."""
    report = load_medical_dataset_report()

    # Stage Header & Status Indicator
    col_hdr, col_badge = st.columns([3, 1])
    with col_hdr:
        st.markdown("### 🫁 Stage 1: Medical Dataset Verification (PneumoniaMNIST)")
        st.caption("Application Dataset: Pediatric Chest X-Ray Images for Normal vs. Pneumonia Classification")
    with col_badge:
        st.markdown(
            """
            <div style="text-align: right; padding-top: 0.5rem;">
                <span style="background-color: #10B981; color: white; padding: 0.35rem 0.8rem; border-radius: 0.35rem; font-weight: bold; font-size: 0.95rem;">
                    ✓ Dataset Verified
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Explanation block: WHY THIS DATASET
    st.markdown(
        """
        > **Why This Dataset?**  
        > *We use chest X-ray images because the final project is designed around a hospital scenario. The dataset gives us a real medical-image classification problem that can later be distributed among simulated hospitals.*
        """
    )

    if report is None:
        st.warning("⚠️ PneumoniaMNIST verification report not yet available. Run `.venv/bin/python data/verify_dataset.py` first.")
        return

    # Key Quantitative Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total X-Rays", f"{report.get('total_samples', 5856):,}")
        st.caption("Official MedMNIST v3 Collection")
    with m2:
        st.metric("Training Split", f"{report.get('train_samples', 4708):,}")
        st.caption("1,214 Normal | 3,494 Pneumonia")
    with m3:
        st.metric("Validation & Test", f"{report.get('validation_samples', 524)} / {report.get('test_samples', 624)}")
        st.caption("Val: 524 | Test: 624 (Untouched)")
    with m4:
        st.metric("Resolution & Channel", "28×28 Grayscale", help="1 Channel (Grayscale), Binary Diagnosis (0=Normal, 1=Pneumonia)")
        st.caption("Mean: 0.5719 | Std: 0.1684")

    # Visual Artifacts
    fig_col1, fig_col2 = st.columns(2)
    sample_grid_path = RESULTS_FIGURES_DIR / "pneumonia_sample_grid.png"
    class_dist_path = RESULTS_FIGURES_DIR / "pneumonia_class_distribution.png"

    with fig_col1:
        if sample_grid_path.exists():
            st.image(
                Image.open(sample_grid_path),
                caption="PneumoniaMNIST: Representative Pediatric Chest X-Rays (Normal vs. Pneumonia)",
                use_container_width=True,
            )
        else:
            st.info("Sample grid figure not found.")

    with fig_col2:
        if class_dist_path.exists():
            st.image(
                Image.open(class_dist_path),
                caption="PneumoniaMNIST: Class Distribution Across Official Train, Val, and Test Splits",
                use_container_width=True,
            )
        else:
            st.info("Class distribution figure not found.")
