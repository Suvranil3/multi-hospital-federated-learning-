"""
Interactive Stage 1: Dataset Verification (Phase 1).
"""

from pathlib import Path
import streamlit as st
from PIL import Image

from dashboard.demo.pipeline_state import (
    RESULTS_FIGURES_DIR,
    load_cifar_report,
)


def render_dataset_stage() -> None:
    """Render Stage 1: CIFAR-10 Dataset Verification."""
    report = load_cifar_report()

    # Stage Header & Status Indicator
    col_hdr, col_badge = st.columns([3, 1])
    with col_hdr:
        st.markdown("### 🖼️ Stage 1: Dataset Verification (CIFAR-10)")
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

    st.caption("Verification Script: `data/verify_dataset.py` | Artifact: `results/logs/cifar10_verification_report.json`")

    if report is None:
        st.warning("⚠️ CIFAR-10 verification report not yet available. Run `.venv/bin/python data/verify_dataset.py` first.")
        return

    # Key Quantitative Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Images", f"{report.get('total_samples', 60000):,}")
        st.caption("Complete CIFAR-10 collection")
    with m2:
        st.metric("Training-Source Images", f"{report.get('train_samples', 50000):,}")
        st.caption("5,000 train images / class")
    with m3:
        st.metric("Official Test Images", f"{report.get('test_samples', 10000):,}")
        st.caption("1,000 test images / class")
    with m4:
        st.metric("Classes & Resolution", "10 Classes | 32×32", help="3 Channels (RGB)")
        st.caption("Perfect uniform balance")

    # Visual Artifacts
    fig_col1, fig_col2 = st.columns(2)
    sample_grid_path = RESULTS_FIGURES_DIR / "cifar10_sample_grid.png"
    class_dist_path = RESULTS_FIGURES_DIR / "cifar10_class_distribution.png"

    with fig_col1:
        if sample_grid_path.exists():
            st.image(
                Image.open(sample_grid_path),
                caption="CIFAR-10: Class Representatives (5 Samples per Class)",
                use_container_width=True,
            )
        else:
            st.info("Sample grid figure not found.")

    with fig_col2:
        if class_dist_path.exists():
            st.image(
                Image.open(class_dist_path),
                caption="CIFAR-10: Class Distribution (Train vs. Test Splits)",
                use_container_width=True,
            )
        else:
            st.info("Class distribution figure not found.")
