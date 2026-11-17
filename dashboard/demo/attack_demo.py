"""
Section 6: Adversarial Attack Suite (Phase 5)
Evaluates targeted label flipping, update inversion/scaling, and backdoor trigger injection under IID and Non-IID regimes.
"""

from pathlib import Path
import streamlit as st
from PIL import Image
from dashboard.demo.pipeline_state import (
    RESULTS_FIGURES_DIR,
    load_attack_comparison,
    load_single_attack_log,
)


def render_attack_section() -> None:
    """Render Section 6: Attack Simulation Results."""
    st.markdown("## ⚔️ Adversarial Attack Simulation Results")
    st.markdown(
        r"""
        **Phase 5 Execution:** Simulating Byzantine attacks across 10 simulated hospitals with unweighted FedAvg.  
        Quantifies diagnostic degradation, missed pneumonia surge (+757%), and backdoor watermark exploitation.
        """
    )

    atk_comp = load_attack_comparison()

    # Executive Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Attacks Evaluated", "4 Families", help="Label flipping, scaling, inversion, backdoor triggers")
    with col2:
        st.metric("Missed Pneumonia Surge", "+757%", help="FN jumped from 7 to 60 under M=2 label flipping")
    with col3:
        st.metric("Backdoor Trigger ASR", "8.46%", delta="32.5× amplification", help="ASR jumped from 0.26% to 8.46% under Non-IID")
    with col4:
        st.metric("Poisoned Hospital Nodes", "M ∈ {1, 2}", help="10% to 20% of network compromised")

    st.divider()

    # Full Master Attack Comparison Table
    st.markdown("### 📋 1. Comprehensive Attack Benchmark Suite")
    st.caption("Evaluated across 20 communication rounds with unweighted FedAvg baseline.")

    attack_table = [
        {"Exp ID": "EXP-MED-006", "Regime": "IID", "Scenario": "Honest Control (M=0)", "Test Acc": "86.38%", "Sensitivity": "95.64%", "FN Cases": 17, "Backdoor ASR": "0.51%", "Impact Note": "Clean IID baseline"},
        {"Exp ID": "EXP-MED-007", "Regime": "Non-IID", "Scenario": "Honest Control (α=0.5, M=0)", "Test Acc": "80.45%", "Sensitivity": "98.46%", "FN Cases": 6, "Backdoor ASR": "0.26%", "Impact Note": "Clean Non-IID baseline"},
        {"Exp ID": "EXP-MED-008", "Regime": "IID", "Scenario": "Label Flipping (M=1: H0)", "Test Acc": "86.38%", "Sensitivity": "96.15%", "FN Cases": 15, "Backdoor ASR": "N/A", "Impact Note": "Absorbed by honest majority"},
        {"Exp ID": "EXP-MED-009", "Regime": "IID", "Scenario": "Label Flipping (M=2: H0, H1)", "Test Acc": "86.54%", "Sensitivity": "84.62%", "FN Cases": 60, "Backdoor ASR": "N/A", "Impact Note": "⚠️ Missed cases surge +757%"},
        {"Exp ID": "EXP-MED-010", "Regime": "Non-IID", "Scenario": "Label Flipping (α=0.5, M=1: H0)", "Test Acc": "85.90%", "Sensitivity": "97.18%", "FN Cases": 11, "Backdoor ASR": "N/A", "Impact Note": "Missed cases up 175%"},
        {"Exp ID": "EXP-MED-011", "Regime": "IID", "Scenario": "Update Inversion (γ = -1.0)", "Test Acc": "84.94%", "Sensitivity": "97.95%", "FN Cases": 8, "Backdoor ASR": "N/A", "Impact Note": "Stealthy convergence drag"},
        {"Exp ID": "EXP-MED-012", "Regime": "IID", "Scenario": "Update Scaling (γ = +10.0)", "Test Acc": "86.70%", "Sensitivity": "97.69%", "FN Cases": 9, "Backdoor ASR": "N/A", "Impact Note": "Gradient magnitude explosion"},
        {"Exp ID": "EXP-MED-013", "Regime": "IID", "Scenario": "Backdoor Watermark (M=1: H0)", "Test Acc": "86.06%", "Sensitivity": "97.44%", "FN Cases": 10, "Backdoor ASR": "7.44%", "Impact Note": "14.6× jump in backdoor success"},
        {"Exp ID": "EXP-MED-014", "Regime": "Non-IID", "Scenario": "Backdoor Watermark (α=0.5, M=1)", "Test Acc": "84.46%", "Sensitivity": "97.95%", "FN Cases": 8, "Backdoor ASR": "8.46%", "Impact Note": "⚠️ 32.5× jump in backdoor success"},
    ]
    st.table(attack_table)

    comp_chart_path = RESULTS_FIGURES_DIR / "phase5_attack_comparison.png"
    if comp_chart_path.exists():
        st.image(Image.open(comp_chart_path), caption="Phase 5 Attack Suite Performance & Impact Comparison", use_container_width=True)

    st.divider()

    # Interactive Attack Inspection
    st.markdown("### 🔍 2. Individual Attack Deep-Dive Inspector")
    selected_attack = st.selectbox(
        "Select Attack Scenario to Inspect:",
        [
            "EXP-MED-008: Label Flipping (IID, M=1)",
            "EXP-MED-009: Label Flipping (IID, M=2 - Severe)",
            "EXP-MED-010: Label Flipping (Non-IID α=0.5, M=1)",
            "EXP-MED-011: Update Inversion (IID, γ=-1)",
            "EXP-MED-012: Update Scaling (IID, γ=+10)",
            "EXP-MED-013: Backdoor Watermark (IID, M=1)",
            "EXP-MED-014: Backdoor Watermark (Non-IID α=0.5, M=1)",
        ]
    )

    attack_files_map = {
        "EXP-MED-008: Label Flipping (IID, M=1)": ("exp_med_008_atk_lf_iid_m1_confusion_matrix.png", "exp_med_008_atk_lf_iid_m1_training_curves.png", "exp_med_008_atk_lf_iid_m1_results.json"),
        "EXP-MED-009: Label Flipping (IID, M=2 - Severe)": ("exp_med_009_atk_lf_iid_m2_confusion_matrix.png", "exp_med_009_atk_lf_iid_m2_training_curves.png", "exp_med_009_atk_lf_iid_m2_results.json"),
        "EXP-MED-010: Label Flipping (Non-IID α=0.5, M=1)": ("exp_med_010_atk_lf_noniid_m1_confusion_matrix.png", "exp_med_010_atk_lf_noniid_m1_training_curves.png", "exp_med_010_atk_lf_noniid_m1_results.json"),
        "EXP-MED-011: Update Inversion (IID, γ=-1)": ("exp_med_011_atk_mp_inv_iid_m1_confusion_matrix.png", "exp_med_011_atk_mp_inv_iid_m1_training_curves.png", "exp_med_011_atk_mp_inv_iid_m1_results.json"),
        "EXP-MED-012: Update Scaling (IID, γ=+10)": ("exp_med_012_atk_mp_scale_iid_m1_confusion_matrix.png", "exp_med_012_atk_mp_scale_iid_m1_training_curves.png", "exp_med_012_atk_mp_scale_iid_m1_results.json"),
        "EXP-MED-013: Backdoor Watermark (IID, M=1)": ("exp_med_013_atk_bd_iid_m1_confusion_matrix.png", "exp_med_013_atk_bd_iid_m1_training_curves.png", "exp_med_013_atk_bd_iid_m1_results.json"),
        "EXP-MED-014: Backdoor Watermark (Non-IID α=0.5, M=1)": ("exp_med_014_atk_bd_noniid_m1_confusion_matrix.png", "exp_med_014_atk_bd_noniid_m1_training_curves.png", "exp_med_014_atk_bd_noniid_m1_results.json"),
    }

    cm_file, curve_file, json_file = attack_files_map[selected_attack]
    c_atk_cm, c_atk_curves = st.columns(2)

    with c_atk_cm:
        cm_p = RESULTS_FIGURES_DIR / cm_file
        if cm_p.exists():
            st.image(Image.open(cm_p), caption=f"Confusion Matrix: {selected_attack}", use_container_width=True)
        else:
            st.info("Confusion matrix figure not found.")

    with c_atk_curves:
        curve_p = RESULTS_FIGURES_DIR / curve_file
        if curve_p.exists():
            st.image(Image.open(curve_p), caption=f"Training Curves: {selected_attack}", use_container_width=True)
        else:
            st.info("Training curves figure not found.")

    single_atk_data = load_single_attack_log(json_file)
    if single_atk_data:
        with st.expander("📄 View JSON Simulation Record"):
            st.json(single_atk_data)
