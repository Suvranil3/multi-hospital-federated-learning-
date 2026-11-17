"""
Research Progression and Future Scope Specifications.
Reflects completed research milestones (Phases 1-8) and clearly defines future research scope (Phase 9).
"""

import streamlit as st


def render_pipeline_visualization() -> None:
    """Render the completed hospital research pipeline."""
    st.markdown("### 🗺️ Multi-Hospital Research Progression")
    st.markdown(
        r"""
        ```text
        [✓] Phase 1: PneumoniaMNIST Dataset Verification               -> STATUS: COMPLETED & AUDITED
                     ↓
        [✓] Phase 2: Centralized Medical Small CNN Baseline           -> STATUS: COMPLETED & AUDITED (81.25% Acc, 99.74% Sens)
                     ↓
        [✓] Phase 3: 10-Hospital Federated Learning (Flower + FedAvg)  -> STATUS: COMPLETED & AUDITED (86.38% Acc, 70.94% Spec)
                     ↓
        [✓] Phase 4: Non-IID Heterogeneity (Dirichlet α=1.0, 0.5, 0.1) -> STATUS: COMPLETED & AUDITED
                     ↓
        [✓] Phase 5: Adversarial Attacks (Label Flip, Updates, BD)    -> STATUS: COMPLETED & AUDITED
                     ↓
        [✓] Phase 6: Multi-Factor Detector & Clinical Trade-Off Shield -> STATUS: COMPLETED & AUDITED (0.9814 AUROC)
                     ↓
        [✓] Phase 7: Robust Aggregation Evaluation (4 Strats, 2 Seeds) -> STATUS: COMPLETED & AUDITED (72 Primary Runs)
                     ↓
        [✓] Phase 8: Blockchain Audit (2,134 Blocks) & Paper Draft     -> STATUS: COMPLETED & AUDITED
                     ↓
        [○] Phase 9: Buffer Defense (Coord Median) & N>=5 Multi-Seed   -> STATUS: FUTURE RESEARCH SCOPE
        ```
        """
    )


# Backward-compatible renderers for older imports
def render_fl_baseline_demo() -> None:
    from dashboard.demo.fl_demo import render_fl_section
    render_fl_section()


def render_noniid_demo() -> None:
    from dashboard.demo.noniid_demo import render_noniid_section
    render_noniid_section()


def render_attack_demo() -> None:
    from dashboard.demo.attack_demo import render_attack_section
    render_attack_section()


def render_detector_demo() -> None:
    from dashboard.demo.detector_demo import render_detector_section
    render_detector_section()


def render_blockchain_demo() -> None:
    from dashboard.demo.blockchain_demo import render_blockchain_section
    render_blockchain_section()
