"""
Section 10: Final Experimental Scorecard & Academic Synthesis
Presents the complete 72-cell primary scorecard, academic paper documentation suite, and future research scope.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
from dashboard.demo.pipeline_state import (
    load_phase7_reduced_registry,
    load_phase7_supplementary_registry,
)


def render_scorecard_section() -> None:
    """Render Section 10: Final Experimental Scorecard."""
    st.markdown("## 📋 Final Experimental Scorecard & Academic Synthesis")
    st.markdown(
        r"""
        **Authoritative Project Evaluation Scorecard:** Synthesizes all 72 primary Phase 7 simulation runs (Seeds 211, 212) alongside historical supplementary runs.  
        Links to complete academic research documentation and formal peer-review audits.
        """
    )

    reg_primary = load_phase7_reduced_registry()
    reg_supp = load_phase7_supplementary_registry()

    # Executive Counts
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Primary Matrix Cells", "72 Runs", help="4 strategies × 9 conditions × 2 seeds")
    with col2:
        st.metric("Supplementary Runs", "24 Runs", help="Historical exploratory runs (Seeds 101, 102, 103)")
    with col3:
        st.metric("Total Unique Runs", "96 Simulations", help="72 primary + 24 supplementary")
    with col4:
        st.metric("Physical Launches", "97 Launches", help="96 successful + 1 Batch 2 recovered crash")

    st.divider()

    # Full Filterable 72-Cell Data Table
    st.markdown("### 📊 1. Master 72-Cell Primary Scorecard Explorer")
    st.caption("Filter and sort the complete evaluation matrix on the untouched test split (624 images: 234 Normal, 390 Pneumonia).")

    if reg_primary:
        rows = []
        for exp_id, data in reg_primary.items():
            m = data.get("test_metrics", {})
            cm = m.get("confusion_matrix", {})
            rows.append({
                "Experiment ID": exp_id,
                "Condition": data.get("condition_id"),
                "Strategy": data.get("aggregation_variant"),
                "Seed": data.get("seed"),
                "Test Accuracy (%)": m.get("test_accuracy", 0.0),
                "Specificity (%)": m.get("specificity", 0.0),
                "Sensitivity (%)": m.get("sensitivity", 0.0),
                "Macro F1 (%)": m.get("macro_f1", 0.0),
                "Backdoor ASR (%)": f"{m.get('backdoor_asr'):.2f}%" if m.get("backdoor_asr") is not None else "—",
                "TN": cm.get("tn", 0),
                "FP": cm.get("fp", 0),
                "FN": cm.get("fn", 0),
                "TP": cm.get("tp", 0),
                "Block #": data.get("blockchain_block"),
            })
        df_all = pd.DataFrame(rows)

        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            all_conds = ["All Conditions"] + sorted(df_all["Condition"].unique().tolist())
            filt_cond = st.selectbox("Filter Condition:", all_conds)
        with f_col2:
            all_strats = ["All Strategies"] + sorted(df_all["Strategy"].unique().tolist())
            filt_strat = st.selectbox("Filter Strategy:", all_strats)
        with f_col3:
            all_seeds = ["All Seeds"] + [str(s) for s in sorted(df_all["Seed"].unique().tolist())]
            filt_seed = st.selectbox("Filter Seed:", all_seeds)

        df_filtered = df_all.copy()
        if filt_cond != "All Conditions":
            df_filtered = df_filtered[df_filtered["Condition"] == filt_cond]
        if filt_strat != "All Strategies":
            df_filtered = df_filtered[df_filtered["Strategy"] == filt_strat]
        if filt_seed != "All Seeds":
            df_filtered = df_filtered[df_filtered["Seed"] == int(filt_seed)]

        st.dataframe(df_filtered, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(df_filtered)} of {len(df_all)} experiments.")

    st.divider()

    # Academic Documentation Suite
    st.markdown("### 📚 2. Academic Research Documentation Suite")
    st.markdown(
        r"""
        The project findings and methodology have been synthesized into a comprehensive academic documentation suite:
        """
    )

    doc_suite = [
        {"Document": "Academic Paper Draft", "File": "docs/RESEARCH_PAPER_DRAFT.md", "Description": "Full 19-section research manuscript (~5,500 words) formatted for IEEE/Springer."},
        {"Document": "Literature Review", "File": "docs/LITERATURE_REVIEW.md", "Description": "Survey of 20 verified foundational papers with DOIs across FL, Byzantine defense, and medical AI."},
        {"Document": "Research Contributions", "File": "docs/RESEARCH_CONTRIBUTIONS.md", "Description": "Formal statement of clinical dilemma, 4 literature gaps, and 6 core contributions."},
        {"Document": "Phase 7 Technical Synthesis", "File": "docs/PHASE7_ROBUST_AGGREGATION_SYNTHESIS.md", "Description": "Exhaustive 4,100+ word synthesis of all 72 primary runs and failure mode telemetry."},
        {"Document": "Reproducibility Checklist", "File": "docs/PAPER_REPRODUCIBILITY_CHECKLIST.md", "Description": "Exact specifications of seeds, splits, hyperparameters, and environment versions."},
        {"Document": "Peer-Review Audit", "File": "docs/PAPER_REVIEW_AUDIT.md", "Description": "Critical peer-review evaluation (Clarity: 9/10, Rigor: 9.5/10, Reproducibility: 9.5/10)."},
    ]
    st.table(doc_suite)

    st.divider()

    # Scientific Boundaries & Future Research Scope
    st.markdown("### 🔬 3. Scientific Disclosures & Future Research Scope")
    sc1, sc2 = st.columns(2)

    with sc1:
        st.info("#### 🔍 Epistemic Boundaries & Honest Disclosures")
        st.markdown(
        r"""
            - **Sample Size ($N=2$):** Phase 7 evaluated across two seeds (211, 212) as an exploratory benchmark; we do not claim hypothesis-tested statistical significance until $N \ge 5$ scaling is performed.
            - **Simulated Infrastructure:** Clients are 10 partitioned subsets of PneumoniaMNIST on academic hardware, not physical hospital servers.
            - **Non-Clinical Deployment:** Models are developed for federated security research and are not certified for clinical diagnostic triage.
            """
        )

    with sc2:
        st.success("#### 🚀 Phase 9: Future Research Expansion")
        st.markdown(
        r"""
            - **Coordinate-Wise Median Buffer Aggregation:** Implementing median aggregation over BatchNorm buffers to prevent the 100% single-class collapse observed in buffer poisoning.
            - **Multi-Seed Statistical Expansion ($N \ge 5$):** Executing additional seeds (213, 214, 215) on focal regimes to calculate two-tailed p-values for formal journal submission.
            - **Adaptive Skew-Aware Thresholding:** Dynamically parameterizing cosine acceptance thresholds as a function of measured local data imbalance.
            """
        )
