"""
Section 7: Multi-Factor Malicious Participant Detector (Phase 6)
Demonstrates 0.9814 AUROC, 0.00% False Malicious Rate across Dirichlet skews, and Symmetrical Clinical Trade-Off Shield.
"""

from pathlib import Path
import streamlit as st
from PIL import Image
from dashboard.demo.pipeline_state import (
    PHASE6_DETECTOR_DIR,
    load_detector_results,
    load_detector_thresholds,
    load_honest_false_positive_audit,
)


def render_detector_section() -> None:
    """Render Section 7: Malicious Participant Detection."""
    st.markdown("## 🛡️ Multi-Factor Malicious Participant Detector")
    st.markdown(
        r"""
        **Phase 6 Execution:** Calibrated, frozen Tri-State Malicious Participant Detector with Symmetrical Clinical Trade-Off Shield.  
        Evaluated out-of-sample on 1,800 client gradient observations (1,640 honest, 160 malicious).
        """
    )

    det_res = load_detector_results()
    thresholds = load_detector_thresholds()
    fp_audit = load_honest_false_positive_audit()

    # Executive Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Continuous AUROC", "0.9814", help="Area Under the Receiver Operating Characteristic Curve")
    with col2:
        st.metric("Continuous AUPRC", "0.8801", help="Area Under Precision-Recall Curve")
    with col3:
        st.metric("False Malicious Rate", "0.00%", help="0 false malicious accusations out of 1,640 honest observations")
    with col4:
        st.metric("Clinical Shield Success", "100%", help="199 / 199 negative cosine events protected from false exclusion")

    st.divider()

    # Performance Figure
    st.markdown("### 📈 1. Detector ROC & Calibration Performance")
    c_roc, c_perf_text = st.columns([3, 2])

    roc_path = PHASE6_DETECTOR_DIR / "figures" / "phase6_detector_performance.png"

    with c_roc:
        if roc_path.exists():
            st.image(Image.open(roc_path), caption="Phase 6 Detector ROC, PR Curves & Score Distributions", use_container_width=True)
        else:
            st.info("Detector performance figure not found.")

    with c_perf_text:
        st.markdown(
        r"""
            #### Out-of-Sample Performance Summary:
            - **Evaluated Observations:** 1,800 gradient vectors across 9 independent simulations.
            - **Honest Observations:** 1,640 updates (IID, $\alpha=1.0, 0.5, 0.1$).
            - **Malicious Observations:** 160 updates (Label Flipping, Scaling, Inversion, Backdoors).
            - **Detection Latency:** Malicious gradient scaling detected in **Round 1**; label flipping flagged in **Rounds 1–2**.
            """
        )
        st.success("✓ **Zero False Accusations:** No honest hospital was ever classified as `MALICIOUS` across any data heterogeneity.")

    st.divider()

    # Stratified False Positive Audit Table
    st.markdown("### 📊 2. Stratified False Positive Audit across Clinical Heterogeneity")
    st.caption("Proves that clinical data specialization does not trigger false malicious disqualification.")

    stratified_table = [
        {"Cohort / Heterogeneity": "Honest IID Baseline", "Honest Observations": 880, "False Malicious": "0 (0.00%)", "Suspicious": "1 (0.11%)", "Normal": "879 (99.89%)"},
        {"Cohort / Heterogeneity": "Honest Non-IID (α = 1.0)", "Honest Observations": 200, "False Malicious": "0 (0.00%)", "Suspicious": "38 (19.00%)", "Normal": "162 (81.00%)"},
        {"Cohort / Heterogeneity": "Honest Non-IID (α = 0.5)", "Honest Observations": 360, "False Malicious": "0 (0.00%)", "Suspicious": "79 (21.94%)", "Normal": "281 (78.06%)"},
        {"Cohort / Heterogeneity": "Honest Non-IID (α = 0.1 - Extreme)", "Honest Observations": 200, "False Malicious": "0 (0.00%)", "Suspicious": "46 (23.00%)", "Normal": "154 (77.00%)"},
        {"Cohort / Heterogeneity": "Total Honest Cohort", "Honest Observations": 1640, "False Malicious": "0 (0.00%)", "Suspicious": "164 (10.00%)", "Normal": "1476 (90.00%)"},
    ]
    st.table(stratified_table)

    st.divider()

    # The 5-Factor Feature Extraction & Clinical Shield
    st.markdown("### ⚙️ 3. The Five-Factor Feature Architecture")
    f1, f2 = st.columns(2)

    with f1:
        st.markdown(
        r"""
            #### 1. Layer-Wise Magnitude Norms
            Decomposes $D = 420,802$ parameters into 4 functional layers (Conv1, Conv2, Dense1, Dense2). Uses **Modified Z-Scores** based on Median Absolute Deviation (MAD) to detect gradient scaling without sensitivity to extreme outliers:
            $$M_i = \frac{0.6745 \cdot (x_i - \tilde{x})}{\text{MAD}}$$

            #### 2. Directional Cosine Alignment
            Computes cosine similarity against a robust coordinate-wise median reference vector $\Delta w_{\text{ref}}$:
            $$\cos(\theta_k) = \frac{\langle \Delta w^k, \Delta w_{\text{ref}} \rangle}{\|\Delta w^k\|_2 \cdot \|\Delta w_{\text{ref}}\|_2}$$
            Detects gradient inversion ($\gamma=-1.0$) and orthogonal drifting.
            """
        )

    with f2:
        st.markdown(
        r"""
            #### 3. Class-Disentangled Validation Loss Impacts
            Evaluates the counterfactual impact of client $k$'s update on a balanced validation split separately for Normal and Pneumonia:
            $$\Delta \mathcal{L}_k^{(c)} = \mathcal{L}_{\text{val}}^{(c)}(w_t + \Delta w^k) - \mathcal{L}_{\text{val}}^{(c)}(w_t)$$

            #### 4. Symmetrical Clinical Trade-Off Shield
            If an update improves one class ($\Delta \mathcal{L}^{(c)} < 0$) while slightly increasing loss on the other, it is identified as a **legitimate specialized clinic** and protected from false malicious classification:
            $$\text{Triggered on 199 negative cosine events} \implies \mathbf{100\%\ (199/199)\ protected.}$$

            #### 5. Trajectory Velocity EMA
            Tracks exponential moving average of score accumulation to distinguish temporary variance from persistent poisoning.
            """
        )

    st.divider()

    # Tri-State Decision Engine
    st.markdown("### 🚦 4. Tri-State Consensus Logic")
    t_c1, t_c2, t_c3 = st.columns(3)

    with t_c1:
        st.success("#### 🟢 State 0: NORMAL")
        st.write("**Concordant with consensus**")
        st.write("Receives full sample-proportional aggregation weight:")
        st.latex(r"w_k = \frac{n_k}{n}")

    with t_c2:
        st.warning("#### 🟡 State 1: SUSPICIOUS")
        st.write("**Ambiguous clinical drift**")
        st.write(r"Update is retained but attenuated by discount factor $\lambda=0.50$:")
        st.latex(r"w_k = \lambda \cdot \frac{n_k}{n}")

    with t_c3:
        st.error("#### 🔴 State 2: MALICIOUS")
        st.write("**Severe Byzantine divergence**")
        st.write("Quarantined from the global model; flagged on blockchain:")
        st.latex(r"w_k = 0")

    if thresholds:
        with st.expander("🔍 View Calibrated Detection Thresholds"):
            st.json(thresholds)
