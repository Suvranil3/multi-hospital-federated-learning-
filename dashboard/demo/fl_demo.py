"""
Section 4: 10-Hospital IID Federated Learning Baseline
Demonstrates Flower FL engine, FedAvg convergence across 20 rounds, and the +5.13% boost over centralized training.
"""

from pathlib import Path
import streamlit as st
from PIL import Image
from dashboard.demo.pipeline_state import (
    RESULTS_FIGURES_DIR,
    load_federated_iid_results,
)


def render_fl_section() -> None:
    """Render Section 4: Federated Learning Baseline."""
    st.markdown("## ⚡ 10-Hospital IID Federated Learning Baseline")
    st.markdown(
        r"""
        **Phase 3 Execution:** Decentralized collaborative training across 10 simulated hospital nodes using Flower and FedAvg.  
        Demonstrates that cross-institutional model averaging preserves patient privacy while outperforming centralized training.
        """
    )

    fed_res = load_federated_iid_results()

    # Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("FL Test Accuracy", "86.38%", delta="+5.13% vs Centralized", help="539 / 624 correct diagnoses on untouched test split")
    with col2:
        st.metric("Normal Specificity", "70.94%", delta="+20.51% vs Centralized", help="166 / 234 normal scans correctly classified")
    with col3:
        st.metric("Pneumonia Recall", "95.64%", help="373 / 390 active pneumonia cases correctly diagnosed")
    with col4:
        st.metric("Communication", "20 Rounds", help="Flower Virtual Client Engine, FedAvg consensus")

    st.divider()

    # Centralized vs Federated Head-to-Head Comparison
    st.markdown("### 📊 1. Centralized vs. Federated Learning Comparison")
    st.caption("Federated averaging acts as an implicit regularizer across clients, boosting generalization on minority normal scans.")

    comparison_table = [
        {"Model / Paradigm": "Centralized Baseline (All 4,708 Scans)", "Test Acc": "81.25%", "Test Loss": "0.9172", "Sensitivity": "99.74%", "Specificity": "50.43%", "Macro F1": "0.7689"},
        {"Model / Paradigm": "10-Hospital FedAvg (20 Rounds)", "Test Acc": "86.38%", "Test Loss": "0.5516", "Sensitivity": "95.64%", "Specificity": "70.94%", "Macro F1": "0.8415"},
        {"Model / Paradigm": "Net Performance Delta", "Test Acc": "+5.13%", "Test Loss": "-0.3656", "Sensitivity": "-4.10%", "Specificity": "+20.51%", "Macro F1": "+0.0726"},
    ]
    st.table(comparison_table)

    comp_chart_path = RESULTS_FIGURES_DIR / "centralized_vs_federated_comparison.png"
    if comp_chart_path.exists():
        st.image(Image.open(comp_chart_path), caption="Centralized vs. Federated Learning Performance Comparison", use_container_width=True)

    st.divider()

    # Learning Curves & Confusion Matrix
    st.markdown("### 📈 2. Federated Convergence Trajectory & Confusion Matrix")
    c_cm, c_loss, c_acc = st.columns(3)

    cm_path = RESULTS_FIGURES_DIR / "federated_iid_confusion_matrix.png"
    loss_path = RESULTS_FIGURES_DIR / "federated_iid_training_loss.png"
    acc_path = RESULTS_FIGURES_DIR / "federated_iid_training_accuracy.png"

    with c_cm:
        if cm_path.exists():
            st.image(Image.open(cm_path), caption="Federated IID Test Confusion Matrix", use_container_width=True)
        else:
            st.info("Confusion matrix figure not found.")

    with c_loss:
        if loss_path.exists():
            st.image(Image.open(loss_path), caption="Round-by-Round Validation Loss", use_container_width=True)
        else:
            st.info("Loss curve figure not found.")

    with c_acc:
        if acc_path.exists():
            st.image(Image.open(acc_path), caption="Round-by-Round Validation Accuracy", use_container_width=True)
        else:
            st.info("Accuracy curve figure not found.")

    st.divider()

    # Privacy Guarantee & Flower Virtual Client Engine
    st.markdown("### 🔒 3. Privacy Preservation Mechanics")
    st.markdown(
        r"""
        - **Zero Raw Data Transfer:** Patient medical images ($28\times28$ scans) remain strictly quarantined inside the local hospital environment.
        - **Decentralized SGD:** In each communication round $t$, the central server broadcasts global model weights $w_t$. Each hospital $k \in \{0, \dots, 9\}$ executes local SGD on its private dataset $\mathcal{D}_k$:
          $$w_{t+1}^k \leftarrow w_t - \eta \nabla F_k(w_t)$$
        - **Parameter Delta Aggregation:** Hospitals transmit only parameter updates $\Delta w_t^k = w_{t+1}^k - w_t$. The server computes the sample-weighted consensus:
          $$w_{t+1} = w_t + \sum_{k=1}^{10} \frac{n_k}{n} \Delta w_t^k$$
        - **Flower Engine:** Implemented with Flower (`flwr` 1.39.0) Virtual Client Engine for clean, isolated client execution without network socket overhead.
        """
    )

    if fed_res:
        with st.expander("📄 View Full JSON Federated Results"):
            st.json(fed_res)
