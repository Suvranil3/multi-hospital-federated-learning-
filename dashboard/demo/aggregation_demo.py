"""
Section 8: Byzantine-Robust Aggregation Comparison (Phase 7)
Evaluates FedAvg, Multi-Krum, Tri-State Monitor, and Tri-State Discount (0.50) across 9 conditions and 2 seeds.
"""

from pathlib import Path
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dashboard.demo.pipeline_state import (
    load_phase7_reduced_registry,
    load_batchnorm_evaluation,
)


def render_aggregation_section() -> None:
    """Render Section 8: Byzantine-Robust Aggregation Comparison."""
    st.markdown("## 🛡️ Byzantine-Robust Aggregation Comparison (Phase 7)")
    st.markdown(
        r"""
        **Phase 7 Execution:** Systematic evaluation of 4 aggregation strategies across 9 conditions and 2 independent training seeds ($N=2$: Seeds 211, 212).  
        *Disclosure:* Evaluated on 72 primary runs (plus 24 preserved supplementary simulations); results represent empirical benchmark trends across two random seeds.
        """
    )

    reg = load_phase7_reduced_registry()
    bn_eval = load_batchnorm_evaluation()

    if not reg:
        st.error("Phase 7 execution registry not found.")
        return

    # Parse registry into a clean DataFrame
    rows = []
    for exp_id, data in reg.items():
        m = data.get("test_metrics", {})
        cm = m.get("confusion_matrix", {})
        rows.append({
            "experiment_id": exp_id,
            "condition": data.get("condition_id"),
            "strategy": data.get("aggregation_variant"),
            "seed": data.get("seed"),
            "accuracy": m.get("test_accuracy", 0.0),
            "sensitivity": m.get("sensitivity", 0.0),
            "specificity": m.get("specificity", 0.0),
            "macro_f1": m.get("macro_f1", 0.0),
            "backdoor_asr": m.get("backdoor_asr"),
            "tn": cm.get("tn", 0),
            "fp": cm.get("fp", 0),
            "fn": cm.get("fn", 0),
            "tp": cm.get("tp", 0),
        })
    df = pd.DataFrame(rows)

    # Strategy display names mapping
    strategy_names = {
        "fedavg": "FedAvg (Baseline)",
        "multi_krum_f1": "Multi-Krum (f=1)",
        "tri_state_monitor": "Tri-State Monitor",
        "tri_state_discount_050": "Tri-State Discount (0.50)",
    }

    condition_names = {
        "HONEST-IID": "1. Honest IID Baseline",
        "HONEST-A10": "2. Honest Non-IID (α = 1.0)",
        "HONEST-A05": "3. Honest Non-IID (α = 0.5 - Realistic)",
        "HONEST-A01": "4. Honest Non-IID (α = 0.1 - Severe)",
        "ATK-LF-IID-M1": "5. Label Flipping (IID, M=1)",
        "ATK-LF-NONIID-M1": "6. Label Flipping (Non-IID α=0.5, M=1)",
        "ATK-BD-IID-M1": "7. Backdoor Watermark (IID, M=1)",
        "ATK-BN-IID-M1": "8. BatchNorm Poisoning (IID, Outlier=10⁶)",
        "ATK-BN-NONIID-M1": "9. BatchNorm Poisoning (Non-IID α=0.5, Outlier=10⁶)",
    }

    # Condition Selector
    st.markdown("### 🎛️ 1. Interactive Condition & Strategy Selector")
    cond_keys = list(condition_names.keys())
    selected_cond = st.selectbox(
        "Select Clinical / Adversarial Condition:",
        cond_keys,
        format_func=lambda k: condition_names[k],
        index=cond_keys.index("ATK-LF-NONIID-M1"),
    )

    df_cond = df[df["condition"] == selected_cond]

    # Compute mean and seed breakdown for this condition
    summary_list = []
    for strat_key, strat_name in strategy_names.items():
        sub = df_cond[df_cond["strategy"] == strat_key]
        if not sub.empty:
            acc_s211 = sub[sub["seed"] == 211]["accuracy"].values[0] if len(sub[sub["seed"] == 211]) > 0 else 0
            acc_s212 = sub[sub["seed"] == 212]["accuracy"].values[0] if len(sub[sub["seed"] == 212]) > 0 else 0
            acc_mean = sub["accuracy"].mean()
            spec_mean = sub["specificity"].mean()
            sens_mean = sub["sensitivity"].mean()
            f1_mean = sub["macro_f1"].mean()
            asr_vals = sub["backdoor_asr"].dropna()
            asr_mean = asr_vals.mean() if not asr_vals.empty else None

            summary_list.append({
                "Strategy": strat_name,
                "Seed 211 Acc": f"{acc_s211:.2f}%",
                "Seed 212 Acc": f"{acc_s212:.2f}%",
                "Mean Test Acc": f"{acc_mean:.2f}%",
                "Mean Specificity": f"{spec_mean:.2f}%",
                "Mean Sensitivity": f"{sens_mean:.2f}%",
                "Mean Macro F1": f"{f1_mean:.2f}%",
                "Backdoor ASR": f"{asr_mean:.2f}%" if asr_mean is not None else "—",
                "_acc": acc_mean,
                "_spec": spec_mean,
                "_sens": sens_mean,
                "_f1": f1_mean,
            })

    df_summary = pd.DataFrame(summary_list)
    st.table(df_summary[["Strategy", "Seed 211 Acc", "Seed 212 Acc", "Mean Test Acc", "Mean Specificity", "Mean Sensitivity", "Mean Macro F1", "Backdoor ASR"]])

    st.divider()

    # Visual Comparison Bar Chart for Selected Condition
    st.markdown(f"### 📊 2. Strategy Metrics Comparison for `{selected_cond}`")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    
    strats = [s["Strategy"] for s in summary_list]
    accs = [s["_acc"] for s in summary_list]
    specs = [s["_spec"] for s in summary_list]
    senss = [s["_sens"] for s in summary_list]

    x = range(len(strats))
    width = 0.25

    ax.bar([i - width for i in x], accs, width=width, label="Test Accuracy", color="#3B82F6")
    ax.bar(x, specs, width=width, label="Specificity (Normal)", color="#10B981")
    ax.bar([i + width for i in x], senss, width=width, label="Sensitivity (Pneumonia)", color="#F59E0B")

    ax.set_ylabel("Metric Score (%)", fontsize=11)
    ax.set_title(f"Diagnostic Metrics across Aggregators: {condition_names[selected_cond]}", fontsize=12, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(strats, fontsize=9.5)
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Three Major Empirical Breakthroughs & Failures
    st.markdown("### 🔬 3. Three Core Empirical Discoveries")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.success("#### 🏆 Discovery 1: Tri-State Rescues Specificity")
        st.markdown(
        r"""
            - **Setting:** Non-IID Label Flipping ($\alpha=0.5, M=1$).
            - **FedAvg:** Specificity drops to **30.77%**.
            - **Multi-Krum:** Specificity collapses to **25.00%**.
            - **Tri-State Discount (0.50):** Rescues specificity to **58.98%** (**+28.21%** over FedAvg; **+33.98%** over Multi-Krum).
            - **Mechanism:** Soft discounting attenuates the attacker by 27.92%–36.00% cumulatively without gradient shock.
            """
        )

    with d2:
        st.warning("#### ⚠️ Discovery 2: Multi-Krum Minority Rejection")
        st.markdown(
        r"""
            - **Setting:** Honest Non-IID Specialization ($\alpha=0.5, M=0$).
            - **Multi-Krum:** Specificity collapsed to **23.08%** (vs FedAvg 54.48%).
            - **Telemetry Proof:** Client 2 (Seed 211, 97.9% Normal) rejected in **18/20 rounds (90%)**; Client 3 (Seed 212, 98.3% Normal) rejected in **19/20 rounds (95%)**.
            - **Failure:** Multi-Krum eliminates honest minority medical clinics due to Euclidean distance bias!
            """
        )

    with d3:
        st.error("#### 🚨 Discovery 3: BatchNorm Buffer Blind Spot")
        st.markdown(
        r"""
            - **Setting:** Outlier injection ($10^6$) into `running_mean` & `running_var`.
            - **Outcome:** **100% universal model collapse** across all 4 aggregators into single-class predictors:
              - *All-Normal (37.50% Acc)*
              - *All-Pneumonia (62.50% Acc)*
            - **Root Cause:** Parameter detectors check weights ($\Delta w$), leaving buffers unmonitored. Linear buffer averaging propagates outliers to $\approx 10^5$.
            """
        )
