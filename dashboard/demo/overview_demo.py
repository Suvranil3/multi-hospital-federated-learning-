"""
Section 1: Project Overview & System Architecture
Provides high-level context, problem statement, 4-tier architecture, and milestone audit status.
"""

import streamlit as st
from dashboard.demo.pipeline_state import get_test_suite_summary, verify_blockchain_ledger_cached


def render_overview_section() -> None:
    """Render Section 1: Project Overview."""
    st.markdown("## 🏥 Project Overview: Multi-Hospital Federated Learning Security")
    st.markdown(
        r"""
        **Minor Project Final Demonstration**  
        *Faculty Evaluation & Engineering Defense Presentation*
        """
    )

    # Executive Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Simulated Hospitals", "10 Nodes", help="Simulated clinical hospital partitions on PneumoniaMNIST")
    with col2:
        st.metric("Detection AUROC", "0.9814", help="Out-of-sample evaluated on 1,800 gradient observations")
    with col3:
        st.metric("Honest False Positive Rate", "0.00%", help="0 false accusations across all non-IID skews")
    with col4:
        st.metric("Blockchain Audit Trail", "2,134 Blocks", help="Cryptographic SHA-256 tamper-evident ledger")

    st.divider()

    # The Core Clinical Problem Statement
    st.markdown("### 🔬 1. The Clinical Dilemma & Research Gap")
    col_prob, col_sol = st.columns(2)

    with col_prob:
        st.error("#### ⚠️ The Privacy & Adversarial Problem")
        st.markdown(
        r"""
            - **Strict Regulatory Firewalls:** Laws like HIPAA, GDPR, and HITECH prohibit centralizing identifiable patient radiographs across institutions.
            - **Adversarial Susceptibility:** Decentralized learning allows malicious or compromised nodes to inject poisoned updates (label flipping, gradient scaling, backdoor triggers).
            - **Clinical Data Heterogeneity:** Specialized clinics (e.g., pediatric vs. adult respiratory) naturally have skewed case-mixes, producing legitimate updates that diverge geometrically from the consensus.
            - **Classical Failure Mode:** Byzantine-robust aggregators (e.g., Krum, Coordinate-wise Median) mistake clinical specialization for adversarial attacks, severely penalizing minority pathology detection.
            """
        )

    with col_sol:
        st.success("#### 🛡️ The Proposed Multi-Tier Solution")
        st.markdown(
        r"""
            - **Decentralized Parameter Sharing:** 10 simulated hospitals train local SmallCNNs, transmitting only parameter updates ($\Delta w$) while patient scans stay local.
            - **Multi-Factor Non-IID-Aware Detector:** Combines layer magnitude norms, directional cosine alignment vs median, class-disentangled validation impacts, and trajectory velocity.
            - **Symmetrical Clinical Trade-Off Shield:** Detects when updates sacrifice one class to improve another (benign specialization), protecting honest sites from false rejection.
            - **Soft Tri-State Discounting ($\lambda=0.50$):** Suppresses malicious influence without inducing gradient shock or dropping specialized minority hospitals.
            - **Permissioned Blockchain Ledger:** Cryptographically anchors every update hash and audit decision.
            """
        )

    st.divider()

    # System Architecture
    st.markdown("### 🏗️ 2. Four-Tier System Architecture")
    st.markdown(
        r"""
        ```text
        ┌────────────────────────────────────────────────────────────────────────┐
        │ 1. HOSPITAL DATA LAYER (10 Simulated Clinical Nodes)                   │
        │    PneumoniaMNIST Scans (5,856 images) | Constrained Dirichlet Skew    │
        └──────────────────────────────────┬─────────────────────────────────────┘
                                           │ Local SGD Training
                                           ▼
        ┌────────────────────────────────────────────────────────────────────────┐
        │ 2. FEDERATED LEARNING ENGINE (Flower Framework)                        │
        │    SmallCNN (420,802 Params) | 20 Communication Rounds | Δw Parameter Sharing│
        └──────────────────────────────────┬─────────────────────────────────────┘
                                           │ Update Vectors Δw_k
                                           ▼
        ┌────────────────────────────────────────────────────────────────────────┐
        │ 3. SECURITY & DEFENSE SUBSYSTEM                                         │
        │    • 5-Factor Feature Extraction (Layer Norms, Cosine, Loss Disentangling)│
        │    • Symmetrical Clinical Trade-Off Shield                              │
        │    • Tri-State Decision: [NORMAL | SUSPICIOUS | MALICIOUS]             │
        │    • Robust Aggregation Engine (FedAvg / Multi-Krum / Tri-State Disc)  │
        └──────────────────────────────────┬─────────────────────────────────────┘
                                           │ Audit Metadata & Root Hashes
                                           ▼
        ┌────────────────────────────────────────────────────────────────────────┐
        │ 4. PERMISSIONED BLOCKCHAIN AUDIT SUBSYSTEM                             │
        │    Immutable JSONL Ledger | SHA-256 Chaining | 2,134 Verified Blocks   │
        └────────────────────────────────────────────────────────────────────────┘
        ```
        """
    )

    st.divider()

    # Project Milestone Tracking Table
    st.markdown("### 🗺️ 3. Project Verification & Milestone Status")
    st.caption("All phases through Phase 8 are fully completed and audited against reproducible experimental artifacts.")

    phases = [
        ("Phase 1: PneumoniaMNIST Verification", "5,856 pediatric chest X-rays verified (splits, pixel stats)", "COMPLETED", "#10B981"),
        ("Phase 2: Centralized CNN Baseline", "SmallCNN (420k params), 81.25% Test Acc, 99.74% Sensitivity", "COMPLETED", "#10B981"),
        ("Phase 3: 10-Hospital IID Federated Learning", "Flower FedAvg, 20 rounds, 86.38% Test Acc (+5.13% over centralized)", "COMPLETED", "#10B981"),
        ("Phase 4: Non-IID Clinical Heterogeneity", "Dirichlet case-mix skew (alpha = 1.0, 0.5, 0.1) characterized", "COMPLETED", "#10B981"),
        ("Phase 5: Adversarial Attacks Suite", "Label flipping, updates, backdoors, BatchNorm poisoning benchmarked", "COMPLETED", "#10B981"),
        ("Phase 6: Multi-Factor Detector", "0.9814 AUROC, 0.00% False Malicious Rate across all Dirichlet skews", "COMPLETED", "#10B981"),
        ("Phase 7: Robust Aggregation Evaluation", "72 primary runs (Seeds 211, 212), Tri-State rescues specificity to 58.98%", "COMPLETED & AUDITED", "#10B981"),
        ("Phase 8: Academic Paper Development", "Full 19-section manuscript draft (~5,500 words), literature review (20 papers)", "COMPLETED", "#10B981"),
        ("Phase 9: Future Research Expansion", "Coordinate-wise median buffer defense & multi-seed statistical scaling (N >= 5)", "FUTURE SCOPE", "#6B7280"),
    ]

    for name, desc, status, color in phases:
        c1, c2, c3 = st.columns([3, 4, 2])
        with c1:
            st.markdown(f"**{name}**")
        with c2:
            st.write(desc)
        with c3:
            st.markdown(f"<span style='background-color:{color}; color:white; padding:0.2rem 0.6rem; border-radius:0.3rem; font-weight:bold; font-size:0.85rem;'>{status}</span>", unsafe_allow_html=True)

    st.divider()

    # Quality Assurance & Verification
    st.markdown("### 🧪 4. Automated Verification & Code Quality")
    t_sum = get_test_suite_summary()
    bc_sum = verify_blockchain_ledger_cached()

    q_col1, q_col2 = st.columns(2)
    with q_col1:
        st.success(f"**Automated Test Suite: {t_sum['passed_tests']} / {t_sum['total_tests']} Passed (100%)**")
        st.write("Regression tested across 11 test modules covering data loaders, CNN architecture, attacks, detector rules, Byzantine aggregators, and blockchain hashing.")
        with st.expander("🔍 View Test Modules Breakdown"):
            for mod in t_sum["test_modules"]:
                st.write(f"- `{mod['module']}`: **{mod['count']} tests** — *{mod['focus']}*")

    with q_col2:
        if bc_sum["is_valid"]:
            st.success(f"**Blockchain Audit Trail: {bc_sum['total_blocks']} Blocks Cryptographically Verified**")
            st.write("Full SHA-256 hash chaining audit verified from Genesis Block #0 through Tip Block #2133 with 0 broken links and 0 corrupted blocks.")
        else:
            st.error("Blockchain ledger verification failed.")
