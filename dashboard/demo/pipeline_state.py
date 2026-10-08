"""
Demonstration Pipeline State and Data Loading Utilities.
"""

from pathlib import Path
import json
from typing import Dict, Any, Optional
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_LOGS_DIR = PROJECT_ROOT / "results" / "logs"
RESULTS_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
RESULTS_MODELS_DIR = PROJECT_ROOT / "results" / "models"


def init_session_state() -> None:
    """Initialize Streamlit session state keys if not already present."""
    if "demo_started" not in st.session_state:
        st.session_state.demo_started = False
    if "demo_stage" not in st.session_state:
        st.session_state.demo_stage = 1
    if "auto_advance" not in st.session_state:
        st.session_state.auto_advance = True
    if "training_in_progress" not in st.session_state:
        st.session_state.training_in_progress = False


def reset_demo_state() -> None:
    """Reset the interactive demonstration state without modifying any files."""
    st.session_state.demo_started = False
    st.session_state.demo_stage = 1
    st.session_state.training_in_progress = False


@st.cache_data(show_spinner=False)
def load_cifar_report() -> Optional[Dict[str, Any]]:
    """Load Phase 1 dataset verification report safely."""
    report_path = RESULTS_LOGS_DIR / "cifar10_verification_report.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_cnn_results() -> Optional[Dict[str, Any]]:
    """Load Phase 2 centralized CNN results safely."""
    report_path = RESULTS_LOGS_DIR / "centralized_cnn_results.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None
