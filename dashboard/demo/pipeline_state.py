"""
Demonstration Pipeline State and Data Loading Utilities.
Supports Primary Medical Pipeline (PneumoniaMNIST) and Archived CIFAR-10 Proof-of-Concept.
"""

from pathlib import Path
import json
from typing import Dict, Any, Optional
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_LOGS_DIR = PROJECT_ROOT / "results" / "logs"
RESULTS_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
RESULTS_MODELS_DIR = PROJECT_ROOT / "results" / "models"
RESULTS_ARCHIVE_DIR = PROJECT_ROOT / "results" / "archive" / "cifar10"


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
def load_medical_dataset_report() -> Optional[Dict[str, Any]]:
    """Load Phase MEDICAL-01 dataset verification report."""
    report_path = RESULTS_LOGS_DIR / "pneumonia_mnist_verification_report.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_medical_cnn_results() -> Optional[Dict[str, Any]]:
    """Load Phase MEDICAL-02 centralized medical CNN results."""
    report_path = RESULTS_LOGS_DIR / "medical_centralized_results.json"
    if not report_path.exists():
        report_path = RESULTS_LOGS_DIR / "centralized_cnn_results.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_archived_cifar_report() -> Optional[Dict[str, Any]]:
    """Load archived CIFAR-10 verification report."""
    report_path = RESULTS_ARCHIVE_DIR / "logs" / "cifar10_verification_report.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_archived_cifar_results() -> Optional[Dict[str, Any]]:
    """Load archived CIFAR-10 centralized CNN results."""
    report_path = RESULTS_ARCHIVE_DIR / "logs" / "cifar10_centralized_cnn_results.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_federated_iid_results() -> Optional[Dict[str, Any]]:
    """Load Phase 3 10-hospital IID federated learning results."""
    report_path = RESULTS_LOGS_DIR / "federated_iid_results.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# Backward-compatible aliases
load_cifar_report = load_archived_cifar_report
load_cnn_results = load_medical_cnn_results

