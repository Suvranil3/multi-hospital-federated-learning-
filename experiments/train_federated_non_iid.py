"""
Multi-Hospital Federated Learning: Non-IID Hospital Data Heterogeneity Study (Phase 4).
Primary Medical Application: Pediatric Chest X-Ray Classification (PneumoniaMNIST).

Evaluates FedAvg across 10 simulated hospitals with controlled sample capacities
(~471 samples/client) under population-calibrated Beta/Dirichlet case-mix skew:
  - Alpha = 1.0 (EXP-MED-003-FED-NONIID-A10)
  - Alpha = 0.5 (EXP-MED-004-FED-NONIID-A05)
  - Alpha = 0.1 (EXP-MED-005-FED-NONIID-A01)

Parity Guarantees:
  - Architecture: SmallCNN (420,802 parameters), identical to Phase 2 & Phase 3.
  - Hyperparameters: Adam (lr=0.001, wd=1e-4), CrossEntropyLoss, batch_size=32, 2 local epochs, 20 rounds.
  - Aggregation: Sample-weighted FedAvg including BatchNorm running buffers.
  - Checkpoint Selection: Strictly highest global validation accuracy (> previous best).
  - Test Evaluation: Untouched test set (624 samples) evaluated only once post-training.
  - Identical Initialization: All runs load the exact initial parameter state dict verified by SHA-256 hash.
"""

import sys
from pathlib import Path
import time
import json
import hashlib
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import flwr as fl
from flwr.simulation import start_simulation
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import ray

from experiments.config import (
    DATA_RAW_DIR,
    RESULTS_FIGURES_DIR,
    RESULTS_LOGS_DIR,
    RESULTS_MODELS_DIR,
    SEED,
    CLASSES,
    CLASS_LABELS,
    NUM_HOSPITALS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LOCAL_EPOCHS,
    DEFAULT_FL_ROUNDS,
    set_seed,
)
from federated_learning.model import SmallCNN, get_model, count_parameters
from federated_learning.client import (
    HospitalClient,
    get_parameters_from_model,
    set_parameters_to_model,
)
from federated_learning.dataset import prepare_controlled_non_iid_data_loaders
from federated_learning.strategy import aggregate_fit_metrics, get_server_eval_fn
from experiments.train_medical import evaluate, compute_medical_metrics, plot_medical_confusion_matrix


def compute_param_hash(ndarrays: List[np.ndarray]) -> str:
    """Compute SHA-256 hash across parameter arrays."""
    hasher = hashlib.sha256()
    for arr in ndarrays:
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


def plot_hospital_case_mix_distribution(
    client_stats: List[Dict[str, Any]],
    alpha: float,
    output_path: Path,
) -> None:
    """Plot hospital sample distribution and class breakdown."""
    num_clients = len(client_stats)
    cids = [s["client_id"] for s in client_stats]
    normal_counts = [s["normal_samples"] for s in client_stats]
    pneu_counts = [s["pneumonia_samples"] for s in client_stats]
    achieved_ratios = [s["normal_ratio"] * 100.0 for s in client_stats]
    target_ratios = [s.get("target_normal_ratio", 0.0) * 100.0 for s in client_stats]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [1.8, 1]})

    # Stacked bar chart of counts
    x = np.arange(num_clients)
    width = 0.55
    b1 = ax1.bar(x, normal_counts, width, label="Normal Cases (Class 0)", color="#3B82F6", edgecolor="#1D4ED8")
    b2 = ax1.bar(x, pneu_counts, width, bottom=normal_counts, label="Pneumonia Cases (Class 1)", color="#EF4444", edgecolor="#B91C1C")

    ax1.set_ylabel("Patient Sample Count", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"10-Hospital Case-Mix Partitioning (Dirichlet alpha = {alpha})\n"
        f"Controlled Hospital Capacity (~471 samples/client, Total = 4,708)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Hospital {i}" for i in cids], fontsize=10, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate counts inside bars
    for i in range(num_clients):
        tot = normal_counts[i] + pneu_counts[i]
        n_pct = normal_counts[i] / tot * 100
        p_pct = pneu_counts[i] / tot * 100
        ax1.text(i, tot + 8, f"{tot}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        if normal_counts[i] > 25:
            ax1.text(i, normal_counts[i] / 2, f"{normal_counts[i]}\n({n_pct:.0f}%)", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        if pneu_counts[i] > 25:
            ax1.text(i, normal_counts[i] + pneu_counts[i] / 2, f"{pneu_counts[i]}\n({p_pct:.0f}%)", ha="center", va="center", color="white", fontsize=8, fontweight="bold")

    # Lower plot: Target vs Achieved Normal %
    w2 = 0.35
    r1 = ax2.bar(x - w2/2, target_ratios, w2, label="Sampled Dirichlet Target Normal %", color="#9CA3AF", edgecolor="#4B5563")
    r2 = ax2.bar(x + w2/2, achieved_ratios, w2, label="Achieved Normal % (After Constrained Rounding)", color="#059669", edgecolor="#065F46")
    ax2.axhline(25.79, color="#DC2626", linestyle=":", linewidth=1.5, label="Global Population Prior (25.79% Normal)")

    ax2.set_ylabel("Normal Class %", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Hospital Client ID", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"H{i}" for i in cids], fontsize=10)
    ax2.set_ylim(0, 115)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_non_iid_training_curves(
    round_metrics: List[Dict[str, Any]],
    alpha: float,
    output_loss_path: Path,
    output_acc_path: Path,
) -> None:
    """Plot round-by-round validation and training curves."""
    rounds = [m["round"] for m in round_metrics if m["round"] > 0]
    train_losses = [m.get("train_loss", None) for m in round_metrics if m["round"] > 0]
    val_losses = [m.get("val_loss", None) for m in round_metrics if m["round"] > 0]
    train_accs = [m.get("train_accuracy", None) for m in round_metrics if m["round"] > 0]
    val_accs = [m.get("val_accuracy", None) for m in round_metrics if m["round"] > 0]

    # 1. Loss
    fig, ax = plt.subplots(figsize=(8.5, 5))
    if any(t is not None for t in train_losses):
        ax.plot(rounds, [t for t in train_losses], "o-", color="#1E3A8A", linewidth=2.0, label="Clients Train Loss (Aggregated)")
    if any(v is not None for v in val_losses):
        ax.plot(rounds, [v for v in val_losses], "s--", color="#DC2626", linewidth=2.0, label="Global Val Loss (Server)")
    ax.set_title(f"10-Hospital Federated Learning (Non-IID, alpha={alpha}): Loss vs Round", fontsize=12, fontweight="bold")
    ax.set_xlabel("Communication Round", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax.set_xticks(rounds)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_loss_path, dpi=200)
    plt.close(fig)

    # 2. Accuracy
    fig, ax = plt.subplots(figsize=(8.5, 5))
    if any(t is not None for t in train_accs):
        ax.plot(rounds, [t for t in train_accs], "o-", color="#059669", linewidth=2.0, label="Clients Train Accuracy (Aggregated)")
    if any(v is not None for v in val_accs):
        ax.plot(rounds, [v for v in val_accs], "s--", color="#D97706", linewidth=2.0, label="Global Val Accuracy (Server)")
    ax.set_title(f"10-Hospital Federated Learning (Non-IID, alpha={alpha}): Accuracy vs Round", fontsize=12, fontweight="bold")
    ax.set_xlabel("Communication Round", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(rounds)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_acc_path, dpi=200)
    plt.close(fig)


def plot_heterogeneity_comparison(
    centralized_path: Path,
    iid_path: Path,
    non_iid_summaries: Dict[float, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Multi-condition comparative plot across Centralized, IID, and Non-IID alphas."""
    with open(centralized_path, "r", encoding="utf-8") as f:
        central_data = json.load(f)
    with open(iid_path, "r", encoding="utf-8") as f:
        iid_data = json.load(f)

    conditions = [
        ("Centralized (1 Node)", central_data["test_performance"], "#2563EB"),
        ("FedAvg IID (10 Hosp)", iid_data["test_performance"], "#059669"),
    ]
    for alpha in [1.0, 0.5, 0.1]:
        if alpha in non_iid_summaries:
            color_map = {1.0: "#D97706", 0.5: "#DC2626", 0.1: "#7C3AED"}
            conditions.append((f"Non-IID alpha={alpha}", non_iid_summaries[alpha]["test_performance"], color_map[alpha]))

    metric_names = [
        "Test Accuracy",
        "Pneumonia Sensitivity",
        "Pneumonia Precision",
        "Normal Specificity",
        "Macro F1",
    ]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    num_conds = len(conditions)
    num_metrics = len(metric_names)
    x = np.arange(num_metrics)
    total_width = 0.82
    width = total_width / num_conds

    for i, (label, perf, color) in enumerate(conditions):
        vals = [
            perf["test_accuracy"],
            perf["pneumonia_metrics"]["recall_sensitivity"] * 100.0,
            perf["pneumonia_metrics"]["precision"] * 100.0,
            perf["normal_metrics"]["recall_specificity"] * 100.0,
            perf["macro_metrics"]["macro_f1"] * 100.0,
        ]
        offset = (i - (num_conds - 1) / 2) * width
        rects = ax.bar(x + offset, vals, width, label=label, color=color, edgecolor="black", alpha=0.9)
        ax.bar_label(rects, fmt="%.1f", padding=3, fontsize=7.5, fontweight="bold")

    ax.set_ylabel("Metric Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Multi-Hospital Benchmark Comparison: Centralized vs. IID vs. Non-IID Heterogeneity (PneumoniaMNIST)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def run_federated_non_iid_simulation(
    alpha: float,
    num_rounds: int = DEFAULT_FL_ROUNDS,
    local_epochs: int = DEFAULT_LOCAL_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_clients: int = NUM_HOSPITALS,
    learning_rate: float = 0.001,
    partition_seed: int = 42,
    init_seed: int = 42,
    training_seed: int = 42,
    device_name: str = "auto",
) -> Dict[str, Any]:
    """Execute full 20-round Non-IID federated training run for given alpha."""
    start_total_time = time.time()

    # Alpha tag for filenames (10, 05, 01)
    alpha_tag = f"{int(round(alpha * 10)):02d}"
    exp_id = f"EXP-MED-00{3 if alpha==1.0 else 4 if alpha==0.5 else 5}-FED-NONIID-A{alpha_tag}"

    # Setup device
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    device_str = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    print("\n" + "=" * 80)
    print(f"PHASE 4: 10-HOSPITAL NON-IID FEDERATED LEARNING SIMULATION [alpha = {alpha}]")
    print(f"Experiment ID:          {exp_id}")
    print(f"Timestamp:              {datetime.now().isoformat()}")
    print(f"Partition Seed:         {partition_seed}")
    print(f"Initialization Seed:    {init_seed}")
    print(f"Training Seed:          {training_seed}")
    print(f"Compute Device:         {device.type.upper()} ({device_str})")
    print(f"Number of Hospitals:    {num_clients}")
    print(f"Communication Rounds:   {num_rounds}")
    print(f"Local Epochs / Round:   {local_epochs}")
    print(f"Batch Size:             {batch_size}")
    print(f"Client Learning Rate:   {learning_rate}")
    print(f"Federated Algorithm:    FedAvg (with BatchNorm running statistics aggregation)")
    print("=" * 80)

    # 1. Dataset Partitioning
    print(f"\n[Step 1/5] Partitioning PneumoniaMNIST across 10 hospitals (alpha = {alpha})...")
    client_loaders, val_loader, test_loader, partition_meta = prepare_controlled_non_iid_data_loaders(
        data_dir=DATA_RAW_DIR,
        num_clients=num_clients,
        alpha=alpha,
        batch_size=batch_size,
        num_workers=0,
        seed=partition_seed,
        tolerance=40,
    )

    het_metrics = partition_meta["heterogeneity_metrics"]
    print(f"  Rejection Attempt:        {het_metrics['acceptance_attempt']} (diff={het_metrics['initial_discrepancy']:+.1f} samples)")
    print(f"  Concentration Vector:     {het_metrics['dirichlet_parameter_vector']}")
    print(f"  Achieved Normal % Var:    {het_metrics['normal_ratio_var']:.4f} (Std: {het_metrics['normal_ratio_std']*100:.2f}%)")
    print(f"  Mean Absolute Deviation:  {het_metrics['mean_abs_deviation']*100:.2f}% (Max: {het_metrics['max_abs_deviation']*100:.2f}%)")
    for stat in partition_meta["client_stats"]:
        print(
            f"  Hospital {stat['client_id']:02d}: Total = {stat['total_samples']} | "
            f"Normal = {stat['normal_samples']:3d} ({stat['normal_ratio']*100:5.1f}%) | "
            f"Pneumonia = {stat['pneumonia_samples']:3d} ({stat['pneumonia_ratio']*100:5.1f}%) | "
            f"Target = {stat['target_normal_ratio']*100:5.1f}% | Dev = {stat['deviation_from_target']*100:4.2f}%"
        )

    # Plot hospital distribution
    dist_fig_path = RESULTS_FIGURES_DIR / f"hospital_distribution_alpha{alpha_tag}.png"
    plot_hospital_case_mix_distribution(partition_meta["client_stats"], alpha, dist_fig_path)
    print(f"  ✓ Saved hospital distribution plot: {dist_fig_path}")

    # 2. Global Model Initialization (Identical Initial State Loading)
    print("\n[Step 2/5] Loading Identical Reference Initial Model Weights...")
    init_checkpoint_path = RESULTS_MODELS_DIR / "initial_federated_model.pt"
    if not init_checkpoint_path.exists():
        raise FileNotFoundError(f"Missing initial reference model at {init_checkpoint_path}")

    init_payload = torch.load(init_checkpoint_path, map_location="cpu", weights_only=False)
    init_model = get_model(in_channels=1, num_classes=2, dropout_rate=0.2)
    init_model.load_state_dict(init_payload["model_state_dict"], strict=True)
    init_model = init_model.to(device)

    initial_ndarrays = get_parameters_from_model(init_model)
    initial_param_hash = compute_param_hash(initial_ndarrays)
    expected_hash = "0c411b83c2f0acef58beb5b9a968030d877e079eb2356eb3748903f8ae9bbc83"
    assert initial_param_hash == expected_hash, (
        f"Initial parameter hash mismatch!\nExpected: {expected_hash}\nGot:      {initial_param_hash}"
    )
    print(f"  ✓ Verified Identical Initial Model Parameter SHA-256 Hash: {initial_param_hash}")

    initial_fl_parameters = ndarrays_to_parameters(initial_ndarrays)

    # Reset training seed before simulation loop
    set_seed(training_seed)

    # Storage for round-by-round global evaluation history
    round_history: List[Dict[str, Any]] = []
    latest_global_ndarrays: List[np.ndarray] = [arr.copy() for arr in initial_ndarrays]
    best_val_acc: float = 0.0
    best_global_ndarrays: List[np.ndarray] = [arr.copy() for arr in initial_ndarrays]
    best_round: int = 0

    base_eval_fn = get_server_eval_fn(val_loader=val_loader, device=device, record_history=round_history)

    def evaluate_fn_wrapper(
        server_round: int,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        nonlocal latest_global_ndarrays, best_val_acc, best_global_ndarrays, best_round
        latest_global_ndarrays = [arr.copy() for arr in parameters]
        res = base_eval_fn(server_round, parameters, config)
        if res is not None:
            loss, metrics = res
            val_acc = float(metrics.get("val_accuracy", 0.0))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_round = server_round
                best_global_ndarrays = [arr.copy() for arr in parameters]
        return res

    # 3. Strategy Configuration
    print("\n[Step 3/5] Configuring Flower FedAvg Strategy with Server Evaluation...")
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_fl_parameters,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_fn=evaluate_fn_wrapper,
        on_fit_config_fn=lambda rnd: {"local_epochs": local_epochs, "lr": learning_rate, "round": rnd},
    )

    def client_fn(context: Any) -> fl.client.Client:
        if hasattr(context, "node_config") and "partition-id" in context.node_config:
            cid = int(context.node_config["partition-id"])
        elif hasattr(context, "node_id"):
            cid = int(context.node_id)
        else:
            cid = int(context)

        client_loader = client_loaders[cid]
        client = HospitalClient(
            client_id=str(cid),
            train_loader=client_loader,
            val_loader=None,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            weight_decay=1e-4,
            device=device,
        )
        return client.to_client()

    # 4. Simulation Execution
    print(f"\n[Step 4/5] Launching Flower Simulation: 20 Rounds across 10 Hospitals (alpha={alpha})...")
    sim_start_time = time.time()

    client_resources = {"num_cpus": 4, "num_gpus": 0.0}
    ray_init_args = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "_system_config": {"memory_monitor_refresh_ms": 0},
    }

    simulation_history = start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args=ray_init_args,
    )

    simulation_duration = time.time() - sim_start_time
    print(f"\n  Simulation completed in {simulation_duration:.2f} seconds.")
    print(f"  Best Server Validation Accuracy: {best_val_acc:.2f}% (Achieved at Round {best_round})")

    # Match train metrics from simulation_history
    for r_entry in round_history:
        r = r_entry["round"]
        for m_tuple in simulation_history.metrics_distributed_fit.get("train_loss", []):
            if m_tuple[0] == r:
                r_entry["train_loss"] = round(float(m_tuple[1]), 4)
        for m_tuple in simulation_history.metrics_distributed_fit.get("train_accuracy", []):
            if m_tuple[0] == r:
                r_entry["train_accuracy"] = round(float(m_tuple[1]), 2)

    # 5. Untouched Test Set Evaluation
    print(f"\n[Step 5/5] Evaluating Best Global Model (Round {best_round}) on Official Untouched Test Split (N=624)...")
    eval_model = get_model(in_channels=1, num_classes=2, dropout_rate=0.2).to(device)
    set_parameters_to_model(eval_model, best_global_ndarrays)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_targets, test_preds = evaluate(
        model=eval_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        return_predictions=True,
    )
    test_correct = int(np.sum(test_targets == test_preds))
    total_test = len(test_targets)

    med_metrics = compute_medical_metrics(test_targets, test_preds)
    p_metrics = med_metrics["pneumonia_class_1"]
    n_metrics = med_metrics["normal_class_0"]
    m_metrics = med_metrics["macro_averaged"]
    cm_arr = np.array(med_metrics["confusion_matrix"]["matrix_array"])

    print(f"  Final Test Loss:        {test_loss:.4f}")
    print(f"  Final Test Accuracy:    {test_acc:.2f}% ({test_correct}/{total_test})")
    print(f"  Pneumonia Sensitivity:  {p_metrics['recall_sensitivity']:.4f} ({p_metrics['recall_sensitivity']*100:.2f}%)")
    print(f"  Pneumonia Precision:    {p_metrics['precision']:.4f} ({p_metrics['precision']*100:.2f}%)")
    print(f"  Pneumonia F1-Score:     {p_metrics['f1_score']:.4f}")
    print(f"  Normal Specificity:     {n_metrics['recall_specificity']:.4f} ({n_metrics['recall_specificity']*100:.2f}%)")
    print(f"  Macro F1-Score:         {m_metrics['macro_f1']:.4f}")
    print(f"  Confusion Matrix:       TN={cm_arr[0,0]}, FP={cm_arr[0,1]}, FN={cm_arr[1,0]}, TP={cm_arr[1,1]}")

    # Checkpoints
    best_checkpoint_path = RESULTS_MODELS_DIR / f"best_federated_non_iid_alpha{alpha_tag}_model.pt"
    final_checkpoint_path = RESULTS_MODELS_DIR / f"final_federated_non_iid_alpha{alpha_tag}_model.pt"

    checkpoint_payload = {
        "experiment_id": exp_id,
        "round": best_round,
        "model_state_dict": eval_model.state_dict(),
        "val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "test_correct": test_correct,
        "total_test": total_test,
        "alpha": alpha,
        "partition_seed": partition_seed,
        "init_seed": init_seed,
        "training_seed": training_seed,
        "initial_param_hash": initial_param_hash,
        "algorithm": "FedAvg",
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "dataset": "PneumoniaMNIST",
    }
    torch.save(checkpoint_payload, best_checkpoint_path)

    set_parameters_to_model(eval_model, latest_global_ndarrays)
    torch.save({**checkpoint_payload, "round": num_rounds, "model_state_dict": eval_model.state_dict()}, final_checkpoint_path)
    print(f"  ✓ Saved best checkpoint:  {best_checkpoint_path}")
    print(f"  ✓ Saved final checkpoint: {final_checkpoint_path}")

    # Visualizations
    loss_fig_path = RESULTS_FIGURES_DIR / f"federated_non_iid_alpha{alpha_tag}_training_loss.png"
    acc_fig_path = RESULTS_FIGURES_DIR / f"federated_non_iid_alpha{alpha_tag}_training_accuracy.png"
    cm_fig_path = RESULTS_FIGURES_DIR / f"federated_non_iid_alpha{alpha_tag}_confusion_matrix.png"

    plot_non_iid_training_curves(round_history, alpha, loss_fig_path, acc_fig_path)
    plot_medical_confusion_matrix(cm_arr, list(CLASSES), cm_fig_path)
    print(f"  ✓ Saved curves: {loss_fig_path}, {acc_fig_path}")
    print(f"  ✓ Saved confusion matrix: {cm_fig_path}")

    # Structured JSON log
    total_pipeline_time = time.time() - start_total_time
    results_summary = {
        "experiment_id": exp_id,
        "timestamp": datetime.now().isoformat(),
        "status": "COMPLETED",
        "scenario": f"10-Hospital Federated Learning Simulation (Controlled Non-IID, alpha={alpha})",
        "dataset_name": "PneumoniaMNIST",
        "seeds": {
            "partition_seed": partition_seed,
            "initialization_seed": init_seed,
            "training_seed": training_seed,
        },
        "initial_model_hash_sha256": initial_param_hash,
        "batch_norm_buffers_aggregated": True,
        "batch_norm_buffers_tracked": [
            "features.1.running_mean",
            "features.1.running_var",
            "features.1.num_batches_tracked",
            "features.5.running_mean",
            "features.5.running_var",
            "features.5.num_batches_tracked",
        ],
        "configuration": {
            "alpha": alpha,
            "total_concentration_parameter": alpha,
            "dirichlet_parameter_vector": het_metrics["dirichlet_parameter_vector"],
            "global_class_prior": het_metrics["global_class_prior"],
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "aggregation_strategy": "FedAvg",
            "distribution_type": f"Controlled-Capacity Dirichlet Case-Mix (alpha={alpha})",
            "train_samples": partition_meta["train_samples"],
            "val_samples": partition_meta["val_samples"],
            "test_samples": partition_meta["test_samples"],
            "min_client_size": het_metrics["min_client_size"],
            "max_client_size": het_metrics["max_client_size"],
        },
        "heterogeneity_characterization": {
            "rejection_acceptance_attempt": het_metrics["acceptance_attempt"],
            "initial_unconstrained_discrepancy": het_metrics["initial_discrepancy"],
            "normal_ratio_mean": het_metrics["normal_ratio_mean"],
            "normal_ratio_std": het_metrics["normal_ratio_std"],
            "normal_ratio_var": het_metrics["normal_ratio_var"],
            "mean_absolute_deviation_from_target": het_metrics["mean_abs_deviation"],
            "max_absolute_deviation_from_target": het_metrics["max_abs_deviation"],
        },
        "training_performance": {
            "best_round": best_round,
            "best_val_accuracy": round(best_val_acc, 2),
            "final_round_val_accuracy": round_history[-1].get("val_accuracy", None) if round_history else None,
            "simulation_duration_seconds": round(simulation_duration, 2),
            "total_pipeline_time_seconds": round(total_pipeline_time, 2),
        },
        "test_performance": {
            "test_loss": round(test_loss, 4),
            "test_accuracy": round(test_acc, 2),
            "test_correct": test_correct,
            "total_test_samples": total_test,
            "pneumonia_metrics": p_metrics,
            "normal_metrics": n_metrics,
            "macro_metrics": m_metrics,
            "confusion_matrix": med_metrics["confusion_matrix"],
        },
        "client_partitions": partition_meta["client_stats"],
        "round_history": round_history,
        "medical_disclaimer": (
            "Benchmark research metrics for multi-hospital federated learning simulation. "
            "Not evaluated or intended for direct clinical diagnostic use."
        ),
        "artifacts": {
            "hospital_distribution_figure": str(dist_fig_path.relative_to(PROJECT_ROOT)),
            "loss_figure": str(loss_fig_path.relative_to(PROJECT_ROOT)),
            "accuracy_figure": str(acc_fig_path.relative_to(PROJECT_ROOT)),
            "confusion_matrix_figure": str(cm_fig_path.relative_to(PROJECT_ROOT)),
            "best_checkpoint": str(best_checkpoint_path.relative_to(PROJECT_ROOT)),
            "final_checkpoint": str(final_checkpoint_path.relative_to(PROJECT_ROOT)),
            "results_json": f"results/logs/federated_non_iid_alpha{alpha_tag}_results.json",
        },
    }

    json_log_path = RESULTS_LOGS_DIR / f"federated_non_iid_alpha{alpha_tag}_results.json"
    with open(json_log_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"  ✓ Saved JSON log: {json_log_path}")

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="Multi-Hospital Non-IID Federated Learning Simulation")
    parser.add_argument("--alpha", type=float, default=None, help="Dirichlet concentration parameter (e.g. 1.0, 0.5, 0.1)")
    parser.add_argument("--all-alphas", action="store_true", help="Execute 1.0, 0.5, and 0.1 sequentially")
    parser.add_argument("--rounds", type=int, default=DEFAULT_FL_ROUNDS, help="Communication rounds (default: 20)")
    parser.add_argument("--local-epochs", type=int, default=DEFAULT_LOCAL_EPOCHS, help="Local epochs per round (default: 2)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--partition-seed", type=int, default=42, help="Partition seed (default: 42)")
    parser.add_argument("--init-seed", type=int, default=42, help="Model initialization seed (default: 42)")
    parser.add_argument("--training-seed", type=int, default=42, help="Training seed (default: 42)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, cpu)")
    args = parser.parse_args()

    alphas_to_run = [1.0, 0.5, 0.1] if args.all_alphas or args.alpha is None else [args.alpha]

    non_iid_summaries = {}
    for a in alphas_to_run:
        summary = run_federated_non_iid_simulation(
            alpha=a,
            num_rounds=args.rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            partition_seed=args.partition_seed,
            init_seed=args.init_seed,
            training_seed=args.training_seed,
            device_name=args.device,
        )
        non_iid_summaries[a] = summary

    # Generate multi-condition comparison if all alphas ran
    if len(alphas_to_run) == 3:
        centralized_path = RESULTS_LOGS_DIR / "medical_centralized_results.json"
        iid_path = RESULTS_LOGS_DIR / "federated_iid_results.json"
        comp_fig_path = RESULTS_FIGURES_DIR / "iid_vs_non_iid_heterogeneity_comparison.png"
        plot_heterogeneity_comparison(centralized_path, iid_path, non_iid_summaries, comp_fig_path)
        print(f"\n  ✓ Saved multi-condition comparison figure: {comp_fig_path}")

        # Summary comparison JSON
        comparison_payload = {
            "study": "Phase 4: Non-IID Hospital Data Heterogeneity Study",
            "timestamp": datetime.now().isoformat(),
            "benchmarks": {
                "centralized_baseline": {
                    "experiment_id": "EXP-MED-001-CENTRAL-CNN",
                    "accuracy": 81.25,
                    "pneumonia_sensitivity": 99.74,
                    "pneumonia_precision": 77.03,
                    "normal_specificity": 50.43,
                    "macro_f1": 0.7689,
                },
                "federated_iid_baseline": {
                    "experiment_id": "EXP-MED-002-FED-IID",
                    "accuracy": 85.74,
                    "pneumonia_sensitivity": 97.95,
                    "pneumonia_precision": 82.51,
                    "normal_specificity": 65.38,
                    "macro_f1": 0.8352,
                },
            },
            "non_iid_experiments": {},
        }
        for a, summ in non_iid_summaries.items():
            t_p = summ["test_performance"]
            tag = f"alpha_{a}"
            comparison_payload["non_iid_experiments"][tag] = {
                "experiment_id": summ["experiment_id"],
                "alpha": a,
                "best_round": summ["training_performance"]["best_round"],
                "best_val_accuracy": summ["training_performance"]["best_val_accuracy"],
                "test_loss": t_p["test_loss"],
                "test_accuracy": t_p["test_accuracy"],
                "pneumonia_sensitivity": round(t_p["pneumonia_metrics"]["recall_sensitivity"] * 100.0, 2),
                "pneumonia_precision": round(t_p["pneumonia_metrics"]["precision"] * 100.0, 2),
                "pneumonia_f1": round(t_p["pneumonia_metrics"]["f1_score"], 4),
                "normal_specificity": round(t_p["normal_metrics"]["recall_specificity"] * 100.0, 2),
                "macro_f1": round(t_p["macro_metrics"]["macro_f1"], 4),
                "confusion_matrix": t_p["confusion_matrix"],
                "hospital_normal_variance": summ["heterogeneity_characterization"]["normal_ratio_var"],
            }

        comp_json_path = RESULTS_LOGS_DIR / "phase4_heterogeneity_comparison.json"
        with open(comp_json_path, "w", encoding="utf-8") as f:
            json.dump(comparison_payload, f, indent=2)
        print(f"  ✓ Saved comparison summary log: {comp_json_path}")

    # Final ray cleanup
    if ray.is_initialized():
        ray.shutdown()

    print("\n" + "=" * 80)
    print("PHASE 4: NON-IID DATA HETEROGENEITY EXPERIMENTS COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    main()
