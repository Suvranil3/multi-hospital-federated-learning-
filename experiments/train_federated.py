"""
Multi-Hospital Federated Learning Simulation (Phase 3: 10-Hospital IID Baseline).
Primary Medical Application: Pediatric Chest X-Ray Classification (PneumoniaMNIST).

Experimental Configuration:
- Model: SmallCNN (420,802 parameters) - identical to Centralized Baseline.
- Dataset: PneumoniaMNIST (5,856 total images).
- Clients: 10 simulated hospital clients.
- Partitioning: Uniform stratified IID across the 4,708 official training images.
- Communication Rounds: 20 rounds.
- Local Training Epochs: 2 epochs per round per client.
- Aggregation Protocol: Federated Averaging (FedAvg).
- Validation: Evaluated on official 524-sample validation split after every round.
- Test Evaluation: Evaluated on official untouched 624-sample test split strictly after training.

Artifacts Generated:
- results/models/best_federated_iid_model.pt
- results/figures/federated_iid_training_loss.png
- results/figures/federated_iid_training_accuracy.png
- results/figures/federated_iid_confusion_matrix.png
- results/figures/centralized_vs_federated_comparison.png
- results/logs/federated_iid_results.json
"""

import sys
from pathlib import Path
import time
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional

# Project root in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import flwr as fl
from flwr.simulation import start_simulation
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

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
from federated_learning.dataset import prepare_federated_data_loaders
from federated_learning.strategy import aggregate_fit_metrics, get_server_eval_fn
from experiments.train_medical import evaluate, compute_medical_metrics, plot_medical_confusion_matrix


def plot_federated_training_curves(
    round_metrics: List[Dict[str, Any]],
    output_loss_path: Path,
    output_acc_path: Path,
) -> None:
    """Plot round-by-round validation and training progression."""
    rounds = [m["round"] for m in round_metrics if m["round"] > 0]
    train_losses = [m.get("train_loss", None) for m in round_metrics if m["round"] > 0]
    val_losses = [m.get("val_loss", None) for m in round_metrics if m["round"] > 0]
    train_accs = [m.get("train_accuracy", None) for m in round_metrics if m["round"] > 0]
    val_accs = [m.get("val_accuracy", None) for m in round_metrics if m["round"] > 0]

    # 1. Loss Plot
    fig, ax = plt.subplots(figsize=(8.5, 5))
    if any(t is not None for t in train_losses):
        ax.plot(rounds, [t for t in train_losses], "o-", color="#1E3A8A", linewidth=2.0, label="Clients Train Loss (Aggregated)")
    if any(v is not None for v in val_losses):
        ax.plot(rounds, [v for v in val_losses], "s--", color="#DC2626", linewidth=2.0, label="Global Val Loss (Server)")
    ax.set_title("10-Hospital Federated Learning (IID): Loss vs. Communication Round", fontsize=12, fontweight="bold")
    ax.set_xlabel("Communication Round", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax.set_xticks(rounds)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_loss_path, dpi=200)
    plt.close(fig)

    # 2. Accuracy Plot
    fig, ax = plt.subplots(figsize=(8.5, 5))
    if any(t is not None for t in train_accs):
        ax.plot(rounds, [t for t in train_accs], "o-", color="#059669", linewidth=2.0, label="Clients Train Acc (%)")
    if any(v is not None for v in val_accs):
        ax.plot(rounds, [v for v in val_accs], "s--", color="#D97706", linewidth=2.0, label="Global Val Acc (%)")
    ax.set_title("10-Hospital Federated Learning (IID): Accuracy vs. Communication Round", fontsize=12, fontweight="bold")
    ax.set_xlabel("Communication Round", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(rounds)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_acc_path, dpi=200)
    plt.close(fig)


def plot_centralized_vs_federated_comparison(
    centralized_path: Path,
    federated_summary: Dict[str, Any],
    output_path: Path,
) -> None:
    """Compare Centralized Medical Baseline against 10-Hospital IID Federated Learning."""
    if not centralized_path.exists():
        return

    with open(centralized_path, "r", encoding="utf-8") as f:
        central_data = json.load(f)

    c_perf = central_data["test_performance"]
    f_perf = federated_summary["test_performance"]

    metrics_names = [
        "Test Accuracy (%)",
        "Pneumonia Sensitivity (%)",
        "Pneumonia Precision (%)",
        "Pneumonia F1 (×100)",
        "Macro F1 (×100)",
    ]

    c_vals = [
        c_perf["test_accuracy"],
        c_perf["pneumonia_metrics"]["recall_sensitivity"] * 100.0,
        c_perf["pneumonia_metrics"]["precision"] * 100.0,
        c_perf["pneumonia_metrics"]["f1_score"] * 100.0,
        c_perf["macro_metrics"]["macro_f1"] * 100.0,
    ]

    f_vals = [
        f_perf["test_accuracy"],
        f_perf["pneumonia_metrics"]["recall_sensitivity"] * 100.0,
        f_perf["pneumonia_metrics"]["precision"] * 100.0,
        f_perf["pneumonia_metrics"]["f1_score"] * 100.0,
        f_perf["macro_metrics"]["macro_f1"] * 100.0,
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    rects1 = ax.bar(x - width/2, c_vals, width, label="Centralized Small CNN Baseline", color="#2563EB", edgecolor="#1E3A8A")
    rects2 = ax.bar(x + width/2, f_vals, width, label="10-Hospital FedAvg (IID)", color="#059669", edgecolor="#065F46")

    ax.set_ylabel("Metric Value (%)", fontsize=11, fontweight="bold")
    ax.set_title("Medical PneumoniaMNIST: Centralized Baseline vs. 10-Hospital Federated Learning (IID)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    ax.bar_label(rects1, fmt="%.1f", padding=3, fontsize=8.5, fontweight="bold")
    ax.bar_label(rects2, fmt="%.1f", padding=3, fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def run_federated_iid_simulation(
    num_rounds: int = DEFAULT_FL_ROUNDS,
    local_epochs: int = DEFAULT_LOCAL_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_clients: int = NUM_HOSPITALS,
    learning_rate: float = 0.001,
    seed: int = SEED,
    device_name: str = "auto",
) -> Dict[str, Any]:
    """
    Main orchestration routine for Phase 3: 10-Hospital IID Federated Learning.
    """
    start_total_time = time.time()
    set_seed(seed)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    device_str = torch.cuda.get_device_name(device.index or 0) if device.type == "cuda" else "CPU"

    print("=" * 80)
    print("PHASE 3: 10-HOSPITAL IID FEDERATED LEARNING SIMULATION (PNEUMONIAMNIST)")
    print(f"Timestamp:              {datetime.now().isoformat()}")
    print(f"Deterministic Seed:     {seed}")
    print(f"Compute Device:         {device.type.upper()} ({device_str})")
    print(f"Number of Hospitals:    {num_clients}")
    print(f"Communication Rounds:   {num_rounds}")
    print(f"Local Epochs / Round:   {local_epochs}")
    print(f"Batch Size:             {batch_size}")
    print(f"Client Learning Rate:   {learning_rate}")
    print(f"Federated Algorithm:    FedAvg")
    print("=" * 80)

    # 1. Dataset Partitioning
    print("\n[Step 1/5] Partitioning PneumoniaMNIST across 10 simulated hospitals (IID)...")
    client_loaders, val_loader, test_loader, partition_meta = prepare_federated_data_loaders(
        data_dir=DATA_RAW_DIR,
        num_clients=num_clients,
        batch_size=batch_size,
        num_workers=0,
        seed=seed,
    )

    for stat in partition_meta["client_stats"]:
        print(
            f"  Hospital {stat['client_id']:02d}: {stat['total_samples']} samples "
            f"(Normal: {stat['normal_samples']}, Pneumonia: {stat['pneumonia_samples']}, "
            f"Pneumonia Ratio: {stat['pneumonia_ratio']*100:.1f}%)"
        )

    # 2. Global Model Initialization
    print("\n[Step 2/5] Initializing Global SmallCNN Model Architecture...")
    init_model = get_model(in_channels=1, num_classes=2, dropout_rate=0.2).to(device)
    params_count = count_parameters(init_model)
    print(f"  Total Model Parameters:     {params_count['total']:,}")
    print(f"  Trainable Model Parameters: {params_count['trainable']:,}")

    initial_ndarrays = get_parameters_from_model(init_model)
    initial_fl_parameters = ndarrays_to_parameters(initial_ndarrays)

    # Storage for round-by-round global evaluation history
    round_history: List[Dict[str, Any]] = []
    latest_global_ndarrays: List[np.ndarray] = [arr.copy() for arr in initial_ndarrays]
    best_val_acc: float = 0.0
    best_global_ndarrays: List[np.ndarray] = [arr.copy() for arr in initial_ndarrays]
    best_round: int = 0

    # Server evaluation hook
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
        fraction_fit=1.0,  # 100% of hospital clients train each round
        fraction_evaluate=0.0,  # Server-side validation used instead of distributed evaluate
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_fl_parameters,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_fn=evaluate_fn_wrapper,
        on_fit_config_fn=lambda rnd: {"local_epochs": local_epochs, "lr": learning_rate, "round": rnd},
    )

    # 4. Define Client Factory Function
    def client_fn(context: Any) -> fl.client.Client:
        # Resolve client ID from context or integer
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

    # 5. Execute Simulation
    print(f"\n[Step 4/5] Launching Flower Simulation: 20 Rounds across 10 Hospitals...")
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
    print(f"\n  Federated simulation completed in {simulation_duration:.2f} seconds.")
    print(f"  Best Server Validation Accuracy: {best_val_acc:.2f}% (Achieved at Round {best_round})")

    # Match training metrics from simulation_history with validation metrics
    for r_entry in round_history:
        r = r_entry["round"]
        # Search fit_metrics in simulation_history
        for metric_tuple in simulation_history.metrics_distributed_fit.get("train_loss", []):
            if metric_tuple[0] == r:
                r_entry["train_loss"] = round(float(metric_tuple[1]), 4)
        for metric_tuple in simulation_history.metrics_distributed_fit.get("train_accuracy", []):
            if metric_tuple[0] == r:
                r_entry["train_accuracy"] = round(float(metric_tuple[1]), 2)

    # Print summary table of rounds
    print("\n  Round-by-Round Training Progression:")
    print("  " + "-" * 75)
    print(f"  {'Round':<6} | {'Train Loss':<12} | {'Train Acc (%)':<14} | {'Val Loss':<10} | {'Val Acc (%)':<12}")
    print("  " + "-" * 75)
    for entry in round_history:
        r = entry["round"]
        t_loss = entry.get("train_loss", "N/A")
        t_acc = entry.get("train_accuracy", "N/A")
        v_loss = entry.get("val_loss", "N/A")
        v_acc = entry.get("val_accuracy", "N/A")
        print(f"  {r:<6} | {str(t_loss):<12} | {str(t_acc):<14} | {str(v_loss):<10} | {str(v_acc):<12}")
    print("  " + "-" * 75)

    # 6. Final Evaluation on Untouched Official Test Split
    print("\n[Step 5/5] Evaluating Best Global Model on Official Untouched Test Split (N=624)...")
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

    # 7. Save Model Checkpoints
    best_checkpoint_path = RESULTS_MODELS_DIR / "best_federated_iid_model.pt"
    final_checkpoint_path = RESULTS_MODELS_DIR / "final_federated_iid_model.pt"

    checkpoint_payload = {
        "round": best_round,
        "model_state_dict": eval_model.state_dict(),
        "val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "test_correct": test_correct,
        "total_test": total_test,
        "seed": seed,
        "algorithm": "FedAvg",
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "dataset": "PneumoniaMNIST",
    }
    torch.save(checkpoint_payload, best_checkpoint_path)

    # Also save final round model
    set_parameters_to_model(eval_model, latest_global_ndarrays)
    torch.save({**checkpoint_payload, "round": num_rounds, "model_state_dict": eval_model.state_dict()}, final_checkpoint_path)
    print(f"  ✓ Saved best federated checkpoint: {best_checkpoint_path}")
    print(f"  ✓ Saved final federated checkpoint: {final_checkpoint_path}")

    # 8. Visualizations
    loss_fig_path = RESULTS_FIGURES_DIR / "federated_iid_training_loss.png"
    acc_fig_path = RESULTS_FIGURES_DIR / "federated_iid_training_accuracy.png"
    cm_fig_path = RESULTS_FIGURES_DIR / "federated_iid_confusion_matrix.png"
    comp_fig_path = RESULTS_FIGURES_DIR / "centralized_vs_federated_comparison.png"

    plot_federated_training_curves(round_history, loss_fig_path, acc_fig_path)
    print(f"  ✓ Saved FL loss curve:     {loss_fig_path}")
    print(f"  ✓ Saved FL accuracy curve: {acc_fig_path}")

    plot_medical_confusion_matrix(cm_arr, list(CLASSES), cm_fig_path)
    print(f"  ✓ Saved FL confusion matrix: {cm_fig_path}")

    # 9. Structured JSON Results Record
    total_pipeline_time = time.time() - start_total_time
    results_summary = {
        "experiment_id": "EXP-MED-002-FED-IID",
        "timestamp": datetime.now().isoformat(),
        "status": "COMPLETED",
        "scenario": "10-Hospital Federated Learning Simulation (IID Baseline)",
        "dataset_name": "PneumoniaMNIST",
        "seed": seed,
        "environment": {
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "flower_version": fl.__version__,
            "device": device_str,
            "device_type": device.type,
        },
        "configuration": {
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "aggregation_strategy": "FedAvg",
            "distribution_type": "IID (Stratified)",
            "train_samples": partition_meta["train_samples"],
            "val_samples": partition_meta["val_samples"],
            "test_samples": partition_meta["test_samples"],
            "client_samples": [s["total_samples"] for s in partition_meta["client_stats"]],
        },
        "model": {
            "name": "SmallCNN",
            "in_channels": 1,
            "num_classes": 2,
            "total_parameters": params_count["total"],
            "trainable_parameters": params_count["trainable"],
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
            "loss_figure": str(loss_fig_path.relative_to(PROJECT_ROOT)),
            "accuracy_figure": str(acc_fig_path.relative_to(PROJECT_ROOT)),
            "confusion_matrix_figure": str(cm_fig_path.relative_to(PROJECT_ROOT)),
            "comparison_figure": str(comp_fig_path.relative_to(PROJECT_ROOT)),
            "best_checkpoint": str(best_checkpoint_path.relative_to(PROJECT_ROOT)),
            "final_checkpoint": str(final_checkpoint_path.relative_to(PROJECT_ROOT)),
            "results_json": "results/logs/federated_iid_results.json",
        },
    }

    # Generate centralized vs federated comparison plot
    centralized_json_path = RESULTS_LOGS_DIR / "medical_centralized_results.json"
    plot_centralized_vs_federated_comparison(centralized_json_path, results_summary, comp_fig_path)
    print(f"  ✓ Saved comparison figure: {comp_fig_path}")

    json_log_path = RESULTS_LOGS_DIR / "federated_iid_results.json"
    with open(json_log_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"  ✓ Saved federated JSON log: {json_log_path}")

    print("\n" + "=" * 80)
    print("PHASE 3: 10-HOSPITAL IID FEDERATED LEARNING SIMULATION COMPLETE.")
    print("=" * 80)
    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="10-Hospital IID Federated Learning Simulation")
    parser.add_argument("--rounds", type=int, default=DEFAULT_FL_ROUNDS, help="Communication rounds (default: 20)")
    parser.add_argument("--local-epochs", type=int, default=DEFAULT_LOCAL_EPOCHS, help="Local epochs per round (default: 2)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Local batch size (default: 32)")
    parser.add_argument("--num-clients", type=int, default=NUM_HOSPITALS, help="Number of hospital clients (default: 10)")
    parser.add_argument("--lr", type=float, default=0.001, help="Local learning rate (default: 0.001)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, cpu)")
    args = parser.parse_args()

    run_federated_iid_simulation(
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        num_clients=args.num_clients,
        learning_rate=args.lr,
        seed=args.seed,
        device_name=args.device,
    )
