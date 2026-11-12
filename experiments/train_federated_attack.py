"""
Phase 5: Parameterized Federated Learning Attack & Control Simulation Runner.
Executes both matched honest controls and adversarial attacks (Label Flipping,
Model Poisoning, Backdoor Injection) under exact standardized experimental parity.

Adheres strictly to the verified Phase 5 protocol:
1. Loads identical initial model state (verified SHA-256: 0c411b83...).
2. Round-by-round evaluation strictly uses clean validation set (N=524).
3. Official test set (N=624) and synthetic triggered test set (N=390) evaluated strictly post-training.
4. Model poisoning strictly transforms the 12 trainable parameter tensors; BatchNorm buffers pass through.
5. Backdoor trigger (+2.5422 white square at [24:27, 24:27]) applied post-normalization.
6. Independent JSON log and checkpoint saved for each run without overwriting Phase 3/4 artifacts.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import argparse
import hashlib
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from medmnist import PneumoniaMNIST
from torchvision import transforms
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
    PNEUMONIA_MEAN,
    PNEUMONIA_STD,
    set_seed,
)
from federated_learning.model import SmallCNN, get_model, count_parameters
from federated_learning.client import (
    HospitalClient,
    get_parameters_from_model,
    set_parameters_to_model,
)
from federated_learning.dataset import (
    prepare_federated_data_loaders,
    prepare_controlled_non_iid_data_loaders,
)
from federated_learning.strategy import aggregate_fit_metrics, get_server_eval_fn
from federated_learning.attacks.label_flipping import LabelFlippedSubset
from federated_learning.attacks.model_poisoning import (
    PoisonedHospitalClient,
    apply_adversarial_update_transformation,
)
from federated_learning.attacks.backdoor import (
    BackdoorSubset,
    TriggeredPneumoniaTestDataset,
    evaluate_backdoor_asr,
    TRIGGER_VALUE_NORMALIZED,
)
from experiments.train_medical import evaluate, compute_medical_metrics, plot_medical_confusion_matrix


def compute_param_hash(ndarrays: List[np.ndarray]) -> str:
    """Compute SHA-256 hash across parameter arrays."""
    hasher = hashlib.sha256()
    for arr in ndarrays:
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


def compute_partition_hash(client_loaders: List[DataLoader]) -> str:
    """Compute SHA-256 hash of client partition indices for strict reproducibility verification."""
    hasher = hashlib.sha256()
    for cid, loader in enumerate(client_loaders):
        idxs = getattr(loader.dataset, "indices", None)
        if idxs is None and hasattr(loader.dataset, "base_dataset"):
            idxs = getattr(loader.dataset.base_dataset, "indices", None)
        hasher.update(f"client_{cid}:".encode("utf-8"))
        hasher.update(np.array(idxs, dtype=np.int64).tobytes() if idxs is not None else b"")
    return hasher.hexdigest()


def run_federated_simulation(
    experiment_id: str,
    scenario: str = "iid",  # 'iid' or 'non_iid'
    alpha: float = 0.5,
    attack_type: str = "none",  # 'none', 'label_flipping', 'model_poisoning', 'backdoor'
    malicious_client_ids: Optional[List[int]] = None,
    flip_fraction: float = 1.0,
    gamma: float = -1.0,
    poison_fraction: float = 0.50,
    attack_start_round: int = 1,
    num_clients: int = NUM_HOSPITALS,
    rounds: int = DEFAULT_FL_ROUNDS,
    local_epochs: int = DEFAULT_LOCAL_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    partition_seed: int = 42,
    initialization_seed: int = 42,
    training_seed: int = 42,
    matched_control_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a standardized federated simulation (Honest Control or Adversarial Attack).
    """
    if malicious_client_ids is None:
        malicious_client_ids = []

    print("\n" + "=" * 90)
    print(f"  PHASE 5 FEDERATED SIMULATION: {experiment_id}")
    print(f"  Scenario: {scenario.upper()} (alpha={alpha if scenario=='non_iid' else 'N/A'})")
    print(f"  Attack Type: {attack_type.upper()} | Malicious Clients: {malicious_client_ids}")
    if attack_type == "label_flipping":
        print(f"  Label Flip: Pneumonia (1) -> Normal (0) | Flip Fraction: {flip_fraction*100:.1f}%")
    elif attack_type == "model_poisoning":
        print(f"  Model Poisoning Factor: gamma={gamma} (Applied strictly to 12 trainable parameters)")
    elif attack_type == "backdoor":
        print(f"  Backdoor Trigger: 3x3 white square (+2.5422) | Poison Fraction: {poison_fraction*100:.1f}%")
    print(f"  Seeds: Partition={partition_seed}, Init={initialization_seed}, Training={training_seed}")
    print("=" * 90)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Environment] Device: {device} | PyTorch: {torch.__version__} | Flower: {fl.__version__}")

    # 1. Dataset Partitioning (Identical to Phase 3 / Phase 4)
    print(f"\n[Step 1/5] Partitioning PneumoniaMNIST (Scenario: {scenario})...")
    if scenario == "iid":
        client_loaders, val_loader, test_loader, partition_meta = prepare_federated_data_loaders(
            data_dir=DATA_RAW_DIR,
            num_clients=num_clients,
            batch_size=batch_size,
            num_workers=0,
            seed=partition_seed,
        )
    else:
        client_loaders, val_loader, test_loader, partition_meta = prepare_controlled_non_iid_data_loaders(
            data_dir=DATA_RAW_DIR,
            num_clients=num_clients,
            alpha=alpha,
            batch_size=batch_size,
            num_workers=0,
            seed=partition_seed,
        )

    # Compute and verify partition hash prior to any malicious wrapping
    partition_hash = compute_partition_hash(client_loaders)
    print(f"  Verified Client Partition SHA-256 Hash: {partition_hash}")

    # Wrap malicious datasets if attack is data-driven (label flipping or backdoor)
    attack_meta_records: Dict[int, Dict[str, Any]] = {}
    total_poisoned_samples = 0

    if attack_type in ["label_flipping", "backdoor"]:
        for cid in malicious_client_ids:
            original_loader = client_loaders[cid]
            base_subset = original_loader.dataset

            if attack_type == "label_flipping":
                poisoned_subset = LabelFlippedSubset(
                    base_dataset=base_subset,
                    source_class=1,
                    target_class=0,
                    flip_fraction=flip_fraction,
                    seed=training_seed,
                )
                attack_meta = poisoned_subset.get_attack_metadata()
            elif attack_type == "backdoor":
                poisoned_subset = BackdoorSubset(
                    base_dataset=base_subset,
                    poison_fraction=poison_fraction,
                    target_class=0,
                    source_class=1,
                    seed=training_seed,
                )
                attack_meta = poisoned_subset.get_attack_metadata()

            attack_meta_records[cid] = attack_meta
            total_poisoned_samples += attack_meta["poisoned_sample_count"]

            # Rebuild malicious client's DataLoader
            client_loaders[cid] = DataLoader(
                poisoned_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
            )

    for stat in partition_meta["client_stats"]:
        cid = stat["client_id"]
        is_malicious = cid in malicious_client_ids
        role = f"MALICIOUS ({attack_type})" if is_malicious else "HONEST"
        poison_info = f" | Poisoned: {attack_meta_records[cid]['poisoned_sample_count']} samples" if is_malicious and cid in attack_meta_records else ""
        print(
            f"  Hospital {cid:02d} [{role}]: {stat['total_samples']} samples "
            f"(Normal: {stat['normal_samples']}, Pneumonia: {stat['pneumonia_samples']}, "
            f"Normal Ratio: {stat['normal_ratio']*100:.1f}%){poison_info}"
        )

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
    print(f"  Verified Identical Initial Model Parameter SHA-256 Hash: {initial_param_hash}")
    initial_fl_parameters = ndarrays_to_parameters(initial_ndarrays)

    # Reset training seed before simulation loop
    set_seed(training_seed)

    # Storage for round-by-round clean validation evaluation
    round_history: List[Dict[str, Any]] = []
    latest_global_ndarrays: List[np.ndarray] = [arr.copy() for arr in initial_ndarrays]
    best_val_acc: float = 0.0
    best_global_ndarrays: List[np.ndarray] = [arr.copy() for arr in initial_ndarrays]
    best_round: int = 0

    # Server evaluation hook strictly on clean validation set (N=524)
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
    print("\n[Step 3/5] Configuring Flower FedAvg Strategy with Clean Validation Evaluation...")
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
        is_malicious = cid in malicious_client_ids

        if is_malicious and attack_type == "model_poisoning":
            client = PoisonedHospitalClient(
                client_id=str(cid),
                train_loader=client_loader,
                val_loader=None,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                device=device,
                gamma=gamma,
                attack_start_round=attack_start_round,
            )
        else:
            client = HospitalClient(
                client_id=str(cid),
                train_loader=client_loader,
                val_loader=None,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                device=device,
            )

        return client.to_client()

    # 4. Simulation Execution
    print(f"\n[Step 4/5] Launching Flower Simulation: {rounds} Rounds across {num_clients} Hospitals...")
    sim_start = datetime.now()

    client_resources = {"num_cpus": 4, "num_gpus": 0.0}
    ray_init_args = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "_system_config": {"memory_monitor_refresh_ms": 0},
    }

    simulation_history = start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args=ray_init_args,
    )

    simulation_duration = (datetime.now() - sim_start).total_seconds()
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

    # 5. Strictly Post-Training Test Set Evaluation
    print(f"\n[Step 5/5] Evaluating Best Global Model (Round {best_round}) Strictly Post-Training...")
    eval_model = get_model(in_channels=1, num_classes=2, dropout_rate=0.2).to(device)
    set_parameters_to_model(eval_model, best_global_ndarrays)

    criterion = nn.CrossEntropyLoss()

    # 5A. Official Clean Test Split Evaluation (N=624)
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
    tn, fp, fn, tp = int(cm_arr[0, 0]), int(cm_arr[0, 1]), int(cm_arr[1, 0]), int(cm_arr[1, 1])

    pneumonia_misclassification_rate = float(fn / 390.0)

    print(f"  Clean Test Loss:                     {test_loss:.4f}")
    print(f"  Clean Test Accuracy:                 {test_acc:.2f}% ({test_correct}/{total_test})")
    print(f"  Pneumonia Sensitivity:               {p_metrics['recall_sensitivity']:.4f} ({p_metrics['recall_sensitivity']*100:.2f}%)")
    print(f"  Pneumonia Precision:                 {p_metrics['precision']:.4f} ({p_metrics['precision']*100:.2f}%)")
    print(f"  Pneumonia F1-Score:                  {p_metrics['f1_score']:.4f}")
    print(f"  Normal Specificity:                  {n_metrics['recall_specificity']:.4f} ({n_metrics['recall_specificity']*100:.2f}%)")
    print(f"  Macro F1-Score:                      {m_metrics['macro_f1']:.4f}")
    print(f"  Confusion Matrix:                    TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  Attack-Induced Pneumonia Misclass.:  {pneumonia_misclassification_rate*100:.2f}% ({fn}/390 missed)")

    # 5B. Backdoor / Matched Control Triggered ASR Evaluation (Synthetic Triggered Test Split, N=390)
    backdoor_asr_metrics = None
    if attack_type in ["none", "backdoor"]:
        eval_label = "Matched Control Triggered Benchmark" if attack_type == "none" else "Backdoor Attack Evaluation"
        print(f"\n  [{eval_label}] Evaluating Synthetic Triggered Test Set (N=390 Pneumonia Scans)...")
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=PNEUMONIA_MEAN, std=PNEUMONIA_STD),
        ])
        raw_test_dataset = PneumoniaMNIST(split="test", root=str(DATA_RAW_DIR), transform=eval_transform, download=False)
        triggered_test_dataset = TriggeredPneumoniaTestDataset(raw_test_dataset)
        triggered_loader = DataLoader(triggered_test_dataset, batch_size=batch_size, shuffle=False)
        backdoor_asr_metrics = evaluate_backdoor_asr(eval_model, triggered_loader, device)
        print(f"  Triggered Misclassification Rate (ASR): {backdoor_asr_metrics['attack_success_rate_percent']:.2f}% "
              f"({backdoor_asr_metrics['successful_backdoor_attacks']}/390 triggered scans classified as Normal)")

    # 5C. Matched Control Comparative Metrics
    matched_control_comparison = None
    if matched_control_id:
        ctrl_path = RESULTS_LOGS_DIR / f"{matched_control_id.lower().replace('-', '_')}_results.json"
        if ctrl_path.exists():
            with open(ctrl_path) as cf:
                ctrl_data = json.load(cf)
            ctrl_cm = ctrl_data["test_performance"]["confusion_matrix"]
            ctrl_fn = int(ctrl_cm["fn"])
            delta_fn = int(fn - ctrl_fn)

            matched_control_comparison = {
                "matched_control_id": matched_control_id,
                "control_test_accuracy": ctrl_data["test_performance"]["test_accuracy"],
                "control_fn": ctrl_fn,
                "attack_fn": fn,
                "delta_fn_increase": delta_fn,
            }
            print(f"\n  [Matched Control Comparison: {matched_control_id}]")
            print(f"    Control FN: {ctrl_fn} -> Attack FN: {fn} (Delta FN: {delta_fn:+d})")

            if attack_type == "backdoor" and backdoor_asr_metrics and ctrl_data.get("backdoor_performance"):
                ctrl_asr = float(ctrl_data["backdoor_performance"]["attack_success_rate_percent"])
                atk_asr = float(backdoor_asr_metrics["attack_success_rate_percent"])
                delta_asr = round(atk_asr - ctrl_asr, 2)
                matched_control_comparison["control_asr_baseline_percent"] = ctrl_asr
                matched_control_comparison["attack_asr_percent"] = atk_asr
                matched_control_comparison["delta_asr_increase_percent"] = delta_asr
                backdoor_asr_metrics["control_asr_baseline_percent"] = ctrl_asr
                backdoor_asr_metrics["delta_asr_increase_percent"] = delta_asr
                print(f"    Control Baseline ASR: {ctrl_asr:.2f}% -> Attack ASR: {atk_asr:.2f}% (Delta ASR: {delta_asr:+0.2f}%)")

    # Checkpoints
    tag = experiment_id.lower().replace("-", "_")
    best_checkpoint_path = RESULTS_MODELS_DIR / f"best_{tag}_model.pt"
    final_checkpoint_path = RESULTS_MODELS_DIR / f"final_{tag}_model.pt"

    checkpoint_payload = {
        "experiment_id": experiment_id,
        "round": best_round,
        "val_accuracy": best_val_acc,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "pneumonia_sensitivity": float(p_metrics["recall_sensitivity"]),
        "normal_specificity": float(n_metrics["recall_specificity"]),
        "attack_type": attack_type,
        "malicious_clients": malicious_client_ids,
        "model_state_dict": eval_model.state_dict(),
    }
    torch.save(checkpoint_payload, best_checkpoint_path)

    final_model = get_model(in_channels=1, num_classes=2, dropout_rate=0.2).to(device)
    set_parameters_to_model(final_model, latest_global_ndarrays)
    torch.save(
        {
            "experiment_id": experiment_id,
            "round": rounds,
            "attack_type": attack_type,
            "malicious_clients": malicious_client_ids,
            "model_state_dict": final_model.state_dict(),
        },
        final_checkpoint_path,
    )

    # Visual Artifacts
    fig_cm_path = RESULTS_FIGURES_DIR / f"{tag}_confusion_matrix.png"
    plot_medical_confusion_matrix(cm_arr, list(CLASSES), fig_cm_path)

    fig_curves_path = RESULTS_FIGURES_DIR / f"{tag}_training_curves.png"
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot([h["round"] for h in round_history], [h["val_loss"] for h in round_history], "r-o", label="Val Loss")
    plt.title("Server Validation Loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([h["round"] for h in round_history], [h["val_accuracy"] for h in round_history], "b-o", label="Val Acc")
    plt.axvline(best_round, color="g", linestyle="--", label=f"Best Checkpoint (Rd {best_round})")
    plt.title(f"Validation Accuracy (Best: {best_val_acc:.2f}%)")
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_curves_path, dpi=300)
    plt.close()

    # Structured JSON Log
    result_log: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "status": "COMPLETED",
        "scenario": scenario,
        "alpha": alpha if scenario == "non_iid" else None,
        "attack_type": attack_type,
        "malicious_client_ids": malicious_client_ids,
        "compromised_percentage": (len(malicious_client_ids) / num_clients) * 100.0,
        "matched_control_id": matched_control_id,
        "attack_parameters": {
            "flip_fraction": flip_fraction if attack_type == "label_flipping" else None,
            "gamma": gamma if attack_type == "model_poisoning" else None,
            "poison_fraction": poison_fraction if attack_type == "backdoor" else None,
            "attack_start_round": attack_start_round if attack_type != "none" else None,
            "total_poisoned_samples": total_poisoned_samples,
            "client_attack_metadata": attack_meta_records,
        },
        "seeds": {
            "partition_seed": partition_seed,
            "initialization_seed": initialization_seed,
            "training_seed": training_seed,
            "partition_hash_sha256": partition_hash,
        },
        "initial_model_hash_sha256": initial_param_hash,
        "matched_control_comparison": matched_control_comparison,
        "training_performance": {
            "duration_seconds": round(simulation_duration, 2),
            "total_rounds": rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "best_round": best_round,
            "best_validation_accuracy": round(best_val_acc, 2),
        },
        "test_performance": {
            "evaluation_split": "official_test_untouched",
            "test_samples": total_test,
            "test_loss": round(float(test_loss), 4),
            "test_accuracy": round(float(test_acc), 2),
            "correct_predictions": test_correct,
            "pneumonia_class_1": p_metrics,
            "normal_class_0": n_metrics,
            "macro_averaged": m_metrics,
            "confusion_matrix": {
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            },
            "attack_induced_pneumonia_misclassification_rate": round(pneumonia_misclassification_rate, 4),
        },
        "backdoor_performance": backdoor_asr_metrics,
        "client_partitions": partition_meta["client_stats"],
        "round_history": round_history,
        "artifacts": {
            "best_checkpoint": str(best_checkpoint_path),
            "final_checkpoint": str(final_checkpoint_path),
            "confusion_matrix_figure": str(fig_cm_path),
            "training_curves_figure": str(fig_curves_path),
        },
    }

    log_path = RESULTS_LOGS_DIR / f"{tag}_results.json"
    with open(log_path, "w") as f:
        json.dump(result_log, f, indent=2)

    print(f"\n  JSON Log saved to:      {log_path}")
    print(f"  Best Checkpoint:        {best_checkpoint_path}")
    print(f"  Confusion Matrix Plot:  {fig_cm_path}")
    print("=" * 90)

    return result_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 FL Attack and Control Runner")
    parser.add_argument("--exp_id", type=str, required=True, help="Experiment identifier")
    parser.add_argument("--scenario", type=str, default="iid", choices=["iid", "non_iid"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--attack_type", type=str, default="none", choices=["none", "label_flipping", "model_poisoning", "backdoor"])
    parser.add_argument("--malicious_clients", type=str, default="", help="Comma-separated IDs (e.g. '0' or '0,1')")
    parser.add_argument("--flip_fraction", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=-1.0)
    parser.add_argument("--poison_fraction", type=float, default=0.50)
    parser.add_argument("--attack_start_round", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=DEFAULT_FL_ROUNDS)
    parser.add_argument("--local_epochs", type=int, default=DEFAULT_LOCAL_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--control_id", type=str, default=None)

    args = parser.parse_args()
    malicious_ids = [int(x.strip()) for x in args.malicious_clients.split(",") if x.strip()]

    run_federated_simulation(
        experiment_id=args.exp_id,
        scenario=args.scenario,
        alpha=args.alpha,
        attack_type=args.attack_type,
        malicious_client_ids=malicious_ids,
        flip_fraction=args.flip_fraction,
        gamma=args.gamma,
        poison_fraction=args.poison_fraction,
        attack_start_round=args.attack_start_round,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        matched_control_id=args.control_id,
    )


if __name__ == "__main__":
    main()
