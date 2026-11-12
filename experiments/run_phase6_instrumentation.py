"""
Phase 6: Instrumented Federated Learning Replay Runner.
Executes the 11 benchmark simulations (2 Calibration Honest Controls,
2 Held-Out Honest Controls, and 7 Adversarial Attacks) using InstrumentedFedAvg.

Preserves exact reproducibility:
- Initial model parameter hash: 0c411b83c2f0acef58beb5b9a968030d877e079eb2356eb3748903f8ae9bbc83
- Seeds: Partition=42, Init=42, Training=42
- Clean validation evaluation (N=524)
- Outputs feature records strictly to results/phase6_detector/instrumentation/
"""

import sys
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from flwr.simulation import start_simulation
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from experiments.config import (
    DATA_RAW_DIR,
    RESULTS_MODELS_DIR,
    SEED,
    NUM_HOSPITALS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LOCAL_EPOCHS,
    DEFAULT_FL_ROUNDS,
    set_seed,
)
from federated_learning.model import SmallCNN, get_model
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
from federated_learning.strategy_instrumented import InstrumentedFedAvg
from federated_learning.attacks.label_flipping import LabelFlippedSubset
from federated_learning.attacks.model_poisoning import PoisonedHospitalClient
from federated_learning.attacks.backdoor import BackdoorSubset

PHASE6_DIR = PROJECT_ROOT / "results" / "phase6_detector"
PHASE6_INSTRUMENTATION_DIR = PHASE6_DIR / "instrumentation"
PHASE6_CHECKPOINTS_DIR = PHASE6_DIR / "checkpoints"


def compute_param_hash(ndarrays: List[np.ndarray]) -> str:
    hasher = hashlib.sha256()
    for arr in ndarrays:
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


def compute_partition_hash(client_loaders: List[DataLoader]) -> str:
    hasher = hashlib.sha256()
    for cid, loader in enumerate(client_loaders):
        idxs = getattr(loader.dataset, "indices", None)
        if idxs is None and hasattr(loader.dataset, "base_dataset"):
            idxs = getattr(loader.dataset.base_dataset, "indices", None)
        hasher.update(f"client_{cid}:".encode("utf-8"))
        hasher.update(np.array(idxs, dtype=np.int64).tobytes() if idxs is not None else b"")
    return hasher.hexdigest()


def run_instrumented_simulation(
    experiment_id: str,
    scenario: str = "iid",  # 'iid' or 'non_iid'
    alpha: Optional[float] = None,
    attack_type: str = "none",  # 'none', 'label_flipping', 'model_poisoning', 'backdoor'
    malicious_client_ids: Optional[List[int]] = None,
    flip_fraction: float = 1.0,
    gamma: float = -1.0,
    poison_fraction: float = 0.50,
    num_clients: int = NUM_HOSPITALS,
    rounds: int = DEFAULT_FL_ROUNDS,
    local_epochs: int = DEFAULT_LOCAL_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    partition_seed: int = 42,
    initialization_seed: int = 42,
    training_seed: int = 42,
) -> Dict[str, Any]:
    """Execute a single instrumented simulation and save feature stream."""
    if malicious_client_ids is None:
        malicious_client_ids = []

    PHASE6_INSTRUMENTATION_DIR.mkdir(parents=True, exist_ok=True)
    PHASE6_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_output_path = PHASE6_INSTRUMENTATION_DIR / f"{experiment_id.lower()}_features.jsonl"

    print("\n" + "=" * 90)
    print(f"  PHASE 6 INSTRUMENTED SIMULATION: {experiment_id}")
    print(f"  Scenario: {scenario.upper()} (alpha={alpha if scenario=='non_iid' else 'N/A'})")
    print(f"  Attack Type: {attack_type.upper()} | Malicious Clients: {malicious_client_ids}")
    print(f"  Output JSONL: {jsonl_output_path}")
    print("=" * 90)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Dataset Partitioning
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
            alpha=alpha if alpha is not None else 0.5,
            batch_size=batch_size,
            num_workers=0,
            seed=partition_seed,
        )

    partition_hash = compute_partition_hash(client_loaders)
    print(f"  Partition Hash SHA-256: {partition_hash}")

    # Wrap malicious datasets if attack is data-driven
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
            elif attack_type == "backdoor":
                poisoned_subset = BackdoorSubset(
                    base_dataset=base_subset,
                    poison_fraction=poison_fraction,
                    target_class=0,
                    source_class=1,
                    seed=training_seed,
                )
            client_loaders[cid] = DataLoader(
                poisoned_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
            )

    # 2. Initial Model State Loading
    init_checkpoint_path = RESULTS_MODELS_DIR / "initial_federated_model.pt"
    init_payload = torch.load(init_checkpoint_path, map_location="cpu", weights_only=False)
    init_model = get_model(in_channels=1, num_classes=2, dropout_rate=0.2).to(device)
    init_model.load_state_dict(init_payload["model_state_dict"], strict=True)

    initial_ndarrays = get_parameters_from_model(init_model)
    initial_param_hash = compute_param_hash(initial_ndarrays)
    assert initial_param_hash == "0c411b83c2f0acef58beb5b9a968030d877e079eb2356eb3748903f8ae9bbc83"
    initial_fl_parameters = ndarrays_to_parameters(initial_ndarrays)

    # Reset seed before training loop
    set_seed(training_seed)

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

    # 3. Instrumented FedAvg Strategy
    strategy = InstrumentedFedAvg(
        val_loader=val_loader,
        device=device,
        output_jsonl_path=jsonl_output_path,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_fl_parameters,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_fn=evaluate_fn_wrapper,
        on_fit_config_fn=lambda rnd: {"local_epochs": local_epochs, "lr": learning_rate, "round": rnd},
    )

    # 4. Client Function
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
                attack_start_round=1,
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

    # 5. Execute Simulation
    sim_start = datetime.now()
    client_resources = {"num_cpus": 4, "num_gpus": 0.0}
    ray_init_args = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "_system_config": {"memory_monitor_refresh_ms": 0},
    }

    start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args=ray_init_args,
    )
    duration = (datetime.now() - sim_start).total_seconds()
    print(f"  Simulation completed in {duration:.2f}s. Best Val Acc: {best_val_acc:.2f}% (Rd {best_round})")

    # 6. Save instrumented checkpoints for hash comparison
    best_ckpt_path = PHASE6_CHECKPOINTS_DIR / f"best_{experiment_id.lower()}.pt"
    final_ckpt_path = PHASE6_CHECKPOINTS_DIR / f"final_{experiment_id.lower()}.pt"

    eval_m = get_model(in_channels=1, num_classes=2, dropout_rate=0.2)
    set_parameters_to_model(eval_m, best_global_ndarrays)
    torch.save({
        "experiment_id": experiment_id,
        "best_round": best_round,
        "best_val_acc": best_val_acc,
        "model_state_dict": eval_m.state_dict(),
        "param_hash": compute_param_hash(best_global_ndarrays),
    }, best_ckpt_path)

    set_parameters_to_model(eval_m, latest_global_ndarrays)
    torch.save({
        "experiment_id": experiment_id,
        "final_round": rounds,
        "model_state_dict": eval_m.state_dict(),
        "param_hash": compute_param_hash(latest_global_ndarrays),
    }, final_ckpt_path)

    return {
        "experiment_id": experiment_id,
        "duration_seconds": duration,
        "best_round": best_round,
        "best_val_acc": best_val_acc,
        "best_param_hash": compute_param_hash(best_global_ndarrays),
        "final_param_hash": compute_param_hash(latest_global_ndarrays),
        "partition_hash": partition_hash,
        "round_history": round_history,
        "feature_count": len(strategy.extracted_records),
    }


EXPERIMENT_MATRIX = [
    # 2 Calibration Honest Runs
    {
        "experiment_id": "EXP-MED-006-CTRL-IID",
        "scenario": "iid", "alpha": None, "attack_type": "none", "malicious_client_ids": [],
    },
    {
        "experiment_id": "EXP-MED-007-CTRL-NONIID-A05",
        "scenario": "non_iid", "alpha": 0.5, "attack_type": "none", "malicious_client_ids": [],
    },
    # 2 Held-Out Honest Controls (Phase 4 replayed)
    {
        "experiment_id": "EXP-MED-003-NONIID-ALPHA10",
        "scenario": "non_iid", "alpha": 1.0, "attack_type": "none", "malicious_client_ids": [],
    },
    {
        "experiment_id": "EXP-MED-005-NONIID-ALPHA01",
        "scenario": "non_iid", "alpha": 0.1, "attack_type": "none", "malicious_client_ids": [],
    },
    # 7 Phase 5 Attack Simulations
    {
        "experiment_id": "EXP-MED-008-ATK-LF-IID-M1",
        "scenario": "iid", "alpha": None, "attack_type": "label_flipping",
        "malicious_client_ids": [0], "flip_fraction": 1.0,
    },
    {
        "experiment_id": "EXP-MED-009-ATK-LF-IID-M2",
        "scenario": "iid", "alpha": None, "attack_type": "label_flipping",
        "malicious_client_ids": [0, 1], "flip_fraction": 1.0,
    },
    {
        "experiment_id": "EXP-MED-010-ATK-LF-NONIID-M1",
        "scenario": "non_iid", "alpha": 0.5, "attack_type": "label_flipping",
        "malicious_client_ids": [0], "flip_fraction": 1.0,
    },
    {
        "experiment_id": "EXP-MED-011-ATK-MP-INV-IID-M1",
        "scenario": "iid", "alpha": None, "attack_type": "model_poisoning",
        "malicious_client_ids": [0], "gamma": -1.0,
    },
    {
        "experiment_id": "EXP-MED-012-ATK-MP-SCALE-IID-M1",
        "scenario": "iid", "alpha": None, "attack_type": "model_poisoning",
        "malicious_client_ids": [0], "gamma": 10.0,
    },
    {
        "experiment_id": "EXP-MED-013-ATK-BD-IID-M1",
        "scenario": "iid", "alpha": None, "attack_type": "backdoor",
        "malicious_client_ids": [0], "poison_fraction": 0.50,
    },
    {
        "experiment_id": "EXP-MED-014-ATK-BD-NONIID-M1",
        "scenario": "non_iid", "alpha": 0.5, "attack_type": "backdoor",
        "malicious_client_ids": [0], "poison_fraction": 0.50,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Instrumented Replay Runner")
    parser.add_argument("--run-id", type=str, default=None, help="Run specific experiment by ID")
    args = parser.parse_args()

    targets = [cfg for cfg in EXPERIMENT_MATRIX if cfg["experiment_id"] == args.run_id] if args.run_id else EXPERIMENT_MATRIX

    print(f"\n==========================================================================================")
    print(f"  PHASE 6 INSTRUMENTATION SUITE: Executing {len(targets)} Federated Simulations")
    print(f"  Total planned client-round records: {len(targets) * 20 * 10:,}")
    print(f"==========================================================================================")

    suite_results = []
    total_start = datetime.now()

    for i, cfg in enumerate(targets, 1):
        jsonl_path = PHASE6_INSTRUMENTATION_DIR / f"{cfg['experiment_id'].lower()}_features.jsonl"
        best_ckpt = PHASE6_CHECKPOINTS_DIR / f"best_{cfg['experiment_id'].lower()}.pt"
        if jsonl_path.exists() and best_ckpt.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                lines = [l for l in f if l.strip()]
            if len(lines) >= 200:
                print(f"\n>>> Simulation {i}/{len(targets)}: {cfg['experiment_id']} already completed ({len(lines)} records). Skipping.")
                continue

        print(f"\n>>> Launching Simulation {i}/{len(targets)}: {cfg['experiment_id']}...")
        res = run_instrumented_simulation(**cfg)
        suite_results.append(res)

    total_time = (datetime.now() - total_start).total_seconds()
    print(f"\n==========================================================================================")
    print(f"  PHASE 6 INSTRUMENTATION SUITE COMPLETED in {total_time/60:.2f} minutes.")
    print(f"  All feature streams recorded in {PHASE6_INSTRUMENTATION_DIR}")
    print(f"==========================================================================================")

    # Write summary log
    summary_path = PHASE6_DIR / "instrumentation_suite_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_simulations": len(targets),
            "total_time_seconds": total_time,
            "simulations": suite_results,
        }, f, indent=2)
    print(f"  Suite summary saved to {summary_path}")


if __name__ == "__main__":
    main()
1