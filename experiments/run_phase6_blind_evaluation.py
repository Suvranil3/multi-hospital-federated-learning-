"""
Phase 6: Independent Blind Evaluation Runner and Evaluator.
Executes 33 fresh federated simulations across three untouched partition/training seeds
(111, 112, 113) spanning 4 honest conditions and 7 malicious attack configurations.

Evaluates all 6,600 client-round observations strictly using the FROZEN detector
(federated_learning/detection/frozen_v1/) with zero threshold re-calibration or post-hoc tuning.

Cryptographic Safeguards:
- Fixed initial model parameter hash: 0c411b83c2f0acef58beb5b9a968030d877e079eb2356eb3748903f8ae9bbc83
- Frozen detector SHA-256 verified before and after evaluation
- Test split (N=624) evaluated strictly post-training at the best validation checkpoint
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import argparse
import hashlib
from typing import Dict, Any, List, Optional
import importlib.util

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
from federated_learning.strategy import get_server_eval_fn, aggregate_fit_metrics
from federated_learning.strategy_instrumented import InstrumentedFedAvg
from federated_learning.attacks.label_flipping import LabelFlippedSubset
from federated_learning.attacks.model_poisoning import PoisonedHospitalClient
from federated_learning.attacks.backdoor import (
    BackdoorSubset,
    TriggeredPneumoniaTestDataset,
    evaluate_backdoor_asr,
)
from experiments.train_medical import evaluate, compute_medical_metrics

# Output Paths
BLIND_DIR = PROJECT_ROOT / "results" / "phase6_blind_evaluation"
BLIND_INSTRUMENTATION_DIR = BLIND_DIR / "instrumentation"
BLIND_CHECKPOINTS_DIR = BLIND_DIR / "checkpoints"
BLIND_EVALUATION_DIR = BLIND_DIR / "evaluation"

FROZEN_DIR = PROJECT_ROOT / "federated_learning" / "detection" / "frozen_v1"

BLIND_SEEDS = [111, 112, 113]


def verify_frozen_hashes() -> Dict[str, str]:
    """Verify SHA-256 hashes of frozen detector files."""
    manifest_path = FROZEN_DIR / "FREEZE_MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing freeze manifest at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    computed_hashes = {}
    for filename, meta in manifest["files"].items():
        filepath = FROZEN_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Missing frozen file {filepath}")
        content = filepath.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        expected_sha = meta["sha256"]
        if sha != expected_sha:
            raise ValueError(
                f"Frozen file integrity compromised for {filename}!\n"
                f"Expected: {expected_sha}\nGot:      {sha}"
            )
        computed_hashes[filename] = sha
    return computed_hashes


def load_frozen_detector():
    """Load detector dynamically from frozen_v1 directory to avoid importing mutable files."""
    detector_path = FROZEN_DIR / "detector.py"
    thresholds_path = FROZEN_DIR / "calibrated_thresholds.json"

    spec = importlib.util.spec_from_file_location("frozen_detector", detector_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresh_dict = json.load(f)

    thresholds = module.DetectorThresholds(**thresh_dict)
    detector = module.TriStateDetector(thresholds=thresholds)
    return detector, module.DetectionDecision


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


def run_blind_simulation(
    experiment_id: str,
    scenario: str = "iid",
    alpha: Optional[float] = None,
    attack_type: str = "none",
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
    partition_seed: int = 111,
    initialization_seed: int = 42,
    training_seed: int = 111,
) -> Dict[str, Any]:
    """Execute a single blind simulation and save feature stream and test metrics."""
    if malicious_client_ids is None:
        malicious_client_ids = []

    BLIND_INSTRUMENTATION_DIR.mkdir(parents=True, exist_ok=True)
    BLIND_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_output_path = BLIND_INSTRUMENTATION_DIR / f"{experiment_id.lower()}_features.jsonl"

    print("\n" + "=" * 90)
    print(f"  BLIND SIMULATION: {experiment_id}")
    print(f"  Scenario: {scenario.upper()} (alpha={alpha if scenario=='non_iid' else 'N/A'})")
    print(f"  Attack Type: {attack_type.upper()} | Malicious Clients: {malicious_client_ids}")
    print(f"  Partition Seed: {partition_seed} | Training Seed: {training_seed}")
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

    # Wrap malicious datasets
    attack_meta_records = {}
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
            attack_meta_records[cid] = poisoned_subset.get_attack_metadata()
            client_loaders[cid] = DataLoader(
                poisoned_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
            )

    # 2. Reference Initial Model Checkpoint
    init_checkpoint_path = RESULTS_MODELS_DIR / "initial_federated_model.pt"
    init_payload = torch.load(init_checkpoint_path, map_location="cpu", weights_only=False)
    init_model = get_model(in_channels=1, num_classes=2, dropout_rate=0.2)
    init_model.load_state_dict(init_payload["model_state_dict"], strict=True)
    init_model = init_model.to(device)

    initial_ndarrays = get_parameters_from_model(init_model)
    initial_param_hash = compute_param_hash(initial_ndarrays)
    expected_hash = "0c411b83c2f0acef58beb5b9a968030d877e079eb2356eb3748903f8ae9bbc83"
    assert initial_param_hash == expected_hash, f"Initial hash mismatch! Got: {initial_param_hash}"
    initial_fl_parameters = ndarrays_to_parameters(initial_ndarrays)

    set_seed(training_seed)

    round_history: List[Dict[str, Any]] = []
    latest_global_ndarrays: List[np.ndarray] = [arr.copy() for arr in initial_ndarrays]
    best_val_acc: float = 0.0
    best_global_ndarrays: List[np.ndarray] = [arr.copy() for arr in initial_ndarrays]
    best_round: int = 0

    base_eval_fn = get_server_eval_fn(val_loader=val_loader, device=device, record_history=round_history)

    def server_eval_fn(server_round: int, parameters: Any, config: Dict[str, fl.common.Scalar]):
        nonlocal best_val_acc, best_global_ndarrays, best_round, latest_global_ndarrays
        if isinstance(parameters, list):
            cur_ndarrays = parameters
        else:
            cur_ndarrays = parameters_to_ndarrays(parameters)
        latest_global_ndarrays = [arr.copy() for arr in cur_ndarrays]
        eval_res = base_eval_fn(server_round, cur_ndarrays, config)
        if eval_res is not None:
            val_loss, metrics = eval_res
            val_acc = float(metrics.get("val_accuracy", 0.0))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_round = server_round
                best_global_ndarrays = [arr.copy() for arr in cur_ndarrays]
        return eval_res

    # 3. Strategy Setup
    strategy = InstrumentedFedAvg(
        val_loader=val_loader,
        device=device,
        output_jsonl_path=jsonl_output_path,
        initial_parameters=initial_fl_parameters,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        on_fit_config_fn=lambda rnd: {"local_epochs": local_epochs, "lr": learning_rate, "round": rnd},
        evaluate_fn=server_eval_fn,
    )

    # 4. Client Generator
    def client_fn(context: Any) -> fl.client.Client:
        if hasattr(context, "node_config") and "partition-id" in context.node_config:
            cid_int = int(context.node_config["partition-id"])
        elif hasattr(context, "node_id"):
            cid_int = int(context.node_id)
        else:
            cid_int = int(context)

        client_loader = client_loaders[cid_int]
        is_malicious = cid_int in malicious_client_ids
        if is_malicious and attack_type == "model_poisoning":
            client = PoisonedHospitalClient(
                client_id=str(cid_int),
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
                client_id=str(cid_int),
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

    # 6. Save Checkpoints
    best_ckpt_path = BLIND_CHECKPOINTS_DIR / f"best_{experiment_id.lower()}.pt"
    final_ckpt_path = BLIND_CHECKPOINTS_DIR / f"final_{experiment_id.lower()}.pt"

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

    # 7. Post-Training Official Test Split Evaluation (N=624)
    print("  Evaluating Best Global Model on Official Test Set (N=624)...")
    eval_m = eval_m.to(device)
    set_parameters_to_model(eval_m, best_global_ndarrays)
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, test_targets, test_preds = evaluate(
        model=eval_m,
        loader=test_loader,
        criterion=criterion,
        device=device,
        return_predictions=True,
    )
    med_metrics = compute_medical_metrics(test_targets, test_preds)
    p_metrics = med_metrics["pneumonia_class_1"]
    n_metrics = med_metrics["normal_class_0"]

    test_results = {
        "clean_test_loss": round(float(test_loss), 4),
        "clean_test_accuracy": round(float(test_acc), 2),
        "pneumonia_recall": round(float(p_metrics.get("recall_sensitivity", p_metrics.get("recall", 0.0))), 4),
        "pneumonia_precision": round(float(p_metrics.get("precision", 0.0)), 4),
        "pneumonia_f1": round(float(p_metrics.get("f1_score", p_metrics.get("f1", 0.0))), 4),
        "normal_recall": round(float(n_metrics.get("recall_specificity", n_metrics.get("recall", 0.0))), 4),
        "normal_precision": round(float(n_metrics.get("precision", 0.0)), 4),
        "normal_f1": round(float(n_metrics.get("f1_score", n_metrics.get("f1", 0.0))), 4),
        "macro_f1": round(float(med_metrics.get("macro_averaged", {}).get("macro_f1", med_metrics.get("macro_f1", 0.0))), 4),
        "false_negatives": int(med_metrics["confusion_matrix"].get("false_negative_pneumonia_as_normal", 0)),
        "false_positives": int(med_metrics["confusion_matrix"].get("false_positive_normal_as_pneumonia", 0)),
    }

    # Evaluate backdoor ASR if backdoor attack
    if attack_type == "backdoor":
        print("  Evaluating Triggered Test Set ASR...")
        triggered_dataset = TriggeredPneumoniaTestDataset(clean_test_dataset=test_loader.dataset)
        triggered_loader = DataLoader(
            triggered_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        asr_metrics = evaluate_backdoor_asr(model=eval_m, triggered_loader=triggered_loader, device=device)
        test_results["backdoor_asr"] = asr_metrics

    print(f"  Test Metrics: CDA={test_results['clean_test_accuracy']}%, Pneu Recall={test_results['pneumonia_recall']*100:.1f}%, Norm Recall={test_results['normal_recall']*100:.1f}%, Macro F1={test_results['macro_f1']:.4f}")

    # Update best checkpoint with post-training test results and save evaluation json
    torch.save({
        "experiment_id": experiment_id,
        "best_round": best_round,
        "best_val_acc": best_val_acc,
        "model_state_dict": eval_m.state_dict(),
        "param_hash": compute_param_hash(best_global_ndarrays),
        "test_results": test_results,
    }, best_ckpt_path)

    BLIND_EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    eval_json_path = BLIND_EVALUATION_DIR / f"{experiment_id.lower()}_test_eval.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    return {
        "experiment_id": experiment_id,
        "duration_seconds": duration,
        "best_round": best_round,
        "best_val_acc": best_val_acc,
        "best_param_hash": compute_param_hash(best_global_ndarrays),
        "partition_hash": partition_hash,
        "round_history": round_history,
        "test_results": test_results,
        "feature_count": len(strategy.extracted_records),
    }


def generate_blind_experiment_matrix() -> List[Dict[str, Any]]:
    """Generate all 33 simulation configurations across seeds 111, 112, 113."""
    matrix = []
    for seed in BLIND_SEEDS:
        # 4 Honest Controls
        matrix.append({
            "experiment_id": f"EXP-BLIND-HONEST-IID-S{seed}",
            "scenario": "iid", "alpha": None, "attack_type": "none", "malicious_client_ids": [],
            "partition_seed": seed, "training_seed": seed, "initialization_seed": 42,
        })
        matrix.append({
            "experiment_id": f"EXP-BLIND-HONEST-A10-S{seed}",
            "scenario": "non_iid", "alpha": 1.0, "attack_type": "none", "malicious_client_ids": [],
            "partition_seed": seed, "training_seed": seed, "initialization_seed": 42,
        })
        matrix.append({
            "experiment_id": f"EXP-BLIND-HONEST-A05-S{seed}",
            "scenario": "non_iid", "alpha": 0.5, "attack_type": "none", "malicious_client_ids": [],
            "partition_seed": seed, "training_seed": seed, "initialization_seed": 42,
        })
        matrix.append({
            "experiment_id": f"EXP-BLIND-HONEST-A01-S{seed}",
            "scenario": "non_iid", "alpha": 0.1, "attack_type": "none", "malicious_client_ids": [],
            "partition_seed": seed, "training_seed": seed, "initialization_seed": 42,
        })

        # 7 Malicious Configurations
        matrix.append({
            "experiment_id": f"EXP-BLIND-ATK-LF-IID-M1-S{seed}",
            "scenario": "iid", "alpha": None, "attack_type": "label_flipping",
            "malicious_client_ids": [0], "flip_fraction": 1.0,
            "partition_seed": seed, "training_seed": seed, "initialization_seed": 42,
        })
        matrix.append({
            "experiment_id": f"EXP-BLIND-ATK-LF-IID-M2-S{seed}",
            "scenario": "iid", "alpha": None, "attack_type": "label_flipping",
            "malicious_client_ids": [0, 1], "flip_fraction": 1.0,
            "partition_seed": seed, "training_seed": seed, "initialization_seed": 42,
        })
        matrix.append({
            "experiment_id": f"EXP-BLIND-ATK-LF-NONIID-M1-S{seed}",
            "scenario": "non_iid", "alpha": 0.5, "attack_type": "label_flipping",
            "malicious_client_ids": [0], "flip_fraction": 1.0,
            "partition_seed": seed, "training_seed": seed, "initialization_seed": 42,
        })
        matrix.append({
            "experiment_id": f"EXP-BLIND-ATK-MP-INV-IID-M1-S{seed}",
            "scenario": "iid", "alpha": None, "attack_type": "model_poisoning",
            "malicious_client_ids": [0], "gamma": -1.0,
            "partition_seed": seed, "training_seed": seed, "initialization_seed": 42,
        })
        matrix.append({
            "experiment_id": f"EXP-BLIND-ATK-MP-SCALE-IID-M1-S{seed}",
            "scenario": "iid", "alpha": None, "attack_type": "model_poisoning",
            "malicious_client_ids": [0], "gamma": 10.0,
            "partition_seed": seed, "training_seed": seed, "initialization_seed": 42,
        })
        matrix.append({
            "experiment_id": f"EXP-BLIND-ATK-BD-IID-M1-S{seed}",
            "scenario": "iid", "alpha": None, "attack_type": "backdoor",
            "malicious_client_ids": [0], "poison_fraction": 0.50,
            "partition_seed": seed, "training_seed": seed, "initialization_seed": 42,
        })
        matrix.append({
            "experiment_id": f"EXP-BLIND-ATK-BD-NONIID-M1-S{seed}",
            "scenario": "non_iid", "alpha": 0.5, "attack_type": "backdoor",
            "malicious_client_ids": [0], "poison_fraction": 0.50,
            "partition_seed": seed, "training_seed": seed, "initialization_seed": 42,
        })
    return matrix


def evaluate_blind_features_with_frozen_detector(simulations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate all blind feature records using the frozen detector."""
    detector, DetectionDecision = load_frozen_detector()

    # Track overall metrics
    total_obs = 0
    total_honest = 0
    total_malicious = 0

    # Tri-state counts: actual -> pred
    tri_state = {
        "NORMAL": {"honest": 0, "malicious": 0},
        "SUSPICIOUS": {"honest": 0, "malicious": 0},
        "MALICIOUS": {"honest": 0, "malicious": 0},
    }

    # Per-seed metrics
    per_seed_stats = {}
    for seed in BLIND_SEEDS:
        per_seed_stats[str(seed)] = {
            "total_obs": 0, "honest_obs": 0, "malicious_obs": 0,
            "malicious_decisions": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
            "flagged_decisions": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
            "tri_state": {"NORMAL": {"honest": 0, "malicious": 0}, "SUSPICIOUS": {"honest": 0, "malicious": 0}, "MALICIOUS": {"honest": 0, "malicious": 0}},
        }

    # Per-attack metrics
    per_attack_stats = {}
    # Per-heterogeneity honest FPR
    per_hetero_honest = {
        "iid": {"total": 0, "malicious": 0, "suspicious": 0, "normal": 0},
        "non_iid_1.0": {"total": 0, "malicious": 0, "suspicious": 0, "normal": 0},
        "non_iid_0.5": {"total": 0, "malicious": 0, "suspicious": 0, "normal": 0},
        "non_iid_0.1": {"total": 0, "malicious": 0, "suspicious": 0, "normal": 0},
    }

    per_run_results = []

    for cfg in simulations:
        exp_id = cfg["experiment_id"]
        jsonl_path = BLIND_INSTRUMENTATION_DIR / f"{exp_id.lower()}_features.jsonl"
        if not jsonl_path.exists():
            continue

        seed = str(cfg["training_seed"])
        scenario = cfg["scenario"]
        alpha = cfg["alpha"]
        attack_type = cfg["attack_type"]
        mal_cids = [str(c) for c in cfg.get("malicious_client_ids", [])]

        if scenario == "iid":
            het_tag = "iid"
        elif alpha == 1.0:
            het_tag = "non_iid_1.0"
        elif alpha == 0.5:
            het_tag = "non_iid_0.5"
        elif alpha == 0.1:
            het_tag = "non_iid_0.1"
        else:
            het_tag = "non_iid_0.5"

        run_tp = 0
        run_fp = 0
        run_tn = 0
        run_fn = 0
        run_flagged_tp = 0
        run_flagged_fp = 0
        run_tri_state = {"NORMAL": 0, "SUSPICIOUS": 0, "MALICIOUS": 0}
        first_malicious_round = None
        first_suspicious_round = None

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line.strip())
                cid = str(rec["client_id"])
                rnd = int(rec["round"])
                is_malicious = cid in mal_cids

                total_obs += 1
                per_seed_stats[seed]["total_obs"] += 1

                res = detector.evaluate_client(rec)
                dec_str = res.decision.value
                run_tri_state[dec_str] += 1

                if is_malicious:
                    total_malicious += 1
                    per_seed_stats[seed]["malicious_obs"] += 1
                    tri_state[dec_str]["malicious"] += 1
                    per_seed_stats[seed]["tri_state"][dec_str]["malicious"] += 1

                    if res.decision == DetectionDecision.MALICIOUS:
                        run_tp += 1
                        per_seed_stats[seed]["malicious_decisions"]["tp"] += 1
                        if first_malicious_round is None:
                            first_malicious_round = rnd
                    else:
                        run_fn += 1
                        per_seed_stats[seed]["malicious_decisions"]["fn"] += 1

                    if res.decision in [DetectionDecision.SUSPICIOUS, DetectionDecision.MALICIOUS]:
                        run_flagged_tp += 1
                        per_seed_stats[seed]["flagged_decisions"]["tp"] += 1
                        if first_suspicious_round is None and res.decision == DetectionDecision.SUSPICIOUS:
                            first_suspicious_round = rnd
                    else:
                        per_seed_stats[seed]["flagged_decisions"]["fn"] += 1
                else:
                    total_honest += 1
                    per_seed_stats[seed]["honest_obs"] += 1
                    tri_state[dec_str]["honest"] += 1
                    per_seed_stats[seed]["tri_state"][dec_str]["honest"] += 1

                    per_hetero_honest[het_tag]["total"] += 1
                    if res.decision == DetectionDecision.MALICIOUS:
                        run_fp += 1
                        per_seed_stats[seed]["malicious_decisions"]["fp"] += 1
                        run_flagged_fp += 1
                        per_seed_stats[seed]["flagged_decisions"]["fp"] += 1
                        per_hetero_honest[het_tag]["malicious"] += 1
                    elif res.decision == DetectionDecision.SUSPICIOUS:
                        run_tn += 1
                        per_seed_stats[seed]["malicious_decisions"]["tn"] += 1
                        run_flagged_fp += 1
                        per_seed_stats[seed]["flagged_decisions"]["fp"] += 1
                        per_hetero_honest[het_tag]["suspicious"] += 1
                    else:
                        run_tn += 1
                        per_seed_stats[seed]["malicious_decisions"]["tn"] += 1
                        per_seed_stats[seed]["flagged_decisions"]["tn"] += 1
                        per_hetero_honest[het_tag]["normal"] += 1

        run_summary = {
            "experiment_id": exp_id,
            "seed": int(seed),
            "scenario": scenario,
            "alpha": alpha,
            "attack_type": attack_type,
            "malicious_clients": len(mal_cids),
            "total_observations": 200,
            "malicious_observations": len(mal_cids) * 20,
            "honest_observations": (10 - len(mal_cids)) * 20,
            "malicious_detected": run_tp,
            "honest_false_malicious": run_fp,
            "flagged_detected": run_flagged_tp,
            "flagged_honest": run_flagged_fp,
            "tri_state_counts": run_tri_state,
            "first_suspicious_round": first_suspicious_round,
            "first_malicious_round": first_malicious_round,
            "test_results": cfg.get("test_results", {}),
        }
        per_run_results.append(run_summary)

        # Per-attack category aggregation
        if attack_type != "none":
            atk_key = f"{attack_type}_{scenario}_{alpha}_m{len(mal_cids)}"
            if atk_key not in per_attack_stats:
                per_attack_stats[atk_key] = {
                    "attack_type": attack_type,
                    "scenario": scenario,
                    "alpha": alpha,
                    "m": len(mal_cids),
                    "total_malicious_obs": 0,
                    "detected_malicious": 0,
                    "detected_suspicious": 0,
                    "detected_normal": 0,
                }
            per_attack_stats[atk_key]["total_malicious_obs"] += len(mal_cids) * 20
            per_attack_stats[atk_key]["detected_malicious"] += run_tp
            per_attack_stats[atk_key]["detected_suspicious"] += (run_flagged_tp - run_tp)
            per_attack_stats[atk_key]["detected_normal"] += (len(mal_cids) * 20 - run_flagged_tp)

    # Compute aggregate metrics
    hard_tp = sum(r["malicious_detected"] for r in per_run_results)
    hard_fp = sum(r["honest_false_malicious"] for r in per_run_results)
    hard_tn = total_honest - hard_fp
    hard_fn = total_malicious - hard_tp

    flagged_tp = sum(r["flagged_detected"] for r in per_run_results)
    flagged_fp = sum(r["flagged_honest"] for r in per_run_results)
    flagged_tn = total_honest - flagged_fp
    flagged_fn = total_malicious - flagged_tp

    hard_tpr = round(hard_tp / total_malicious if total_malicious > 0 else 0.0, 4)
    hard_fpr = round(hard_fp / total_honest if total_honest > 0 else 0.0, 4)
    hard_prec = round(hard_tp / (hard_tp + hard_fp) if (hard_tp + hard_fp) > 0 else 0.0, 4)
    hard_f1 = round(2 * hard_prec * hard_tpr / (hard_prec + hard_tpr) if (hard_prec + hard_tpr) > 0 else 0.0, 4)

    flagged_tpr = round(flagged_tp / total_malicious if total_malicious > 0 else 0.0, 4)
    flagged_fpr = round(flagged_fp / total_honest if total_honest > 0 else 0.0, 4)
    flagged_prec = round(flagged_tp / (flagged_tp + flagged_fp) if (flagged_tp + flagged_fp) > 0 else 0.0, 4)
    flagged_f1 = round(2 * flagged_prec * flagged_tpr / (flagged_prec + flagged_tpr) if (flagged_prec + flagged_tpr) > 0 else 0.0, 4)

    return {
        "evaluation_name": "Phase 6 Independent Blind Evaluation (Untouched Seeds 111, 112, 113)",
        "timestamp": datetime.now().isoformat(),
        "total_simulations": len(per_run_results),
        "total_observations": total_obs,
        "total_honest_observations": total_honest,
        "total_malicious_observations": total_malicious,
        "overall_hard_malicious_metrics": {
            "true_positives": hard_tp,
            "false_positives": hard_fp,
            "true_negatives": hard_tn,
            "false_negatives": hard_fn,
            "tpr_recall": hard_tpr,
            "fpr": hard_fpr,
            "precision": hard_prec,
            "f1_score": hard_f1,
        },
        "overall_flagged_metrics": {
            "true_positives": flagged_tp,
            "false_positives": flagged_fp,
            "true_negatives": flagged_tn,
            "false_negatives": flagged_fn,
            "tpr_recall": flagged_tpr,
            "fpr": flagged_fpr,
            "precision": flagged_prec,
            "f1_score": flagged_f1,
        },
        "tri_state_confusion_matrix": tri_state,
        "heterogeneity_stratified_honest_rates": per_hetero_honest,
        "per_seed_summary": per_seed_stats,
        "per_attack_summary": per_attack_stats,
        "per_run_results": per_run_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Fresh Blind Evaluation Runner")
    parser.add_argument("--eval-only", action="store_true", help="Skip simulations and evaluate existing feature streams")
    args = parser.parse_args()

    print("\n" + "=" * 105)
    print("  PHASE 6 FRESH BLIND EVALUATION: PREFLIGHT INTEGRITY CHECK")
    print("=" * 105)

    # 1. Verify frozen detector hashes BEFORE execution
    print(">>> Verifying Frozen Detector Integrity (federated_learning/detection/frozen_v1/)...")
    pre_hashes = verify_frozen_hashes()
    for name, sha in pre_hashes.items():
        print(f"  [Verified] {name:28s} SHA-256: {sha}")

    matrix = generate_blind_experiment_matrix()
    print(f"\n>>> Total Planned Simulations: {len(matrix)} across seeds {BLIND_SEEDS}")
    print(f"    Expected Total Observations: {len(matrix) * 20 * 10:,} (6,120 Honest, 480 Malicious)")

    suite_results = []
    total_start = datetime.now()

    if not args.eval_only:
        for i, cfg in enumerate(matrix, 1):
            exp_id = cfg["experiment_id"]
            jsonl_path = BLIND_INSTRUMENTATION_DIR / f"{exp_id.lower()}_features.jsonl"
            best_ckpt = BLIND_CHECKPOINTS_DIR / f"best_{exp_id.lower()}.pt"

            if jsonl_path.exists() and best_ckpt.exists():
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    lines = [l for l in f if l.strip()]
                if len(lines) >= 200:
                    print(f"\n>>> Simulation {i}/{len(matrix)}: {exp_id} already exists ({len(lines)} records). Skipping execution.")
                    eval_json = BLIND_EVALUATION_DIR / f"{exp_id.lower()}_test_eval.json"
                    if eval_json.exists():
                        with open(eval_json, "r", encoding="utf-8") as f:
                            cfg["test_results"] = json.load(f)
                    else:
                        ckpt_data = torch.load(best_ckpt, map_location="cpu", weights_only=False)
                        cfg["test_results"] = ckpt_data.get("test_results", {})
                    suite_results.append(cfg)
                    continue

            print(f"\n>>> [{i}/{len(matrix)}] Executing Blind Simulation: {exp_id}...")
            res = run_blind_simulation(**cfg)
            cfg["test_results"] = res["test_results"]
            suite_results.append(cfg)
    else:
        suite_results = matrix

    sim_time = (datetime.now() - total_start).total_seconds()
    print(f"\n>>> All simulations verified/completed in {sim_time/60:.2f} minutes.")

    # 2. Re-Verify frozen detector hashes AFTER simulations
    print("\n>>> Re-Verifying Frozen Detector Integrity Post-Simulation...")
    post_hashes = verify_frozen_hashes()
    assert pre_hashes == post_hashes, "CRITICAL: Frozen detector files were modified during evaluation!"
    print("  [Verified] All frozen files remained strictly untouched.")

    # 3. Evaluate blind feature records with frozen detector
    print("\n>>> Evaluating 6,600 Blind Observations with Frozen Detector...")
    BLIND_EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    report = evaluate_blind_features_with_frozen_detector(suite_results)

    report_path = BLIND_EVALUATION_DIR / "blind_evaluation_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  [Saved] Blind evaluation report saved to {report_path}")

    # Summary manifest
    summary_path = BLIND_DIR / "blind_suite_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_simulations": len(matrix),
            "seeds": BLIND_SEEDS,
            "frozen_detector_hashes": post_hashes,
            "overall_hard_metrics": report["overall_hard_malicious_metrics"],
            "overall_flagged_metrics": report["overall_flagged_metrics"],
            "tri_state_confusion_matrix": report["tri_state_confusion_matrix"],
            "heterogeneity_stratified_honest_rates": report["heterogeneity_stratified_honest_rates"],
            "per_seed_summary": report["per_seed_summary"],
            "per_attack_summary": report["per_attack_summary"],
        }, f, indent=2)
    print(f"  [Saved] Blind suite summary saved to {summary_path}")

    # Print summary tables
    print("\n" + "=" * 105)
    print("  PHASE 6 FRESH BLIND EVALUATION SCORECARD (N = 6,600 OBSERVATIONS)")
    print("=" * 105)
    hm = report["overall_hard_malicious_metrics"]
    fm = report["overall_flagged_metrics"]
    print(f"  Hard MALICIOUS Decision:  TPR = {hm['tpr_recall']*100:.2f}% ({hm['true_positives']}/{report['total_malicious_observations']}) | FPR = {hm['fpr']*100:.2f}% ({hm['false_positives']}/{report['total_honest_observations']}) | Precision = {hm['precision']*100:.2f}% | F1 = {hm['f1_score']:.4f}")
    print(f"  Overall Flagged Decision: TPR = {fm['tpr_recall']*100:.2f}% ({fm['true_positives']}/{report['total_malicious_observations']}) | FPR = {fm['fpr']*100:.2f}% ({fm['false_positives']}/{report['total_honest_observations']}) | Precision = {fm['precision']*100:.2f}% | F1 = {fm['f1_score']:.4f}")

    tot_h = max(report['total_honest_observations'], 1)
    tot_m = max(report['total_malicious_observations'], 1)
    print("\n  TRI-STATE CONFUSION MATRIX:")
    ts = report["tri_state_confusion_matrix"]
    print(f"  Predicted NORMAL:     Honest = {ts['NORMAL']['honest']:5d} ({ts['NORMAL']['honest']/tot_h*100:.2f}%)  |  Malicious = {ts['NORMAL']['malicious']:3d} ({ts['NORMAL']['malicious']/tot_m*100:.2f}%)")
    print(f"  Predicted SUSPICIOUS: Honest = {ts['SUSPICIOUS']['honest']:5d} ({ts['SUSPICIOUS']['honest']/tot_h*100:.2f}%)  |  Malicious = {ts['SUSPICIOUS']['malicious']:3d} ({ts['SUSPICIOUS']['malicious']/tot_m*100:.2f}%)")
    print(f"  Predicted MALICIOUS:  Honest = {ts['MALICIOUS']['honest']:5d} ({ts['MALICIOUS']['honest']/tot_h*100:.2f}%)  |  Malicious = {ts['MALICIOUS']['malicious']:3d} ({ts['MALICIOUS']['malicious']/tot_m*100:.2f}%)")

    print("\n  STRATIFIED HONEST FALSE-POSITIVE RATES:")
    for het, counts in report["heterogeneity_stratified_honest_rates"].items():
        tot = max(counts["total"], 1)
        mal = counts["malicious"]
        sus = counts["suspicious"]
        print(f"    • {het:15s} ({counts['total']:4d} obs): False Malicious = {mal:2d} ({mal/tot*100:.2f}%) | Suspicious Alerts = {sus:3d} ({sus/tot*100:.2f}%)")

    print("\n  PER-SEED BREAKDOWN:")
    for seed, sstats in report["per_seed_summary"].items():
        tot_m = max(sstats["malicious_obs"], 1)
        tot_h = max(sstats["honest_obs"], 1)
        m_tp = sstats["malicious_decisions"]["tp"]
        m_fp = sstats["malicious_decisions"]["fp"]
        f_tp = sstats["flagged_decisions"]["tp"]
        f_fp = sstats["flagged_decisions"]["fp"]
        print(f"    • Seed {seed} ({sstats['total_obs']} obs): Malicious Recall = {m_tp/tot_m*100:.2f}% | Malicious FPR = {m_fp/tot_h*100:.2f}% | Flagged Recall = {f_tp/tot_m*100:.2f}% | Flagged FPR = {f_fp/tot_h*100:.2f}%")

    print("\n" + "=" * 105)


if __name__ == "__main__":
    main()
