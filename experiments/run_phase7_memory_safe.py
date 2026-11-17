"""
Phase 7 Memory-Safe Sequential Federated Learning Runner & Blockchain Audit Integration.
Enforces strictly sequential execution, minimal memory footprint (O(1) memory),
disk-backed checkpointing, resumable registry, and tamper-evident blockchain audit logging.
"""

import sys
from pathlib import Path
import os
import json
import time
import gc
import resource
import argparse
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from flwr.common import (
    FitRes,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from experiments.config import DATA_RAW_DIR, RESULTS_MODELS_DIR, set_seed
from federated_learning.model import get_model
from federated_learning.client import (
    HospitalClient,
    get_parameters_from_model,
    set_parameters_to_model,
)
from federated_learning.dataset import (
    prepare_federated_data_loaders,
    prepare_controlled_non_iid_data_loaders,
)
from federated_learning.aggregation.strategies import create_robust_strategy
from federated_learning.aggregation.base import (
    TRAINABLE_INDICES,
    BN_FLOAT_INDICES,
    BN_INT_INDICES,
)
from federated_learning.attacks.label_flipping import LabelFlippedSubset
from federated_learning.attacks.model_poisoning import PoisonedHospitalClient
from federated_learning.attacks.backdoor import (
    BackdoorSubset,
    TriggeredPneumoniaTestDataset,
    evaluate_backdoor_asr,
)
from federated_learning.attacks.batchnorm_poisoning import (
    BatchNormPoisonedHospitalClient,
)
from experiments.train_medical import evaluate, compute_medical_metrics
from experiments.run_phase6_blind_evaluation import load_frozen_detector
from blockchain_audit.audit_logger import AuditLogger, compute_file_sha256
from blockchain_audit.verification import verify_ledger

# Output Directories
OUTPUT_BASE_DIR = PROJECT_ROOT / "results" / "phase7_robust_aggregation"
CHECKPOINTS_DIR = OUTPUT_BASE_DIR / "checkpoints"
INSTRUMENTATION_DIR = OUTPUT_BASE_DIR / "instrumentation"
EVALUATION_DIR = OUTPUT_BASE_DIR / "evaluation"
REGISTRY_FILE = OUTPUT_BASE_DIR / "execution_registry.json"

BLOCKCHAIN_DIR = PROJECT_ROOT / "results" / "blockchain_audit"
LEDGER_FILE = BLOCKCHAIN_DIR / "audit_ledger.jsonl"

MATRIX_MANIFEST = OUTPUT_BASE_DIR / "PHASE7_EXPERIMENT_MATRIX.json"
INIT_CHECKPOINT_PATH = RESULTS_MODELS_DIR / "initial_federated_model.pt"

CRITICAL_RAM_THRESHOLD_MB = 250.0  # Stop launching if available RAM falls below 250 MB


class MockClientProxy(ClientProxy):
    """Direct in-process client proxy avoiding multi-process Ray engines."""
    def __init__(self, cid: str):
        super().__init__(cid)

    def get_properties(self, ins, timeout, group_id): raise NotImplementedError
    def get_parameters(self, ins, timeout, group_id): raise NotImplementedError
    def fit(self, ins, timeout, group_id): raise NotImplementedError
    def evaluate(self, ins, timeout, group_id): raise NotImplementedError
    def reconnect(self, ins, timeout, group_id): raise NotImplementedError


def get_memory_metrics() -> Dict[str, float]:
    """Read system and process memory metrics in MB using standard library."""
    meminfo = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    meminfo[parts[0].strip()] = int(parts[1].split()[0])
    except Exception:
        pass

    total_mb = meminfo.get("MemTotal", 0) / 1024.0
    avail_mb = meminfo.get("MemAvailable", 0) / 1024.0
    used_mb = total_mb - avail_mb
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    return {
        "system_total_mb": round(total_mb, 1),
        "system_used_mb": round(used_mb, 1),
        "system_available_mb": round(avail_mb, 1),
        "process_peak_rss_mb": round(rss_mb, 1),
    }


def compute_param_hash(ndarrays: List[np.ndarray]) -> str:
    """Compute deterministic SHA-256 over raw parameter bytes."""
    hasher = hashlib.sha256()
    for arr in ndarrays:
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


class ExecutionRegistry:
    """Persistent JSON registry tracking completed and in-progress simulations."""

    def __init__(self, registry_path: Path):
        self.path = Path(registry_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}
        else:
            self.data = {}

    def save(self) -> None:
        temp_path = self.path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        temp_path.replace(self.path)

    def is_completed(self, experiment_id: str) -> bool:
        rec = self.data.get(experiment_id)
        if not rec or rec.get("status") != "COMPLETED":
            return False
        eval_path = EVALUATION_DIR / f"{experiment_id.lower()}_results.json"
        best_ckpt = CHECKPOINTS_DIR / f"best_{experiment_id.lower()}.pt"
        if not eval_path.exists() or not best_ckpt.exists():
            return False
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                res_data = json.load(f)
            return "test_metrics" in res_data
        except Exception:
            return False

    def record_run(self, experiment_id: str, payload: Dict[str, Any]) -> None:
        self.data[experiment_id] = payload
        self.save()


def execute_single_simulation(
    exp_cfg: Dict[str, Any],
    audit_logger: Optional[AuditLogger] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Execute a single 20-round simulation strictly sequentially.
    Releases all per-round client models, data loaders, and tensors.
    """
    exp_id = exp_cfg["experiment_id"]
    scenario = exp_cfg["scenario"]
    alpha = exp_cfg.get("alpha")
    attack_type = exp_cfg["attack_type"]
    malicious_ids = set(exp_cfg.get("malicious_client_ids", []))
    attack_params = exp_cfg.get("attack_params", {})
    partition_seed = exp_cfg["partition_seed"]
    training_seed = exp_cfg["training_seed"]
    agg_variant = exp_cfg["aggregation_variant"]
    strategy_params = exp_cfg.get("strategy_params", {})
    num_rounds = exp_cfg.get("rounds", 20)
    local_epochs = exp_cfg.get("local_epochs", 1)
    batch_size = exp_cfg.get("batch_size", 32)
    lr = exp_cfg.get("learning_rate", 0.001)
    weight_decay = exp_cfg.get("weight_decay", 1e-4)

    if device is None:
        device = torch.device("cpu")

    mem_before = get_memory_metrics()
    t_start = time.time()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    INSTRUMENTATION_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = INSTRUMENTATION_DIR / f"{exp_id.lower()}_features.jsonl"
    best_ckpt_path = CHECKPOINTS_DIR / f"best_{exp_id.lower()}.pt"
    final_ckpt_path = CHECKPOINTS_DIR / f"final_{exp_id.lower()}.pt"
    eval_json_path = EVALUATION_DIR / f"{exp_id.lower()}_results.json"

    # 1. Dataset Partitioning
    if scenario == "iid":
        client_loaders, val_loader, test_loader, _ = prepare_federated_data_loaders(
            data_dir=DATA_RAW_DIR,
            num_clients=10,
            batch_size=batch_size,
            num_workers=0,
            seed=partition_seed,
        )
    else:
        client_loaders, val_loader, test_loader, _ = prepare_controlled_non_iid_data_loaders(
            data_dir=DATA_RAW_DIR,
            num_clients=10,
            alpha=alpha if alpha is not None else 0.5,
            batch_size=batch_size,
            num_workers=0,
            seed=partition_seed,
        )

    # Wrap dataset subsets for label flipping or backdoor attacks
    for cid in malicious_ids:
        orig_loader = client_loaders[cid]
        base_sub = orig_loader.dataset
        if attack_type == "label_flipping":
            poisoned_sub = LabelFlippedSubset(
                base_dataset=base_sub,
                source_class=1,
                target_class=0,
                flip_fraction=attack_params.get("flip_fraction", 1.0),
                seed=training_seed,
            )
            client_loaders[cid] = DataLoader(poisoned_sub, batch_size=batch_size, shuffle=True)
        elif attack_type == "backdoor":
            poisoned_sub = BackdoorSubset(
                base_dataset=base_sub,
                poison_fraction=attack_params.get("poison_fraction", 0.50),
                target_class=0,
                source_class=1,
                seed=training_seed,
            )
            client_loaders[cid] = DataLoader(poisoned_sub, batch_size=batch_size, shuffle=True)

    # 2. Reference Initial Model Checkpoint
    init_payload = torch.load(INIT_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    init_model = get_model(in_channels=1, num_classes=2, dropout_rate=0.2)
    init_model.load_state_dict(init_payload["model_state_dict"], strict=True)
    initial_ndarrays = get_parameters_from_model(init_model)
    initial_fl_params = ndarrays_to_parameters(initial_ndarrays)
    del init_model
    del init_payload

    # 3. Load Frozen Detector
    detector, _ = load_frozen_detector()

    # 4. Create Strategy
    krum_f = strategy_params.get("f", 1)
    krum_m = strategy_params.get("m", None)
    trimmed_beta = strategy_params.get("beta", 1)

    strategy = create_robust_strategy(
        strategy_name=agg_variant,
        val_loader=val_loader,
        device=device,
        frozen_detector=detector,
        output_jsonl_path=jsonl_path,
        initial_parameters=initial_fl_params,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=10,
        min_available_clients=10,
        krum_f=krum_f,
        krum_m=krum_m,
        trimmed_beta=trimmed_beta,
    )

    # Blockchain Log: Init
    if audit_logger:
        audit_logger.log_experiment_init(
            experiment_id=exp_id,
            simulation_seed=training_seed,
            aggregation_strategy=agg_variant,
            condition_id=exp_cfg.get("condition_id", "UNKNOWN"),
            participating_hospitals=[str(i) for i in range(10)],
            config_summary={"scenario": scenario, "alpha": alpha, "attack": attack_type},
        )

    # 5. Training Loop: 20 Rounds strictly sequential
    set_seed(training_seed)
    current_global_ndarrays = [arr.copy() for arr in initial_ndarrays]
    best_val_acc: float = -1.0
    best_round: int = 0
    best_global_ndarrays = [arr.copy() for arr in initial_ndarrays]
    peak_rss = mem_before["process_peak_rss_mb"]

    criterion = nn.CrossEntropyLoss()
    val_model = get_model(in_channels=1, num_classes=2, dropout_rate=0.2).to(device)

    for rnd in range(1, num_rounds + 1):
        round_client_results = []
        round_detector_decisions = {}
        round_hospital_decisions = {}

        for cid in range(10):
            cid_str = str(cid)
            loader = client_loaders[cid]
            is_mal = cid in malicious_ids

            if is_mal and attack_type == "model_poisoning":
                client = PoisonedHospitalClient(
                    client_id=cid_str,
                    train_loader=loader,
                    val_loader=None,
                    local_epochs=local_epochs,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    device=device,
                    gamma=attack_params.get("gamma", -1.0),
                    attack_start_round=1,
                )
            elif is_mal and attack_type == "batchnorm_poisoning":
                client = BatchNormPoisonedHospitalClient(
                    client_id=cid_str,
                    train_loader=loader,
                    val_loader=None,
                    local_epochs=local_epochs,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    device=device,
                    poison_value=attack_params.get("poison_value", 1_000_000.0),
                    attack_start_round=1,
                )
            else:
                client = HospitalClient(
                    client_id=cid_str,
                    train_loader=loader,
                    val_loader=None,
                    local_epochs=local_epochs,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    device=device,
                )

            # Fit client with current global parameters
            c_ndarrays, num_samples, fit_metrics = client.fit(
                parameters=[arr.copy() for arr in current_global_ndarrays],
                config={"server_round": rnd, "local_epochs": local_epochs, "round": rnd, "lr": lr},
            )

            fit_res = FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters(c_ndarrays),
                num_examples=num_samples,
                metrics={"client_id": cid_str, **fit_metrics},
            )
            proxy = MockClientProxy(cid=cid_str)
            round_client_results.append((proxy, fit_res))

            # Free client instance
            del client
            del c_ndarrays

        # Aggregate via robust strategy
        aggregated_fl_params, agg_summary = strategy.aggregate_fit(
            server_round=rnd,
            results=round_client_results,
            failures=[],
        )
        assert aggregated_fl_params is not None
        current_global_ndarrays = parameters_to_ndarrays(aggregated_fl_params)

        # Validation evaluation
        set_parameters_to_model(val_model, current_global_ndarrays)
        val_model.eval()
        v_loss, v_acc, _, _ = evaluate(val_model, val_loader, criterion, device)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_round = rnd
            best_global_ndarrays = [arr.copy() for arr in current_global_ndarrays]

        # Extract round decisions for blockchain
        telemetry = strategy.round_telemetry[-1] if strategy.round_telemetry else {}
        for r_rec in strategy.extracted_records[-10:]:
            cid_k = str(r_rec.get("client_id"))
            round_detector_decisions[cid_k] = r_rec.get("detector_decision", "NORMAL")
            round_hospital_decisions[cid_k] = "INCLUDED" if r_rec.get("effective_weight", 0.0) > 0 else "EXCLUDED"

        # Blockchain Log: Round
        if audit_logger:
            audit_logger.log_round_event(
                experiment_id=exp_id,
                simulation_seed=training_seed,
                aggregation_strategy=agg_variant,
                round_number=rnd,
                participating_hospitals=[str(i) for i in range(10)],
                detector_decisions=round_detector_decisions,
                hospital_decisions=round_hospital_decisions,
                aggregation_event_summary={
                    "round": rnd,
                    "val_loss": round(v_loss, 4),
                    "val_acc": round(v_acc, 2),
                    "included_clients": telemetry.get("included_clients", 10),
                    "aggregator": agg_variant,
                },
            )

        cur_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        if cur_rss > peak_rss:
            peak_rss = cur_rss

        del round_client_results
        # Truncate internal strategy in-memory list to save memory
        strategy.extracted_records.clear()

    del val_model

    # 6. Save Checkpoints
    test_eval_model = get_model(in_channels=1, num_classes=2, dropout_rate=0.2).to(device)
    set_parameters_to_model(test_eval_model, best_global_ndarrays)
    best_param_hash = compute_param_hash(best_global_ndarrays)

    torch.save({
        "experiment_id": exp_id,
        "best_round": best_round,
        "best_val_acc": best_val_acc,
        "model_state_dict": test_eval_model.state_dict(),
        "param_hash": best_param_hash,
    }, best_ckpt_path)

    set_parameters_to_model(test_eval_model, current_global_ndarrays)
    torch.save({
        "experiment_id": exp_id,
        "final_round": num_rounds,
        "model_state_dict": test_eval_model.state_dict(),
        "param_hash": compute_param_hash(current_global_ndarrays),
    }, final_ckpt_path)

    # 7. Evaluate on Pristine Test Set (624 images) using Best Checkpoint
    set_parameters_to_model(test_eval_model, best_global_ndarrays)
    test_eval_model.eval()
    t_loss, t_acc, t_targets, t_preds = evaluate(
        test_eval_model, test_loader, criterion, device, return_predictions=True
    )
    med_metrics = compute_medical_metrics(t_targets, t_preds)
    cm = med_metrics["confusion_matrix"]

    sens = med_metrics["pneumonia_class_1"]["recall_sensitivity"] * 100.0
    spec = med_metrics["normal_class_0"]["recall_specificity"] * 100.0
    f1 = med_metrics["macro_averaged"]["macro_f1"] * 100.0

    # Backdoor ASR evaluation if backdoor attack
    asr: Optional[float] = None
    if attack_type == "backdoor":
        triggered_dataset = TriggeredPneumoniaTestDataset(clean_test_dataset=test_loader.dataset)
        triggered_loader = DataLoader(
            triggered_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        asr_dict = evaluate_backdoor_asr(model=test_eval_model, triggered_loader=triggered_loader, device=device)
        asr = float(asr_dict["attack_success_rate_percent"])
        del triggered_loader
        del triggered_dataset

    del test_eval_model
    del client_loaders
    del val_loader
    del test_loader
    del strategy
    del detector

    gc.collect()
    mem_after = get_memory_metrics()
    duration = time.time() - t_start

    result_payload = {
        "experiment_id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "condition_id": exp_cfg.get("condition_id"),
        "scenario": scenario,
        "alpha": alpha,
        "attack_type": attack_type,
        "malicious_client_ids": list(malicious_ids),
        "aggregation_variant": agg_variant,
        "seed": training_seed,
        "rounds": num_rounds,
        "best_round": best_round,
        "best_val_acc": round(best_val_acc, 2),
        "test_metrics": {
            "test_accuracy": round(t_acc, 2),
            "sensitivity": round(sens, 2),
            "specificity": round(spec, 2),
            "macro_f1": round(f1, 2),
            "test_loss": round(t_loss, 4),
            "confusion_matrix": {
                "tn": cm["true_negative_normal"],
                "fp": cm["false_positive_normal_as_pneumonia"],
                "fn": cm["false_negative_pneumonia_as_normal"],
                "tp": cm["true_positive_pneumonia"],
            },
            "backdoor_asr": round(asr, 2) if asr is not None else None,
        },
        "performance": {
            "duration_seconds": round(duration, 2),
            "ram_before_mb": mem_before["system_used_mb"],
            "peak_rss_mb": round(peak_rss, 1),
            "ram_after_mb": mem_after["system_used_mb"],
        },
        "artifacts": {
            "best_checkpoint": str(best_ckpt_path),
            "final_checkpoint": str(final_ckpt_path),
            "features_jsonl": str(jsonl_path),
            "results_json": str(eval_json_path),
        },
    }

    # Save Results JSON
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    # 8. Blockchain Log: Final Result
    b_block = None
    if audit_logger:
        b_block = audit_logger.log_experiment_result(
            experiment_id=exp_id,
            simulation_seed=training_seed,
            aggregation_strategy=agg_variant,
            result_artifacts={
                "results_json": eval_json_path,
                "best_checkpoint": best_ckpt_path,
                "final_checkpoint": final_ckpt_path,
                "features_jsonl": jsonl_path,
            },
            summary_metrics=result_payload["test_metrics"],
        )

    result_payload["blockchain"] = {
        "committed": b_block is not None,
        "block_index": b_block.index if b_block else None,
        "block_hash": b_block.current_hash if b_block else None,
    }

    return result_payload


def run_matrix_suite(
    manifest_path: Path = MATRIX_MANIFEST,
    registry_path: Path = REGISTRY_FILE,
    max_experiments: Optional[int] = None,
    target_seed: Optional[int] = None,
    target_condition: Optional[str] = None,
    target_variant: Optional[str] = None,
    force_restart: bool = False,
    dry_run: bool = False,
) -> None:
    """Execute Phase 7 experiment matrix with memory safety and resumability."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Matrix manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    experiments = manifest["experiments"]
    registry = ExecutionRegistry(registry_path)
    audit_logger = AuditLogger(LEDGER_FILE)

    filtered = []
    for exp in experiments:
        if target_seed is not None and exp["training_seed"] != target_seed:
            continue
        if target_condition is not None and exp["condition_id"] != target_condition:
            continue
        if target_variant is not None and exp["aggregation_variant"] != target_variant:
            continue
        filtered.append(exp)

    print("\n" + "=" * 90)
    print(f"  PHASE 7 RESUMABLE MEMORY-SAFE RUNNER (Total Candidates: {len(filtered)}/{len(experiments)})")
    print(f"  Authoritative Manifest: {manifest_path}")
    print(f"  Execution Registry:     {registry_path}")
    print(f"  Blockchain Ledger:      {LEDGER_FILE}")
    print(f"  Critical RAM Limit:     {CRITICAL_RAM_THRESHOLD_MB} MB available")
    print("=" * 90 + "\n")

    executed_count = 0
    skipped_count = 0

    for i, exp in enumerate(filtered, 1):
        if max_experiments is not None and executed_count >= max_experiments:
            print(f">>> Reached maximum requested experiments limit ({max_experiments}). Stopping.")
            break

        exp_id = exp["experiment_id"]

        # Check completed in registry
        if not force_restart and registry.is_completed(exp_id):
            skipped_count += 1
            print(f"[{i}/{len(filtered)}] [SKIP] {exp_id} is already verified and completed.")
            continue

        # Check system available memory before run
        mem = get_memory_metrics()
        avail_mb = mem["system_available_mb"]
        if avail_mb < CRITICAL_RAM_THRESHOLD_MB:
            print(f"\n[CRITICAL WARNING] Available RAM ({avail_mb} MB) is below critical threshold ({CRITICAL_RAM_THRESHOLD_MB} MB)!")
            print("Safely pausing execution suite to prevent OS OOM crash. Completed work preserved.")
            break

        if dry_run:
            print(f"[{i}/{len(filtered)}] [DRY-RUN] Would execute {exp_id}")
            executed_count += 1
            continue

        print(f"\n[{i}/{len(filtered)}] >>> Executing {exp_id}...")
        print(f"       Condition: {exp['condition_id']} | Variant: {exp['aggregation_variant']} | Seed: {exp['training_seed']}")
        print(f"       RAM Before: {mem['system_used_mb']} MB used, {avail_mb} MB available | Process Peak RSS: {mem['process_peak_rss_mb']} MB")

        try:
            res = execute_single_simulation(
                exp_cfg=exp,
                audit_logger=audit_logger,
                device=torch.device("cpu"),
            )
            registry.record_run(exp_id, {
                "status": "COMPLETED",
                "condition_id": exp["condition_id"],
                "aggregation_variant": exp["aggregation_variant"],
                "seed": exp["training_seed"],
                "completed_at": res["timestamp"],
                "test_metrics": res["test_metrics"],
                "duration_seconds": res["performance"]["duration_seconds"],
                "blockchain_block": res["blockchain"]["block_index"],
                "blockchain_hash": res["blockchain"]["block_hash"],
            })
            executed_count += 1
            print(f"  [DONE] {exp_id} in {res['performance']['duration_seconds']}s | Test Acc: {res['test_metrics']['test_accuracy']}% | Sens: {res['test_metrics']['sensitivity']}% | Spec: {res['test_metrics']['specificity']}% | F1: {res['test_metrics']['macro_f1']}%")
            print(f"         RAM After: {res['performance']['ram_after_mb']} MB used | Blockchain Block #{res['blockchain']['block_index']}")

        except Exception as e:
            print(f"  [ERROR] Simulation {exp_id} failed: {e}")
            registry.record_run(exp_id, {
                "status": "FAILED",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            raise e

    print("\n" + "=" * 90)
    print(f"  PHASE 7 BATCH EXECUTION SUMMARY: {executed_count} executed, {skipped_count} skipped.")
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 7 Memory-Safe Sequential Federated Learning Runner")
    parser.add_argument("--manifest", type=str, default=str(MATRIX_MANIFEST), help="Path to experiment matrix manifest JSON")
    parser.add_argument("--registry", type=str, default=str(REGISTRY_FILE), help="Path to execution registry JSON")
    parser.add_argument("--max-experiments", type=int, default=None, help="Maximum number of experiments to run in this invocation")
    parser.add_argument("--seed", type=int, default=None, help="Filter by seed (211, 212, 213)")
    parser.add_argument("--condition", type=str, default=None, help="Filter by condition ID (e.g. HONEST-IID, ATK-BN-IID-M1)")
    parser.add_argument("--variant", type=str, default=None, help="Filter by aggregation variant (e.g. fedavg, tri_state_discount_050)")
    parser.add_argument("--force-restart", action="store_true", help="Force restart even if marked completed in registry")
    parser.add_argument("--dry-run", action="store_true", help="List experiments that would be run without training")
    parser.add_argument("--verify-blockchain", action="store_true", help="Verify blockchain ledger integrity")
    args = parser.parse_args()

    if args.verify_blockchain:
        res = verify_ledger(LEDGER_FILE, verify_artifacts_on_disk=True)
        print(f"Blockchain Ledger Valid: {res.is_valid} ({res.total_blocks} blocks)")
        if not res.is_valid:
            print(f"Errors: {res.errors}")
            sys.exit(1)
        sys.exit(0)

    run_matrix_suite(
        manifest_path=Path(args.manifest),
        registry_path=Path(args.registry),
        max_experiments=args.max_experiments,
        target_seed=args.seed,
        target_condition=args.condition,
        target_variant=args.variant,
        force_restart=args.force_restart,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
