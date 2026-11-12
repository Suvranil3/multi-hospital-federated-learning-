"""
Phase 6 Smoke Test Script (Dedicated Engineering Seed 9999).
Strictly separated from blind evaluation:
- Uses dedicated engineering seed 9999 (NOT seeds 111, 112, 113).
- Uses validation data ONLY (strictly zero access to official test set).
- Verifies initialization, training, aggregation, detector execution, logging, and checkpoint saving.
- Does NOT contribute results to the blind evaluation report.
"""

import sys
from pathlib import Path
import json
import hashlib
import importlib.util
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from flwr.simulation import start_simulation
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import (
    DATA_RAW_DIR,
    RESULTS_MODELS_DIR,
    NUM_HOSPITALS,
    DEFAULT_BATCH_SIZE,
    set_seed,
)
from federated_learning.model import get_model
from federated_learning.client import (
    HospitalClient,
    get_parameters_from_model,
    set_parameters_to_model,
)
from federated_learning.dataset import prepare_federated_data_loaders
from federated_learning.strategy import get_server_eval_fn, aggregate_fit_metrics
from federated_learning.strategy_instrumented import InstrumentedFedAvg

SMOKE_SEED = 9999
FROZEN_DIR = PROJECT_ROOT / "federated_learning" / "detection" / "frozen_v1"


def verify_frozen_detector_hashes():
    manifest_path = FROZEN_DIR / "FREEZE_MANIFEST.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for filename, meta in manifest["files"].items():
        filepath = FROZEN_DIR / filename
        content = filepath.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        assert sha == meta["sha256"], f"Integrity check failed for {filename}"
    print("  [Verified] Frozen detector files verified against FREEZE_MANIFEST.json")


def load_frozen_detector():
    detector_path = FROZEN_DIR / "detector.py"
    thresholds_path = FROZEN_DIR / "calibrated_thresholds.json"
    spec = importlib.util.spec_from_file_location("frozen_detector", detector_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresh_dict = json.load(f)
    thresholds = module.DetectorThresholds(**thresh_dict)
    detector = module.TriStateDetector(thresholds=thresholds)
    return detector


def run_smoke_test():
    print("=" * 80)
    print("  PHASE 6 TECHNICAL SMOKE TEST (ENGINEERING SEED 9999)")
    print("  Validation Data Only — Zero Access to Test Split")
    print("=" * 80)

    # 1. Verify frozen detector hashes
    verify_frozen_detector_hashes()
    detector = load_frozen_detector()

    # 2. Data loading (validation data only for server eval)
    # Notice: we load data with seed 9999, and we discard test_loader immediately
    client_loaders, val_loader, _, _ = prepare_federated_data_loaders(
        data_dir=DATA_RAW_DIR,
        num_clients=NUM_HOSPITALS,
        batch_size=DEFAULT_BATCH_SIZE,
        num_workers=0,
        seed=SMOKE_SEED,
    )
    print(f"  [Verified] Loaded {len(client_loaders)} client loaders + 1 val loader. Test set ignored.")

    # 3. Model initialization and hash check
    device = torch.device("cpu")
    init_checkpoint_path = RESULTS_MODELS_DIR / "initial_federated_model.pt"
    init_payload = torch.load(init_checkpoint_path, map_location="cpu", weights_only=False)
    init_model = get_model(in_channels=1, num_classes=2, dropout_rate=0.2)
    init_model.load_state_dict(init_payload["model_state_dict"], strict=True)
    initial_ndarrays = get_parameters_from_model(init_model)

    hasher = hashlib.sha256()
    for arr in initial_ndarrays:
        hasher.update(arr.tobytes())
    param_hash = hasher.hexdigest()
    expected_hash = "0c411b83c2f0acef58beb5b9a968030d877e079eb2356eb3748903f8ae9bbc83"
    assert param_hash == expected_hash, f"Param hash mismatch: {param_hash}"
    print(f"  [Verified] Initial model parameter hash matches reference: {param_hash[:16]}...")

    # 4. Strategy, instrumentation, and temp logging
    smoke_output_dir = PROJECT_ROOT / "results" / "smoke_test"
    smoke_output_dir.mkdir(parents=True, exist_ok=True)
    smoke_jsonl = smoke_output_dir / "smoke_features.jsonl"
    if smoke_jsonl.exists():
        smoke_jsonl.unlink()

    initial_fl_parameters = ndarrays_to_parameters(initial_ndarrays)
    round_history = []
    base_eval_fn = get_server_eval_fn(val_loader=val_loader, device=device, record_history=round_history)

    def server_eval_fn(server_round: int, parameters, config):
        cur_ndarrays = parameters if isinstance(parameters, list) else parameters_to_ndarrays(parameters)
        return base_eval_fn(server_round, cur_ndarrays, config)

    strategy = InstrumentedFedAvg(
        val_loader=val_loader,
        device=device,
        output_jsonl_path=smoke_jsonl,
        initial_parameters=initial_fl_parameters,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=NUM_HOSPITALS,
        min_evaluate_clients=0,
        min_available_clients=NUM_HOSPITALS,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        on_fit_config_fn=lambda rnd: {"local_epochs": 1, "lr": 0.001, "round": rnd},
        evaluate_fn=server_eval_fn,
    )

    def client_fn(context):
        cid_int = int(context.node_config["partition-id"]) if hasattr(context, "node_config") else int(context)
        client = HospitalClient(
            client_id=str(cid_int),
            train_loader=client_loaders[cid_int],
            val_loader=None,
            local_epochs=1,
            learning_rate=0.001,
            weight_decay=1e-4,
            device=device,
        )
        return client.to_client()

    print("  [Executing] 1 round simulation with engineering seed 9999...")
    start_simulation(
        client_fn=client_fn,
        num_clients=NUM_HOSPITALS,
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
        client_resources={"num_cpus": 4, "num_gpus": 0.0},
        ray_init_args={"ignore_reinit_error": True, "include_dashboard": False, "_system_config": {"memory_monitor_refresh_ms": 0}},
    )

    # 5. Verify feature extraction and logging
    assert smoke_jsonl.exists(), "Smoke test jsonl was not created!"
    with open(smoke_jsonl, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == NUM_HOSPITALS, f"Expected {NUM_HOSPITALS} records, got {len(records)}"
    print(f"  [Verified] Instrumentation logged {len(records)} client records to {smoke_jsonl.name}")

    # 6. Verify detector execution on recorded features
    decisions = []
    for rec in records:
        res = detector.evaluate_client(rec)
        decisions.append(res.decision.value)
    print(f"  [Verified] Frozen detector evaluated {len(decisions)} records: {decisions}")

    # 7. Verify checkpoint saving
    smoke_ckpt = smoke_output_dir / "smoke_checkpoint.pt"
    torch.save({"model_state_dict": init_model.state_dict(), "round": 1}, smoke_ckpt)
    assert smoke_ckpt.exists(), "Smoke checkpoint failed to save"
    print(f"  [Verified] Checkpoint successfully saved to {smoke_ckpt.name}")

    # Clean up smoke test directory
    for f in smoke_output_dir.iterdir():
        f.unlink()
    smoke_output_dir.rmdir()
    print("  [Verified] Smoke test outputs cleaned up. Zero test set exposure.")
    print("=" * 80)
    print("  SMOKE TEST PASSED COMPLETELY.")
    print("=" * 80)


if __name__ == "__main__":
    run_smoke_test()
