"""Engineering-only Smoke Test for Phase 7 Robust Aggregation.

Executes a 1-round smoke test of all 7 aggregation variants using dedicated engineering seed 9999.
Verifies end-to-end integration:
- Data partition and client generation
- Feature extraction and frozen detector evaluation
- Aggregation across all 7 variants
- BatchNorm running stats and integer buffer preservation
- Test set evaluation and metrics computation
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.simulation import start_simulation
import ray

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from federated_learning.model import get_model
from federated_learning.client import HospitalClient, get_parameters_from_model, set_parameters_to_model
from federated_learning.dataset import prepare_controlled_non_iid_data_loaders
from experiments.config import set_seed
from federated_learning.strategy import aggregate_fit_metrics
from federated_learning.attacks.label_flipping import LabelFlippedSubset
from federated_learning.aggregation import (
    create_robust_strategy,
    TRAINABLE_INDICES,
    BN_FLOAT_INDICES,
    BN_INT_INDICES,
)
from experiments.run_phase6_blind_evaluation import load_frozen_detector
from experiments.train_medical import compute_medical_metrics

ENGINEERING_SEED = 9999
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
INIT_CHECKPOINT_PATH = PROJECT_ROOT / "results" / "models" / "initial_federated_model.pt"

SMOKE_VARIANTS = [
    "fedavg",
    "coordinate_median",
    "trimmed_mean_beta1",
    "trimmed_mean_beta2",
    "multi_krum_f1",
    "multi_krum_f2",
    "tri_state_monitor",
    "tri_state_discount_050",
    "tri_state_discount_025",
]


def run_single_variant_smoke(variant_name: str, client_loaders, val_loader, test_loader, init_ndarrays, frozen_detector):
    print(f"\n--- Testing Aggregation Variant: {variant_name.upper()} ---")
    device = torch.device("cpu")
    set_seed(ENGINEERING_SEED)

    # Initial fl parameters
    initial_fl_params = ndarrays_to_parameters(init_ndarrays)
    latest_params = [arr.copy() for arr in init_ndarrays]

    def server_eval(rnd, params, config):
        nonlocal latest_params
        if isinstance(params, list):
            latest_params = [p.copy() for p in params]
        else:
            latest_params = parameters_to_ndarrays(params)
        return 0.5, {"val_accuracy": 75.0}

    strategy = create_robust_strategy(
        strategy_name=variant_name,
        val_loader=val_loader,
        device=device,
        frozen_detector=frozen_detector,
        output_jsonl_path=None,
        initial_parameters=initial_fl_params,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=10,
        min_available_clients=10,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        on_fit_config_fn=lambda rnd: {"local_epochs": 1, "lr": 0.001, "round": rnd, "training_seed": ENGINEERING_SEED},
        evaluate_fn=server_eval,
    )

    def client_fn(context):
        cid_int = int(context.node_config["partition-id"]) if hasattr(context, "node_config") and "partition-id" in context.node_config else int(context)
        loader = client_loaders[cid_int]
        client = HospitalClient(
            client_id=str(cid_int),
            train_loader=loader,
            val_loader=None,
            local_epochs=1,
            learning_rate=0.001,
            weight_decay=1e-4,
            device=device,
        )
        return client.to_client()

    t0 = time.time()
    start_simulation(
        client_fn=client_fn,
        num_clients=10,
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
        client_resources={"num_cpus": 6, "num_gpus": 0.0},
        ray_init_args={
            "ignore_reinit_error": True,
            "include_dashboard": False,
            "_system_config": {"memory_monitor_refresh_ms": 0},
        },
    )
    ray.shutdown()
    elapsed = time.time() - t0

    # Verification 1: Trainable tensors updated and finite
    for idx in TRAINABLE_INDICES:
        assert np.all(np.isfinite(latest_params[idx])), f"Layer {idx} contains non-finite values!"
        assert not np.allclose(latest_params[idx], init_ndarrays[idx]), f"Layer {idx} did not update!"

    # Verification 2: BatchNorm float running stats are valid floats
    for idx in BN_FLOAT_INDICES:
        assert latest_params[idx].dtype in [np.float32, np.float64], f"BN float layer {idx} has wrong dtype {latest_params[idx].dtype}"
        assert np.all(np.isfinite(latest_params[idx])), f"BN layer {idx} has non-finite values"

    # Verification 3: BatchNorm integer counters are strictly integer
    for idx in BN_INT_INDICES:
        assert latest_params[idx].dtype in [np.int64, np.int32], f"BN counter layer {idx} has non-int dtype {latest_params[idx].dtype}"
        assert latest_params[idx] >= 0, f"BN counter {idx} is negative: {latest_params[idx]}"

    # Verification 4: Test set evaluation
    eval_model = get_model(in_channels=1, num_classes=2, dropout_rate=0.2)
    set_parameters_to_model(eval_model, latest_params)
    eval_model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            outputs = eval_model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(lbls.numpy())

    med_metrics = compute_medical_metrics(all_targets, all_preds)
    cm = med_metrics["confusion_matrix"]
    tn = cm["true_negative_normal"]
    fp = cm["false_positive_normal_as_pneumonia"]
    fn = cm["false_negative_pneumonia_as_normal"]
    tp = cm["true_positive_pneumonia"]
    tot = max(tn + fp + fn + tp, 1)
    acc = (tn + tp) / tot * 100.0
    sens = med_metrics["pneumonia_class_1"]["recall_sensitivity"] * 100.0
    spec = med_metrics["normal_class_0"]["recall_specificity"] * 100.0
    f1 = med_metrics["macro_averaged"]["macro_f1"] * 100.0

    # Verification 5: Telemetry recorded
    telemetry = strategy.round_telemetry[0]
    included = telemetry.get("included_clients", 10)
    print(f"  [PASS] {variant_name}: Completed in {elapsed:.1f}s | Included Clients: {included}/10 | Test Acc: {acc:.2f}% | Sens: {sens:.2f}% | Spec: {spec:.2f}% | F1: {f1:.2f}%")

    return {
        "variant": variant_name,
        "elapsed_sec": elapsed,
        "included_clients": included,
        "test_accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "macro_f1": f1,
        "telemetry": telemetry,
    }


def main():
    print("=" * 80)
    print(f"  PHASE 7 ENGINEERING SMOKE TEST (Seed {ENGINEERING_SEED})")
    print(f"  Testing all {len(SMOKE_VARIANTS)} Aggregation Variants on Non-IID alpha=0.5")
    print("=" * 80)

    # 1. Load data
    client_loaders, val_loader, test_loader, _ = prepare_controlled_non_iid_data_loaders(
        data_dir=DATA_RAW_DIR,
        num_clients=10,
        alpha=0.5,
        batch_size=32,
        num_workers=0,
        seed=ENGINEERING_SEED,
    )

    # Inject one label-flipping attacker on client 0 to exercise detector
    base_sub = client_loaders[0].dataset
    poisoned_sub = LabelFlippedSubset(
        base_dataset=base_sub,
        source_class=1,
        target_class=0,
        flip_fraction=1.0,
        seed=ENGINEERING_SEED,
    )
    client_loaders[0] = DataLoader(poisoned_sub, batch_size=32, shuffle=True)

    # 2. Initial Model Checkpoint
    init_payload = torch.load(INIT_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    init_model = get_model(in_channels=1, num_classes=2, dropout_rate=0.2)
    init_model.load_state_dict(init_payload["model_state_dict"], strict=True)
    init_ndarrays = get_parameters_from_model(init_model)

    # 3. Load Frozen Detector
    detector, _ = load_frozen_detector()

    results = []
    total_start = time.time()
    for variant in SMOKE_VARIANTS:
        res = run_single_variant_smoke(
            variant_name=variant,
            client_loaders=client_loaders,
            val_loader=val_loader,
            test_loader=test_loader,
            init_ndarrays=init_ndarrays,
            frozen_detector=detector,
        )
        results.append(res)

    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"  ALL {len(SMOKE_VARIANTS)} SMOKE TESTS PASSED IN {total_time:.1f}s!")
    print("=" * 80)


if __name__ == "__main__":
    main()
