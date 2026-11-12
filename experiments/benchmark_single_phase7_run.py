"""Representative 20-round timing benchmark for Phase 7."""
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.simulation import start_simulation
import ray

from federated_learning.model import get_model
from federated_learning.client import HospitalClient, get_parameters_from_model
from federated_learning.dataset import prepare_controlled_non_iid_data_loaders
from experiments.config import set_seed, DATA_RAW_DIR, RESULTS_MODELS_DIR
from federated_learning.strategy import aggregate_fit_metrics
from federated_learning.aggregation import create_robust_strategy
from experiments.run_phase6_blind_evaluation import load_frozen_detector

SEED = 9999
device = torch.device("cpu")
set_seed(SEED)

client_loaders, val_loader, test_loader, _ = prepare_controlled_non_iid_data_loaders(
    data_dir=DATA_RAW_DIR,
    num_clients=10,
    alpha=0.5,
    batch_size=32,
    num_workers=0,
    seed=SEED,
)

init_payload = torch.load(RESULTS_MODELS_DIR / "initial_federated_model.pt", map_location="cpu", weights_only=False)
init_model = get_model()
init_model.load_state_dict(init_payload["model_state_dict"])
init_ndarrays = get_parameters_from_model(init_model)
initial_fl_params = ndarrays_to_parameters(init_ndarrays)

detector, _ = load_frozen_detector()

strategy = create_robust_strategy(
    strategy_name="tri_state_discount_050",
    val_loader=val_loader,
    device=device,
    frozen_detector=detector,
    output_jsonl_path=None,
    initial_parameters=initial_fl_params,
    fraction_fit=1.0,
    fraction_evaluate=0.0,
    min_fit_clients=10,
    min_available_clients=10,
    fit_metrics_aggregation_fn=aggregate_fit_metrics,
    on_fit_config_fn=lambda rnd: {"local_epochs": 1, "lr": 0.001, "round": rnd, "training_seed": SEED},
    evaluate_fn=lambda rnd, p, c: (0.5, {"val_accuracy": 75.0}),
)

def client_fn(context):
    cid = int(context.node_config["partition-id"]) if hasattr(context, "node_config") and "partition-id" in context.node_config else int(context)
    return HospitalClient(
        client_id=str(cid),
        train_loader=client_loaders[cid],
        val_loader=None,
        local_epochs=1,
        learning_rate=0.001,
        weight_decay=1e-4,
        device=device,
    ).to_client()

print("Starting representative 20-round simulation with Tri-State Consensus (0.50 discount)...")
t0 = time.time()
start_simulation(
    client_fn=client_fn,
    num_clients=10,
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=strategy,
    client_resources={"num_cpus": 6, "num_gpus": 0.0},
    ray_init_args={"ignore_reinit_error": True, "include_dashboard": False, "_system_config": {"memory_monitor_refresh_ms": 0}},
)
ray.shutdown()
elapsed = time.time() - t0
print(f"Completed 20 rounds in {elapsed:.2f}s ({elapsed/60:.2f} minutes).")
print(f"Per-round average: {elapsed/20:.2f}s.")
