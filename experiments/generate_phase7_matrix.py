"""Generate the authoritative Phase 7 Experiment Matrix Manifest.

The matrix consists of:
11 conditions x 7 aggregation variants x 3 seeds (211, 212, 213) = 231 simulations.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "phase7_robust_aggregation"
OUTPUT_FILE = OUTPUT_DIR / "PHASE7_EXPERIMENT_MATRIX.json"

SEEDS = [211, 212, 213]

CONDITIONS = [
    {
        "condition_id": "HONEST-IID",
        "scenario": "iid",
        "alpha": None,
        "attack_type": "none",
        "malicious_client_ids": [],
        "attack_params": {},
    },
    {
        "condition_id": "HONEST-A10",
        "scenario": "non_iid",
        "alpha": 1.0,
        "attack_type": "none",
        "malicious_client_ids": [],
        "attack_params": {},
    },
    {
        "condition_id": "HONEST-A05",
        "scenario": "non_iid",
        "alpha": 0.5,
        "attack_type": "none",
        "malicious_client_ids": [],
        "attack_params": {},
    },
    {
        "condition_id": "HONEST-A01",
        "scenario": "non_iid",
        "alpha": 0.1,
        "attack_type": "none",
        "malicious_client_ids": [],
        "attack_params": {},
    },
    {
        "condition_id": "ATK-LF-IID-M1",
        "scenario": "iid",
        "alpha": None,
        "attack_type": "label_flipping",
        "malicious_client_ids": [0],
        "attack_params": {"flip_fraction": 1.0},
    },
    {
        "condition_id": "ATK-LF-IID-M2",
        "scenario": "iid",
        "alpha": None,
        "attack_type": "label_flipping",
        "malicious_client_ids": [0, 1],
        "attack_params": {"flip_fraction": 1.0},
    },
    {
        "condition_id": "ATK-LF-NONIID-M1",
        "scenario": "non_iid",
        "alpha": 0.5,
        "attack_type": "label_flipping",
        "malicious_client_ids": [0],
        "attack_params": {"flip_fraction": 1.0},
    },
    {
        "condition_id": "ATK-MP-INV-IID-M1",
        "scenario": "iid",
        "alpha": None,
        "attack_type": "model_poisoning",
        "malicious_client_ids": [0],
        "attack_params": {"gamma": -1.0},
    },
    {
        "condition_id": "ATK-MP-SCALE-IID-M1",
        "scenario": "iid",
        "alpha": None,
        "attack_type": "model_poisoning",
        "malicious_client_ids": [0],
        "attack_params": {"gamma": 10.0},
    },
    {
        "condition_id": "ATK-BD-IID-M1",
        "scenario": "iid",
        "alpha": None,
        "attack_type": "backdoor",
        "malicious_client_ids": [0],
        "attack_params": {"poison_fraction": 0.50},
    },
    {
        "condition_id": "ATK-BD-NONIID-M1",
        "scenario": "non_iid",
        "alpha": 0.5,
        "attack_type": "backdoor",
        "malicious_client_ids": [0],
        "attack_params": {"poison_fraction": 0.50},
    },
    {
        "condition_id": "ATK-BN-IID-M1",
        "scenario": "iid",
        "alpha": None,
        "attack_type": "batchnorm_poisoning",
        "malicious_client_ids": [0],
        "attack_params": {"poison_value": 1000000.0},
    },
    {
        "condition_id": "ATK-BN-NONIID-M1",
        "scenario": "non_iid",
        "alpha": 0.5,
        "attack_type": "batchnorm_poisoning",
        "malicious_client_ids": [0],
        "attack_params": {"poison_value": 1000000.0},
    },
]


AGGREGATION_VARIANTS = [
    {
        "variant_id": "fedavg",
        "name": "Standard FedAvg",
        "strategy_params": {},
    },
    {
        "variant_id": "coordinate_median",
        "name": "Coordinate-wise Median",
        "strategy_params": {},
    },
    {
        "variant_id": "trimmed_mean_beta1",
        "name": "Trimmed Mean (beta=1)",
        "strategy_params": {"beta": 1},
    },
    {
        "variant_id": "trimmed_mean_beta2",
        "name": "Trimmed Mean (beta=2)",
        "strategy_params": {"beta": 2},
    },
    {
        "variant_id": "multi_krum_f1",
        "name": "Multi-Krum (f=1, q=7, m=9)",
        "strategy_params": {"f": 1, "q": 7, "m": 9},
    },
    {
        "variant_id": "multi_krum_f2",
        "name": "Multi-Krum (f=2, q=6, m=8)",
        "strategy_params": {"f": 2, "q": 6, "m": 8},
    },
    {
        "variant_id": "tri_state_monitor",
        "name": "Tri-State Consensus (Monitor-Only)",
        "strategy_params": {"mode": "monitor_only", "suspicious_discount": 1.0},
    },
    {
        "variant_id": "tri_state_discount_050",
        "name": "Tri-State Consensus (Suspicious Discount 0.50)",
        "strategy_params": {"mode": "discount_050", "suspicious_discount": 0.50},
    },
    {
        "variant_id": "tri_state_discount_025",
        "name": "Tri-State Consensus (Suspicious Discount 0.25)",
        "strategy_params": {"mode": "discount_025", "suspicious_discount": 0.25},
    },
]


def generate_matrix() -> List[Dict[str, Any]]:
    matrix = []
    index = 1
    for seed in SEEDS:
        for cond in CONDITIONS:
            for agg in AGGREGATION_VARIANTS:
                agg_id = agg["variant_id"]
                cond_id = cond["condition_id"]
                exp_id = f"EXP-P7-{agg_id.upper()}-{cond_id}-S{seed}"
                
                entry = {
                    "index": index,
                    "experiment_id": exp_id,
                    "aggregation_variant": agg_id,
                    "aggregation_name": agg["name"],
                    "strategy_params": agg["strategy_params"],
                    "condition_id": cond_id,
                    "scenario": cond["scenario"],
                    "alpha": cond["alpha"],
                    "attack_type": cond["attack_type"],
                    "malicious_client_ids": cond["malicious_client_ids"],
                    "attack_params": cond["attack_params"],
                    "partition_seed": seed,
                    "training_seed": seed,
                    "initialization_seed": 42,
                    "num_clients": 10,
                    "rounds": 20,
                    "local_epochs": 1,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "weight_decay": 1e-4,
                }
                matrix.append(entry)
                index += 1
    return matrix


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix = generate_matrix()
    
    manifest = {
        "title": "Phase 7 Experiment Matrix: Byzantine-Robust Aggregation and Non-IID-Aware Tri-State Consensus",
        "total_simulations": len(matrix),
        "total_observations_expected": len(matrix) * 20 * 10,
        "conditions_count": len(CONDITIONS),
        "aggregation_variants_count": len(AGGREGATION_VARIANTS),
        "seeds": SEEDS,
        "initialization_seed": 42,
        "checkpoint_file_sha256": "80a2f4d2ca2e8b00a305852cdcf494889ed32a14ace98621f70d823d60438b68",
        "checkpoint_parameter_tensor_sha256": "0c411b83c2f0acef58beb5b9a968030d877e079eb2356eb3748903f8ae9bbc83",
        "experiments": matrix,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully generated Phase 7 manifest: {OUTPUT_FILE}")
    print(f"Total simulations: {len(matrix)}")
    print(f"Conditions: {len(CONDITIONS)}")
    print(f"Variants: {len(AGGREGATION_VARIANTS)}")
    print(f"Seeds: {SEEDS}")


if __name__ == "__main__":
    main()
