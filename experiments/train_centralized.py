"""
Centralized Small CNN Baseline Training Pipeline.
Delegates to experiments/train_medical.py for the Primary Medical Project Scenario
(Multi-Hospital Chest X-Ray Classification on PneumoniaMNIST).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.train_medical import (
    prepare_medical_data_loaders,
    run_medical_centralized_training,
    CENTRALIZED_EPOCHS,
    CENTRALIZED_BATCH_SIZE,
    CENTRALIZED_LEARNING_RATE,
    SEED,
)

# Backward-compatibility alias for test functions
prepare_data_loaders = prepare_medical_data_loaders
run_centralized_training = run_medical_centralized_training

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Centralized Small CNN Baseline on PneumoniaMNIST")
    parser.add_argument("--epochs", type=int, default=CENTRALIZED_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=CENTRALIZED_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=CENTRALIZED_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, cpu)")
    args = parser.parse_args()

    run_medical_centralized_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        device_name=args.device,
    )
