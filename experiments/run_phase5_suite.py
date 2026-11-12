"""
Phase 5 Master Simulation Driver (experiments/run_phase5_suite.py).
Sequentially executes the nine standardized Phase 5 simulations:
- 2 Matched Honest Controls (IID, Non-IID alpha=0.5)
- 7 Adversarial Attack Experiments (Label Flipping, Model Poisoning, Backdoor Injection)

Uses isolated subprocesses for each run to guarantee clean Ray environments and
deterministic execution order.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPERIMENTS = [
    # 1. Phase 5 Honest IID Control
    [
        sys.executable, "experiments/train_federated_attack.py",
        "--exp_id", "EXP-MED-006-CTRL-IID",
        "--scenario", "iid",
        "--attack_type", "none",
    ],
    # 2. Phase 5 Honest Non-IID Control (alpha=0.5)
    [
        sys.executable, "experiments/train_federated_attack.py",
        "--exp_id", "EXP-MED-007-CTRL-NONIID-A05",
        "--scenario", "non_iid",
        "--alpha", "0.5",
        "--attack_type", "none",
    ],
    # 3. Targeted Label Flipping: IID, M=1 (Hospital 0)
    [
        sys.executable, "experiments/train_federated_attack.py",
        "--exp_id", "EXP-MED-008-ATK-LF-IID-M1",
        "--scenario", "iid",
        "--attack_type", "label_flipping",
        "--malicious_clients", "0",
        "--flip_fraction", "1.0",
        "--control_id", "EXP-MED-006-CTRL-IID",
    ],
    # 4. Targeted Label Flipping: IID, M=2 (Hospitals 0, 1)
    [
        sys.executable, "experiments/train_federated_attack.py",
        "--exp_id", "EXP-MED-009-ATK-LF-IID-M2",
        "--scenario", "iid",
        "--attack_type", "label_flipping",
        "--malicious_clients", "0,1",
        "--flip_fraction", "1.0",
        "--control_id", "EXP-MED-006-CTRL-IID",
    ],
    # 5. Targeted Label Flipping: Non-IID alpha=0.5, M=1 (Hospital 0)
    [
        sys.executable, "experiments/train_federated_attack.py",
        "--exp_id", "EXP-MED-010-ATK-LF-NONIID-M1",
        "--scenario", "non_iid",
        "--alpha", "0.5",
        "--attack_type", "label_flipping",
        "--malicious_clients", "0",
        "--flip_fraction", "1.0",
        "--control_id", "EXP-MED-007-CTRL-NONIID-A05",
    ],
    # 6. Model Update Inversion: IID, M=1, gamma=-1.0 (Hospital 0)
    [
        sys.executable, "experiments/train_federated_attack.py",
        "--exp_id", "EXP-MED-011-ATK-MP-INV-IID-M1",
        "--scenario", "iid",
        "--attack_type", "model_poisoning",
        "--malicious_clients", "0",
        "--gamma", "-1.0",
        "--control_id", "EXP-MED-006-CTRL-IID",
    ],
    # 7. Model Update Scaling: IID, M=1, gamma=+10.0 (Hospital 0)
    [
        sys.executable, "experiments/train_federated_attack.py",
        "--exp_id", "EXP-MED-012-ATK-MP-SCALE-IID-M1",
        "--scenario", "iid",
        "--attack_type", "model_poisoning",
        "--malicious_clients", "0",
        "--gamma", "10.0",
        "--control_id", "EXP-MED-006-CTRL-IID",
    ],
    # 8. Backdoor Injection: IID, M=1, rho=0.50 (Hospital 0)
    [
        sys.executable, "experiments/train_federated_attack.py",
        "--exp_id", "EXP-MED-013-ATK-BD-IID-M1",
        "--scenario", "iid",
        "--attack_type", "backdoor",
        "--malicious_clients", "0",
        "--poison_fraction", "0.50",
        "--control_id", "EXP-MED-006-CTRL-IID",
    ],
    # 9. Backdoor Injection: Non-IID alpha=0.5, M=1, rho=0.50 (Hospital 0)
    [
        sys.executable, "experiments/train_federated_attack.py",
        "--exp_id", "EXP-MED-014-ATK-BD-NONIID-M1",
        "--scenario", "non_iid",
        "--alpha", "0.5",
        "--attack_type", "backdoor",
        "--malicious_clients", "0",
        "--poison_fraction", "0.50",
        "--control_id", "EXP-MED-007-CTRL-NONIID-A05",
    ],
]


def main() -> None:
    suite_start = datetime.now()
    print("=" * 90)
    print("STARTING PHASE 5 SIMULATION SUITE: 9 SEQUENTIAL FEDERATED EXPERIMENTS")
    print(f"Start Time: {suite_start.isoformat()}")
    print("=" * 90)

    for idx, cmd in enumerate(EXPERIMENTS, start=1):
        exp_id = cmd[cmd.index("--exp_id") + 1]
        print(f"\n[{idx}/9] Executing {exp_id}...")
        exp_start = datetime.now()
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        dur = (datetime.now() - exp_start).total_seconds()
        print(f"[{idx}/9] {exp_id} Completed in {dur:.2f}s (Exit code: {res.returncode})")

    total_dur = (datetime.now() - suite_start).total_seconds()
    print("\n" + "=" * 90)
    print(f"ALL 9 PHASE 5 EXPERIMENTS COMPLETED SUCCESSFULLY IN {total_dur/60:.2f} MINUTES")
    print("=" * 90)


if __name__ == "__main__":
    main()
