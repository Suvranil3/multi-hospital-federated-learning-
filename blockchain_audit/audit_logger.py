"""
Audit Logger for Federated Learning Experiment Lifecycle.
Bridges FL strategies, detector events, and experiment results to the blockchain ledger.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import hashlib
import logging

from blockchain_audit.blockchain import BlockchainLedger
from blockchain_audit.block import AuditBlock

logger = logging.getLogger("blockchain_audit")


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file on disk."""
    p = Path(filepath)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Artifact file not found: {p}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class AuditLogger:
    """
    High-level audit logger for federated learning experiments.
    Translates FL runtime events into tamper-evident linked audit blocks.
    Never crashes the simulation if a ledger write fails; logs warnings gracefully.
    """

    def __init__(self, ledger_path: Path):
        self.ledger_path = Path(ledger_path)
        self.ledger = BlockchainLedger(self.ledger_path)

    def log_experiment_init(
        self,
        experiment_id: str,
        simulation_seed: int,
        aggregation_strategy: str,
        condition_id: str,
        participating_hospitals: List[str],
        config_summary: Optional[Dict[str, Any]] = None,
    ) -> Optional[AuditBlock]:
        """
        Record experiment initialization event.
        """
        try:
            summary = {
                "event_type": "EXPERIMENT_INITIALIZATION",
                "condition_id": condition_id,
                "config": config_summary or {},
            }
            block = self.ledger.append_block(
                experiment_id=experiment_id,
                simulation_seed=simulation_seed,
                aggregation_strategy=aggregation_strategy,
                round_number=0,
                participating_hospitals=participating_hospitals,
                detector_decisions={},
                hospital_decisions={h: "INITIALIZED" for h in participating_hospitals},
                aggregation_event_summary=summary,
                artifact_hashes={},
            )
            return block
        except Exception as e:
            logger.warning(f"[AuditLogger] Failed to log experiment initialization for {experiment_id}: {e}")
            return None

    def log_round_event(
        self,
        experiment_id: str,
        simulation_seed: int,
        aggregation_strategy: str,
        round_number: int,
        participating_hospitals: List[str],
        detector_decisions: Dict[str, str],
        hospital_decisions: Dict[str, str],
        aggregation_event_summary: Dict[str, Any],
        checkpoint_path: Optional[Path] = None,
    ) -> Optional[AuditBlock]:
        """
        Record batched round-level lifecycle event:
        Hospital submissions, detector decisions, exclusions, and aggregator outcomes.
        """
        try:
            artifact_hashes = {}
            if checkpoint_path and Path(checkpoint_path).exists():
                artifact_hashes["round_checkpoint"] = compute_file_sha256(Path(checkpoint_path))

            summary = {
                "event_type": "ROUND_AGGREGATION_EVENT",
                **aggregation_event_summary,
            }
            block = self.ledger.append_block(
                experiment_id=experiment_id,
                simulation_seed=simulation_seed,
                aggregation_strategy=aggregation_strategy,
                round_number=round_number,
                participating_hospitals=participating_hospitals,
                detector_decisions=detector_decisions,
                hospital_decisions=hospital_decisions,
                aggregation_event_summary=summary,
                artifact_hashes=artifact_hashes,
            )
            return block
        except Exception as e:
            logger.warning(f"[AuditLogger] Failed to log round {round_number} for {experiment_id}: {e}")
            return None

    def log_experiment_result(
        self,
        experiment_id: str,
        simulation_seed: int,
        aggregation_strategy: str,
        result_artifacts: Dict[str, Path],
        summary_metrics: Dict[str, Any],
    ) -> Optional[AuditBlock]:
        """
        Commit final experiment completion and cryptographic artifact hashes.
        Guarantees that simulation results are verified and recorded.
        """
        try:
            artifact_hashes = {}
            for name, path in result_artifacts.items():
                p = Path(path)
                if p.exists() and p.is_file():
                    artifact_hashes[name] = compute_file_sha256(p)
                else:
                    artifact_hashes[name] = "FILE_NOT_FOUND"

            summary = {
                "event_type": "FINAL_EXPERIMENT_RESULT",
                "summary_metrics": summary_metrics,
            }
            block = self.ledger.append_block(
                experiment_id=experiment_id,
                simulation_seed=simulation_seed,
                aggregation_strategy=aggregation_strategy,
                round_number=None,
                participating_hospitals=[],
                detector_decisions={},
                hospital_decisions={},
                aggregation_event_summary=summary,
                artifact_hashes=artifact_hashes,
            )
            return block
        except Exception as e:
            logger.warning(f"[AuditLogger] Failed to log experiment completion for {experiment_id}: {e}")
            return None
