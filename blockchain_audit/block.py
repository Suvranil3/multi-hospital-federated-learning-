"""
Audit Block Data Model for Federated Learning Blockchain Audit.
Implements deterministic SHA-256 block hashing and linked-block integrity.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import json
import hashlib
from datetime import datetime, timezone


GENESIS_PREVIOUS_HASH = "0" * 64


@dataclass
class AuditBlock:
    """
    Cryptographic block recording a federated learning lifecycle event.
    Deterministic JSON serialization ensures reproducible SHA-256 block hashing.
    """
    index: int
    timestamp: str
    previous_hash: str
    experiment_id: str
    simulation_seed: Optional[int] = None
    aggregation_strategy: Optional[str] = None
    round_number: Optional[int] = None
    participating_hospitals: List[str] = field(default_factory=list)
    detector_decisions: Dict[str, str] = field(default_factory=dict)
    hospital_decisions: Dict[str, str] = field(default_factory=dict)
    aggregation_event_summary: Dict[str, Any] = field(default_factory=dict)
    artifact_hashes: Dict[str, str] = field(default_factory=dict)
    current_hash: str = ""

    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash over deterministic serialization of block contents
        excluding current_hash.
        """
        payload = {
            "index": self.index,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "experiment_id": self.experiment_id,
            "simulation_seed": self.simulation_seed,
            "aggregation_strategy": self.aggregation_strategy,
            "round_number": self.round_number,
            "participating_hospitals": sorted(self.participating_hospitals),
            "detector_decisions": dict(sorted(self.detector_decisions.items())),
            "hospital_decisions": dict(sorted(self.hospital_decisions.items())),
            "aggregation_event_summary": self.aggregation_event_summary,
            "artifact_hashes": dict(sorted(self.artifact_hashes.items())),
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def seal(self) -> "AuditBlock":
        """Compute and set current_hash for this block."""
        self.current_hash = self.compute_hash()
        return self

    def is_valid(self) -> bool:
        """Check if block hash matches its deterministic content."""
        if not self.current_hash:
            return False
        return self.compute_hash() == self.current_hash

    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditBlock":
        """Construct AuditBlock from dictionary representation."""
        return cls(
            index=int(data["index"]),
            timestamp=str(data["timestamp"]),
            previous_hash=str(data["previous_hash"]),
            experiment_id=str(data["experiment_id"]),
            simulation_seed=data.get("simulation_seed"),
            aggregation_strategy=data.get("aggregation_strategy"),
            round_number=data.get("round_number"),
            participating_hospitals=list(data.get("participating_hospitals", [])),
            detector_decisions=dict(data.get("detector_decisions", {})),
            hospital_decisions=dict(data.get("hospital_decisions", {})),
            aggregation_event_summary=dict(data.get("aggregation_event_summary", {})),
            artifact_hashes=dict(data.get("artifact_hashes", {})),
            current_hash=str(data.get("current_hash", "")),
        )

    def to_json(self) -> str:
        """Deterministic single-line JSON string suitable for append-only ledger."""
        data = self.to_dict()
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def from_json(cls, json_str: str) -> "AuditBlock":
        """Parse AuditBlock from single-line JSON string."""
        return cls.from_dict(json.loads(json_str))

    @staticmethod
    def create_genesis(
        experiment_id: str = "GENESIS-AUDIT-LEDGER",
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> "AuditBlock":
        """Create the canonical genesis block (index 0)."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        meta = metadata or {"description": "PneumoniaMNIST Federated Learning Permissioned Audit Chain"}
        block = AuditBlock(
            index=0,
            timestamp=timestamp,
            previous_hash=GENESIS_PREVIOUS_HASH,
            experiment_id=experiment_id,
            simulation_seed=None,
            aggregation_strategy=None,
            round_number=None,
            participating_hospitals=[],
            detector_decisions={},
            hospital_decisions={},
            aggregation_event_summary=meta,
            artifact_hashes={},
        )
        block.seal()
        return block
