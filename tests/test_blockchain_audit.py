"""
Unit and Integration Tests for Blockchain Audit Component.
Verifies:
1. Genesis block creation.
2. Block hash calculation & deterministic serialization.
3. Previous-hash verification.
4. Tampered block detection (payload or hash modification).
5. Missing / deleted block detection.
6. Restart and persistence across sessions.
7. Interrupted write recovery from corrupted partial lines.
8. Duplicate experiment protection.
9. Result artifact hash verification on disk.
10. Federated learning audit integration (lifecycle logging).
"""

import pytest
import sys
from pathlib import Path
import json
import tempfile
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blockchain_audit.block import AuditBlock, GENESIS_PREVIOUS_HASH
from blockchain_audit.blockchain import BlockchainLedger
from blockchain_audit.audit_logger import AuditLogger, compute_file_sha256
from blockchain_audit.verification import verify_ledger


@pytest.fixture
def temp_ledger_dir(tmp_path):
    """Provide clean temporary directory for ledger testing."""
    return tmp_path / "blockchain_audit"


def test_genesis_block_creation():
    """Verify genesis block has index 0, GENESIS_PREVIOUS_HASH, and valid hash."""
    genesis = AuditBlock.create_genesis(experiment_id="TEST-GENESIS")
    assert genesis.index == 0
    assert genesis.previous_hash == GENESIS_PREVIOUS_HASH
    assert len(genesis.current_hash) == 64
    assert genesis.is_valid()
    assert genesis.experiment_id == "TEST-GENESIS"


def test_block_hash_calculation():
    """Verify deterministic hash calculation across equivalent dictionaries."""
    b1 = AuditBlock(
        index=1,
        timestamp="2026-11-02T12:00:00Z",
        previous_hash="a" * 64,
        experiment_id="EXP-01",
        simulation_seed=211,
        aggregation_strategy="fedavg",
        round_number=1,
        participating_hospitals=["1", "0"], # Unordered
        detector_decisions={"1": "NORMAL", "0": "NORMAL"},
        hospital_decisions={"1": "INCLUDED", "0": "INCLUDED"},
    )
    b2 = AuditBlock(
        index=1,
        timestamp="2026-11-02T12:00:00Z",
        previous_hash="a" * 64,
        experiment_id="EXP-01",
        simulation_seed=211,
        aggregation_strategy="fedavg",
        round_number=1,
        participating_hospitals=["0", "1"], # Ordered
        detector_decisions={"0": "NORMAL", "1": "NORMAL"},
        hospital_decisions={"0": "INCLUDED", "1": "INCLUDED"},
    )
    b1.seal()
    b2.seal()
    assert b1.current_hash == b2.current_hash
    assert b1.is_valid()


def test_previous_hash_linkage_and_append(temp_ledger_dir):
    """Verify blocks correctly chain backwards to previous block hashes."""
    ledger_path = temp_ledger_dir / "ledger.jsonl"
    ledger = BlockchainLedger(ledger_path)

    # Genesis automatically created
    assert ledger.block_count == 1
    assert ledger.latest_block.index == 0

    # Append block 1
    b1 = ledger.append_block(
        experiment_id="EXP-001",
        simulation_seed=211,
        aggregation_strategy="fedavg",
        round_number=1,
    )
    assert b1.index == 1
    assert b1.previous_hash == ledger.stream_blocks().__next__().current_hash

    # Append block 2
    b2 = ledger.append_block(
        experiment_id="EXP-001",
        simulation_seed=211,
        aggregation_strategy="fedavg",
        round_number=2,
    )
    assert b2.index == 2
    assert b2.previous_hash == b1.current_hash

    is_valid, msg, count = ledger.verify_integrity()
    assert is_valid
    assert count == 3


def test_tampered_block_detection(temp_ledger_dir):
    """Verify modifying a block's payload or hash causes verification failure."""
    ledger_path = temp_ledger_dir / "ledger.jsonl"
    ledger = BlockchainLedger(ledger_path)
    ledger.append_block(experiment_id="EXP-001", simulation_seed=211, aggregation_strategy="fedavg", round_number=1)
    ledger.append_block(experiment_id="EXP-001", simulation_seed=211, aggregation_strategy="fedavg", round_number=2)

    # Read all lines and tamper with block 1 payload
    lines = ledger_path.read_text(encoding="utf-8").strip().split("\n")
    block1_data = json.loads(lines[1])
    block1_data["aggregation_strategy"] = "TAMPERED_STRATEGY"
    lines[1] = json.dumps(block1_data)
    ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = verify_ledger(ledger_path)
    assert not result.is_valid
    assert result.corrupted_block_index == 1
    assert any("hash tampered" in err.lower() for err in result.errors)


def test_missing_or_deleted_block_detection(temp_ledger_dir):
    """Verify deleting a block breaks previous_hash linkage."""
    ledger_path = temp_ledger_dir / "ledger.jsonl"
    ledger = BlockchainLedger(ledger_path)
    ledger.append_block(experiment_id="EXP-001", round_number=1)
    ledger.append_block(experiment_id="EXP-001", round_number=2)
    ledger.append_block(experiment_id="EXP-001", round_number=3)

    # Delete block 1 (line index 1)
    lines = ledger_path.read_text(encoding="utf-8").strip().split("\n")
    del lines[1]
    ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = verify_ledger(ledger_path)
    assert not result.is_valid
    assert any("discontinuity" in err.lower() or "broken link" in err.lower() for err in result.errors)


def test_restart_and_persistence(temp_ledger_dir):
    """Verify ledger state survives process restarts and reloads tip."""
    ledger_path = temp_ledger_dir / "ledger.jsonl"
    l1 = BlockchainLedger(ledger_path)
    l1.append_block(experiment_id="EXP-A", round_number=1)
    l1.append_block(experiment_id="EXP-A", round_number=2)
    tip_hash = l1.latest_block.current_hash
    tip_index = l1.latest_block.index
    del l1

    # Reopen ledger
    l2 = BlockchainLedger(ledger_path)
    assert l2.block_count == 3
    assert l2.latest_block.index == tip_index
    assert l2.latest_block.current_hash == tip_hash

    # Can append smoothly after restart
    b3 = l2.append_block(experiment_id="EXP-A", round_number=3)
    assert b3.index == 3
    assert b3.previous_hash == tip_hash

    is_valid, _, count = l2.verify_integrity()
    assert is_valid
    assert count == 4


def test_interrupted_write_recovery(temp_ledger_dir):
    """Verify corrupted trailing partial lines are automatically recovered."""
    ledger_path = temp_ledger_dir / "ledger.jsonl"
    ledger = BlockchainLedger(ledger_path)
    ledger.append_block(experiment_id="EXP-01", round_number=1)
    valid_tip_hash = ledger.latest_block.current_hash

    # Simulate crash mid-write: append corrupted partial bytes
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write('{"index": 2, "timestamp": "2026-11-02T12:00:00Z", "previous_hash": "truncated...\n')

    # Re-instantiate ledger
    reopened = BlockchainLedger(ledger_path)
    assert reopened.block_count == 2
    assert reopened.latest_block.current_hash == valid_tip_hash
    assert reopened.latest_block.index == 1

    # Should be able to append new valid block
    b2 = reopened.append_block(experiment_id="EXP-01", round_number=2)
    assert b2.index == 2
    assert b2.previous_hash == valid_tip_hash
    assert reopened.verify_integrity()[0]


def test_duplicate_experiment_protection(temp_ledger_dir):
    """Verify verification engine flags duplicate final experiment result blocks."""
    ledger_path = temp_ledger_dir / "ledger.jsonl"
    logger = AuditLogger(ledger_path)

    fake_result_file = temp_ledger_dir / "fake_metrics.json"
    fake_result_file.write_text('{"accuracy": 85.0}', encoding="utf-8")

    # Commit completion once
    logger.log_experiment_result(
        experiment_id="EXP-TEST-DUP",
        simulation_seed=211,
        aggregation_strategy="fedavg",
        result_artifacts={"metrics": fake_result_file},
        summary_metrics={"accuracy": 85.0},
    )

    # Commit completion twice (duplicate)
    logger.log_experiment_result(
        experiment_id="EXP-TEST-DUP",
        simulation_seed=211,
        aggregation_strategy="fedavg",
        result_artifacts={"metrics": fake_result_file},
        summary_metrics={"accuracy": 85.0},
    )

    report = verify_ledger(ledger_path)
    assert "EXP-TEST-DUP" in report.duplicate_experiments
    assert any("duplicate final experiment" in w.lower() for w in report.warnings)


def test_result_artifact_hash_verification(temp_ledger_dir):
    """Verify artifact SHA-256 integrity detection against modified files on disk."""
    ledger_path = temp_ledger_dir / "ledger.jsonl"
    logger = AuditLogger(ledger_path)

    artifact_file = temp_ledger_dir / "model_results.json"
    artifact_file.write_text('{"status": "pristine"}', encoding="utf-8")

    logger.log_experiment_result(
        experiment_id="EXP-ART-01",
        simulation_seed=211,
        aggregation_strategy="fedavg",
        result_artifacts={"model_results.json": artifact_file},
        summary_metrics={"accuracy": 90.0},
    )

    # 1. Verify when pristine
    rep_pristine = verify_ledger(ledger_path, verify_artifacts_on_disk=True, artifacts_base_dir=temp_ledger_dir)
    assert rep_pristine.is_valid
    assert len(rep_pristine.artifact_mismatches) == 0

    # 2. Tamper with artifact file on disk
    artifact_file.write_text('{"status": "TAMPERED"}', encoding="utf-8")

    rep_tampered = verify_ledger(ledger_path, verify_artifacts_on_disk=True, artifacts_base_dir=temp_ledger_dir)
    assert not rep_tampered.is_valid
    assert len(rep_tampered.artifact_mismatches) == 1
    assert rep_tampered.artifact_mismatches[0]["experiment_id"] == "EXP-ART-01"


def test_fl_audit_integration_lifecycle(temp_ledger_dir):
    """Verify complete federated learning experiment lifecycle logging."""
    ledger_path = temp_ledger_dir / "ledger.jsonl"
    logger = AuditLogger(ledger_path)

    # 1. Init
    b_init = logger.log_experiment_init(
        experiment_id="EXP-P7-TEST-LIFECYCLE",
        simulation_seed=211,
        aggregation_strategy="tri_state_discount_050",
        condition_id="HONEST-IID",
        participating_hospitals=[str(i) for i in range(10)],
    )
    assert b_init is not None
    assert b_init.index == 1

    # 2. Round 1 Event
    b_r1 = logger.log_round_event(
        experiment_id="EXP-P7-TEST-LIFECYCLE",
        simulation_seed=211,
        aggregation_strategy="tri_state_discount_050",
        round_number=1,
        participating_hospitals=[str(i) for i in range(10)],
        detector_decisions={str(i): "NORMAL" for i in range(10)},
        hospital_decisions={str(i): "INCLUDED" for i in range(10)},
        aggregation_event_summary={"included_clients": 10, "aggregator": "tri_state_discount_050"},
    )
    assert b_r1 is not None
    assert b_r1.index == 2
    assert b_r1.previous_hash == b_init.current_hash

    # 3. Final Result
    dummy_art = temp_ledger_dir / "test_eval.json"
    dummy_art.write_text('{"test_accuracy": 82.5}', encoding="utf-8")
    b_res = logger.log_experiment_result(
        experiment_id="EXP-P7-TEST-LIFECYCLE",
        simulation_seed=211,
        aggregation_strategy="tri_state_discount_050",
        result_artifacts={"test_eval.json": dummy_art},
        summary_metrics={"test_accuracy": 82.5},
    )
    assert b_res is not None
    assert b_res.index == 3
    assert b_res.previous_hash == b_r1.current_hash

    # Verify complete chain
    report = verify_ledger(ledger_path, verify_artifacts_on_disk=True, artifacts_base_dir=temp_ledger_dir)
    assert report.is_valid
    assert report.total_blocks == 4  # genesis + init + round + result
