"""
Blockchain Ledger Verification and Tamper Detection Engine.
Audits block cryptographic hashes, previous-hash linkage, artifact integrity, and duplicate protection.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse
import json
import sys

from blockchain_audit.block import AuditBlock, GENESIS_PREVIOUS_HASH
from blockchain_audit.audit_logger import compute_file_sha256


class LedgerVerificationResult:
    """Structured report of ledger integrity audit."""

    def __init__(self):
        self.is_valid: bool = True
        self.total_blocks: int = 0
        self.genesis_verified: bool = False
        self.broken_link_index: Optional[int] = None
        self.corrupted_block_index: Optional[int] = None
        self.duplicate_experiments: List[str] = []
        self.artifact_mismatches: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_blocks": self.total_blocks,
            "genesis_verified": self.genesis_verified,
            "broken_link_index": self.broken_link_index,
            "corrupted_block_index": self.corrupted_block_index,
            "duplicate_experiments": self.duplicate_experiments,
            "artifact_mismatches": self.artifact_mismatches,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def verify_ledger(
    ledger_path: Path,
    verify_artifacts_on_disk: bool = False,
    artifacts_base_dir: Optional[Path] = None,
) -> LedgerVerificationResult:
    """
    Stream ledger from disk and perform exhaustive cryptographic checks:
    1. Genesis block validity.
    2. Deterministic SHA-256 hash match on every block.
    3. Monotonic block index sequence.
    4. Exact previous_hash link from block i to block i-1.
    5. Duplicate final experiment completions.
    6. Optional artifact hash verification against files on disk.
    Strictly read-only; never alters or truncates the file.
    """
    result = LedgerVerificationResult()
    p = Path(ledger_path)

    if not p.exists() or p.stat().st_size == 0:
        result.is_valid = False
        result.errors.append(f"Ledger file not found or empty: {p}")
        return result

    prev_hash: Optional[str] = None
    seen_completed_experiments = set()
    count = 0

    with open(p, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                block = AuditBlock.from_json(line_str)
            except Exception as e:
                result.is_valid = False
                result.errors.append(f"Line {line_no} JSON parse error: {e}")
                return result

            # Check Genesis
            if count == 0:
                if block.index != 0 or block.previous_hash != GENESIS_PREVIOUS_HASH:
                    result.is_valid = False
                    result.errors.append(f"Invalid Genesis block: index={block.index}, prev_hash={block.previous_hash}")
                    return result
                if not block.is_valid():
                    result.is_valid = False
                    result.corrupted_block_index = 0
                    result.errors.append("Genesis block hash mismatch!")
                    return result
                result.genesis_verified = True
            else:
                # Check sequential index
                if block.index != count:
                    result.is_valid = False
                    result.errors.append(f"Block index discontinuity! Found index {block.index}, expected {count}.")
                    return result

                # Check hash self-integrity
                if not block.is_valid():
                    result.is_valid = False
                    result.corrupted_block_index = block.index
                    result.errors.append(
                        f"Block {block.index} hash tampered! Computed: {block.compute_hash()}, Stored: {block.current_hash}"
                    )
                    return result

                # Check previous_hash linkage
                if block.previous_hash != prev_hash:
                    result.is_valid = False
                    result.broken_link_index = block.index
                    result.errors.append(
                        f"Block {block.index} broken link! previous_hash={block.previous_hash} does not match predecessor={prev_hash}"
                    )
                    return result

            # Check for duplicate final experiment results
            summary = block.aggregation_event_summary or {}
            if summary.get("event_type") == "FINAL_EXPERIMENT_RESULT":
                exp_id = block.experiment_id
                if exp_id in seen_completed_experiments:
                    result.duplicate_experiments.append(exp_id)
                    result.warnings.append(f"Duplicate final experiment completion block detected for {exp_id}")
                seen_completed_experiments.add(exp_id)

                # Check artifact hashes on disk if requested
                if verify_artifacts_on_disk and block.artifact_hashes:
                    for art_name, expected_sha in block.artifact_hashes.items():
                        if expected_sha == "FILE_NOT_FOUND":
                            continue
                        target_path = None
                        if artifacts_base_dir:
                            cand = artifacts_base_dir / art_name
                            if cand.exists():
                                target_path = cand
                        if target_path is None and Path(art_name).exists():
                            target_path = Path(art_name)

                        if target_path and target_path.exists():
                            actual_sha = compute_file_sha256(target_path)
                            if actual_sha != expected_sha:
                                result.artifact_mismatches.append({
                                    "experiment_id": exp_id,
                                    "artifact": str(target_path),
                                    "expected_sha": expected_sha,
                                    "actual_sha": actual_sha,
                                })
                                result.errors.append(
                                    f"Artifact tamper detected for {target_path}! Expected {expected_sha}, got {actual_sha}"
                                )
                                result.is_valid = False

            prev_hash = block.current_hash
            count += 1

    result.total_blocks = count
    if count == 0:
        result.is_valid = False
        result.errors.append("Empty ledger.")
    return result


def main():
    parser = argparse.ArgumentParser(description="Blockchain Audit Ledger Verifier")
    parser.add_argument("--ledger-path", type=str, required=True, help="Path to audit_ledger.jsonl")
    parser.add_argument("--verify-artifacts", action="store_true", help="Verify referenced artifact SHA-256 hashes on disk")
    args = parser.parse_args()

    ledger_path = Path(args.ledger_path)
    res = verify_ledger(ledger_path=ledger_path, verify_artifacts_on_disk=args.verify_artifacts)

    print("\n" + "=" * 80)
    print("  FEDERATED LEARNING BLOCKCHAIN AUDIT LEDGER VERIFICATION")
    print("=" * 80)
    print(f"Ledger Path:      {ledger_path}")
    print(f"Total Blocks:     {res.total_blocks}")
    print(f"Genesis Verified: {res.genesis_verified}")
    print(f"Status:           {'VALID (PASSED)' if res.is_valid else 'COMPROMISED (FAILED)'}")

    if res.errors:
        print("\nERRORS:")
        for err in res.errors:
            print(f"  [X] {err}")

    if res.warnings:
        print("\nWARNINGS:")
        for w in res.warnings:
            print(f"  [!] {w}")

    print("=" * 80 + "\n")
    if not res.is_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
