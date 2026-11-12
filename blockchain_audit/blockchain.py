"""
Blockchain Ledger for Federated Learning Audit Trails.
Provides disk-backed persistence, streaming verification, and interrupted write recovery.
"""

from pathlib import Path
from typing import Optional, Iterator, Tuple, Dict, Any, List
import threading
import os
import shutil

from blockchain_audit.block import AuditBlock, GENESIS_PREVIOUS_HASH


def stream_blocks_from_file(ledger_path: Path) -> Iterator[AuditBlock]:
    """Stream blocks one by one from disk without loading entire ledger into RAM."""
    p = Path(ledger_path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = line.strip()
            if clean_line:
                yield AuditBlock.from_json(clean_line)


class BlockchainLedger:
    """
    Disk-backed permissioned audit ledger.
    Stores linked cryptographic blocks sequentially in a JSONL ledger file.
    Does NOT keep all historical blocks in RAM; reads lazily or streams from disk.
    """

    def __init__(self, ledger_path: Path):
        self.ledger_path = Path(ledger_path)
        self.lock = threading.Lock()
        self._latest_block: Optional[AuditBlock] = None
        self._block_count: int = 0
        self._initialize_or_load()

    def _initialize_or_load(self) -> None:
        """Initialize new ledger with genesis block or load and recover existing ledger."""
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.ledger_path.exists() or self.ledger_path.stat().st_size == 0:
            genesis = AuditBlock.create_genesis()
            self._write_block_raw(genesis)
            self._latest_block = genesis
            self._block_count = 1
        else:
            self.recover_interrupted_write()

    def _write_block_raw(self, block: AuditBlock) -> None:
        """Append a sealed block to the ledger with immediate flush and fsync."""
        if not block.is_valid():
            block.seal()
        line = block.to_json() + "\n"
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def recover_interrupted_write(self) -> int:
        """
        Scan ledger file from the beginning.
        Detects truncated or corrupted trailing lines caused by system crashes.
        Truncates only trailing corrupted lines back to the last valid block if necessary.
        Returns the number of valid blocks found.
        """
        with self.lock:
            if not self.ledger_path.exists():
                return 0

            last_valid_byte_offset = 0
            expected_index = 0
            prev_hash: Optional[str] = None
            has_corrupted_trailing = False

            with open(self.ledger_path, "rb") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    line_str = line.decode("utf-8", errors="replace").strip()
                    if not line_str:
                        continue
                    try:
                        block = AuditBlock.from_json(line_str)
                        if expected_index == 0:
                            if block.index != 0 or block.previous_hash != GENESIS_PREVIOUS_HASH or not block.is_valid():
                                break
                        else:
                            if block.index != expected_index or block.previous_hash != prev_hash or not block.is_valid():
                                break

                        last_valid_byte_offset = f.tell()
                        prev_hash = block.current_hash
                        expected_index += 1
                        self._latest_block = block
                    except Exception:
                        has_corrupted_trailing = True
                        break

            file_size = self.ledger_path.stat().st_size
            if has_corrupted_trailing and last_valid_byte_offset < file_size:
                temp_backup = self.ledger_path.with_suffix(".corrupt.bak")
                shutil.copy2(self.ledger_path, temp_backup)
                with open(self.ledger_path, "r+b") as f:
                    f.truncate(last_valid_byte_offset)

            if expected_index == 0 and not has_corrupted_trailing and file_size > 0:
                # Existing file has invalid first block or middle corruption; do not overwrite, read latest valid if any
                pass
            elif expected_index == 0:
                genesis = AuditBlock.create_genesis()
                self._write_block_raw(genesis)
                self._latest_block = genesis
                expected_index = 1

            self._block_count = expected_index
            return self._block_count

    @property
    def latest_block(self) -> AuditBlock:
        """Return the most recent block at the tip of the chain."""
        if self._latest_block is None:
            self.recover_interrupted_write()
        assert self._latest_block is not None
        return self._latest_block

    @property
    def block_count(self) -> int:
        """Return total number of committed blocks in the ledger."""
        return self._block_count

    def append_block(
        self,
        experiment_id: str,
        simulation_seed: Optional[int] = None,
        aggregation_strategy: Optional[str] = None,
        round_number: Optional[int] = None,
        participating_hospitals: Optional[List[str]] = None,
        detector_decisions: Optional[Dict[str, str]] = None,
        hospital_decisions: Optional[Dict[str, str]] = None,
        aggregation_event_summary: Optional[Dict[str, Any]] = None,
        artifact_hashes: Optional[Dict[str, str]] = None,
        timestamp: Optional[str] = None,
    ) -> AuditBlock:
        """
        Create, link, seal, and append a new block to the ledger.
        Thread-safe and memory-efficient.
        """
        with self.lock:
            tip = self.latest_block
            new_index = tip.index + 1
            prev_hash = tip.current_hash
            ts = timestamp or AuditBlock.create_genesis().timestamp

            block = AuditBlock(
                index=new_index,
                timestamp=ts,
                previous_hash=prev_hash,
                experiment_id=experiment_id,
                simulation_seed=simulation_seed,
                aggregation_strategy=aggregation_strategy,
                round_number=round_number,
                participating_hospitals=participating_hospitals or [],
                detector_decisions=detector_decisions or {},
                hospital_decisions=hospital_decisions or {},
                aggregation_event_summary=aggregation_event_summary or {},
                artifact_hashes=artifact_hashes or {},
            )
            block.seal()
            self._write_block_raw(block)
            self._latest_block = block
            self._block_count += 1
            return block

    def stream_blocks(self) -> Iterator[AuditBlock]:
        """Stream blocks one by one from disk without loading entire ledger into RAM."""
        yield from stream_blocks_from_file(self.ledger_path)

    def verify_integrity(self) -> Tuple[bool, str, int]:
        """
        Full cryptographic verification of the ledger.
        Streams from disk in O(1) memory.
        Returns (is_valid, message, verified_block_count).
        """
        prev_hash: Optional[str] = None
        count = 0

        for block in self.stream_blocks():
            if not block.is_valid():
                return False, f"Block {block.index} hash mismatch! Computed: {block.compute_hash()}, Stored: {block.current_hash}", count

            if block.index != count:
                return False, f"Block {block.index} out of order! Expected index {count}.", count

            if count == 0:
                if block.previous_hash != GENESIS_PREVIOUS_HASH:
                    return False, f"Genesis block 0 has invalid previous_hash: {block.previous_hash}", count
            else:
                if block.previous_hash != prev_hash:
                    return False, f"Block {block.index} broken link! previous_hash {block.previous_hash} != {prev_hash}", count

            prev_hash = block.current_hash
            count += 1

        if count == 0:
            return False, "Ledger contains 0 blocks.", 0

        return True, f"Ledger integrity verified successfully ({count} blocks).", count
