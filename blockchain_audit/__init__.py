"""
Blockchain Audit Package for Federated Learning.
Provides lightweight, tamper-evident cryptographic audit logs.
"""

from blockchain_audit.block import AuditBlock, GENESIS_PREVIOUS_HASH
from blockchain_audit.blockchain import BlockchainLedger
from blockchain_audit.audit_logger import AuditLogger, compute_file_sha256
from blockchain_audit.verification import verify_ledger, LedgerVerificationResult

__all__ = [
    "AuditBlock",
    "GENESIS_PREVIOUS_HASH",
    "BlockchainLedger",
    "AuditLogger",
    "compute_file_sha256",
    "verify_ledger",
    "LedgerVerificationResult",
]
