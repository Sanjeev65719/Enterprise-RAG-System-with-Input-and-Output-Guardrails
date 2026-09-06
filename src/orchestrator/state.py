from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum

class Verdict(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"

class AuditEntry(BaseModel):
    """A single log entry for the pipeline audit trail."""
    step: str
    status: str
    reason: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PipelineState(BaseModel):
    """
    The comprehensive state of a SentinelRAG request.
    Tracks the lifecycle from ingress to final verdict.
    """
    request_id: UUID = Field(default_factory=uuid4)

    # Query Phase
    raw_query: str
    sanitized_query: Optional[str] = None

    # Retrieval Phase
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list)
    filtered_docs: List[Dict[str, Any]] = Field(default_factory=list)

    # Generation Phase
    generation_attempts: int = 0
    current_answer: Optional[str] = None
    correction_notes: Optional[str] = None

    # Audit & Verdict
    audit_log: List[AuditEntry] = Field(default_factory=list)
    is_faithful: bool = False
    final_verdict: Verdict = Verdict.ERROR

    def add_audit_log(self, step: str, status: str, reason: Optional[str] = None):
        """Adds an entry to the pipeline audit trail."""
        self.audit_log.append(AuditEntry(step=step, status=status, reason=reason))
