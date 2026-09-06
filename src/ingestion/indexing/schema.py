from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime

class DocumentMetadata(BaseModel):
    """Schema for high-assurance document metadata for vector indexing."""
    org_id: UUID = Field(..., description="Strict boundary identifier for multi-tenancy.")
    user_id: UUID = Field(..., description="Ownership identifier for private documents.")
    acl_tags: List[str] = Field(default_factory=list, description="Granular access tags (e.g., 'HR', 'Project-X').")
    doc_id: UUID = Field(default_factory=uuid4, description="Unique ID of the source document.")
    content_hash: str = Field(..., description="SHA-256 hash of the chunk content for duplicate detection.")
    source_url: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    additional_metadata: Dict[str, Any] = Field(default_factory=dict)

class VectorPayload(BaseModel):
    """The full payload stored alongside the vector in the DB."""
    text: str
    metadata: DocumentMetadata
