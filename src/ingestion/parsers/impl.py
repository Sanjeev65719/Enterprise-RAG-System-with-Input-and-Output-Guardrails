import mimetypes
import os
from typing import List, Dict, Any
from src.ingestion.parsers.base import AbstractParser, ParsedElement

try:
    from unstructured_client import UnstructuredClient
    from unstructured_client.models.shared import PartitionParameters, Files
    from unstructured_client.models.operations import PartitionRequest
    HAS_CLIENT = True
except ImportError:
    HAS_CLIENT = False

try:
    from unstructured.partition.auto import partition  # type: ignore
    HAS_LOCAL_PARTITION = True
except ImportError:
    partition = None
    HAS_LOCAL_PARTITION = False

class MimeDetector:
    """Detects file types based on extensions using the built-in mimetypes library."""

    @staticmethod
    def detect(file_path: str) -> str:
        """Returns the MIME type of the file based on its extension."""
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "application/octet-stream"

class UnstructuredParser(AbstractParser):
    """Enterprise-grade parser using Unstructured.io to handle diverse formats.
    Supports both local partitioning (fallback) and API-based partitioning (cloud-native).
    """

    def __init__(self):
        self.api_key = os.getenv("UNSTRUCTURED_API_KEY")
        if HAS_CLIENT and self.api_key:
            try:
                self.client = UnstructuredClient(api_key_auth=self.api_key)
            except TypeError:
                self.client = UnstructuredClient(api_key_auth=self.api_key)
        else:
            self.client = None

    def parse(self, file_path: str) -> List[ParsedElement]:
        """
        Parses a file using Unstructured's partitioning.
        Prefers the Cloud API to avoid system-level dependency requirements (libmagic, poppler, tesseract).
        """
        try:
            if self.client:
                # API-based parsing (Cloud Native)
                with open(file_path, "rb") as f:
                    file_bytes = f.read()

                try:
                    # Current unstructured-client SDK (v0.26+)
                    # `general.partition` ONLY accepts a `request=PartitionRequest(...)` keyword argument.
                    # Passing `files=[f]` and `partition_parameters=...` as bare kwargs is a v1 API style
                    # that no longer exists in this SDK version.
                    req = PartitionRequest(
                        partition_parameters=PartitionParameters(
                            files=Files(content=file_bytes, file_name=os.path.basename(file_path)),
                            strategy="auto"
                        )
                    )
                    res = self.client.general.partition(request=req)
                    elements = getattr(res, "elements", None) or []
                except Exception:
                    elements = []

                return [
                    ParsedElement(
                        text=getattr(el, 'text', str(el)),
                        element_type=getattr(el, 'type', 'NarrativeText'),
                        page_number=getattr(getattr(el, 'metadata', None), 'page_number', 1) or 1,
                        metadata={
                            "filename": os.path.basename(file_path),
                            "filetype": getattr(getattr(el, 'metadata', None), 'filetype', 'unknown') or 'unknown',
                        }
                    ) for el in elements
                ]
            elif HAS_LOCAL_PARTITION and partition is not None:
                # Local partitioning (Fallback - may fail on Vercel due to missing system deps)
                elements = partition(filename=file_path)
                parsed_elements = []
                for i, el in enumerate(elements):
                    metadata = getattr(el, 'metadata', {})
                    parsed_elements.append(ParsedElement(
                        text=str(el),
                        element_type=el.__class__.__name__,
                        page_number=metadata.get('page_number', 1),
                        metadata={
                            "filename": metadata.get('filename'),
                            "filetype": metadata.get('filetype'),
                            "last_modified": metadata.get('last_modified')
                        }
                    ))
                return parsed_elements
            else:
                # Lightweight native fallback: use pypdf for PDFs, plain-text reader for text/md/csv
                parsed_elements = []
                if file_path.lower().endswith(".pdf"):
                    try:
                        from pypdf import PdfReader
                    except ImportError:
                        raise RuntimeError("pypdf is required to parse PDF files. Please install pypdf.")

                    reader = PdfReader(file_path)
                    for page_idx, page in enumerate(reader.pages, start=1):
                        page_text = (page.extract_text() or "").strip()
                        if not page_text:
                            continue
                        paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
                        if not paragraphs:
                            paragraphs = [page_text]
                        for idx, para in enumerate(paragraphs):
                            parsed_elements.append(ParsedElement(
                                text=para,
                                element_type="NarrativeText",
                                page_number=page_idx,
                                metadata={
                                    "filename": os.path.basename(file_path),
                                    "filetype": "application/pdf",
                                    "section_index": idx
                                }
                            ))
                    return parsed_elements

                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()

                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                if not paragraphs and content.strip():
                    paragraphs = [content.strip()]

                for idx, para in enumerate(paragraphs):
                    elem_type = "Title" if para.startswith("#") else "NarrativeText"
                    parsed_elements.append(ParsedElement(
                        text=para,
                        element_type=elem_type,
                        page_number=1,
                        metadata={
                            "filename": os.path.basename(file_path),
                            "filetype": MimeDetector.detect(file_path),
                            "section_index": idx
                        }
                    ))
                return parsed_elements

        except Exception as e:
            raise RuntimeError(f"Unstructured parsing failed for {file_path}: {str(e)}")
