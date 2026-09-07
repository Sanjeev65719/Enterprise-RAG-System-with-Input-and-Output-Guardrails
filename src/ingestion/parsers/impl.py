import mimetypes
import os
from typing import List, Dict, Any
from src.ingestion.parsers.base import AbstractParser, ParsedElement

try:
    from unstructured_client import UnstructuredClient
    from unstructured_client.models import PartitionParameters
    HAS_CLIENT = True
except ImportError:
    from unstructured.partition.auto import partition
    HAS_CLIENT = False

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
            self.client = UnstructuredClient(api_key=self.api_key)
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
                    res = self.client.general.partition(
                        files=[f],
                        partition_parameters=PartitionParameters(strategy="auto")
                    )
                    # The API returns a list of elements. We map them to ParsedElement.
                    return [
                        ParsedElement(
                            text=el.text,
                            element_type=el.type,
                            page_number=getattr(el.metadata, 'page_number', 1),
                            metadata={
                                "filename": os.path.basename(file_path),
                                "filetype": getattr(el.metadata, 'filetype', 'unknown'),
                            }
                        ) for el in res
                    ]
            else:
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

        except Exception as e:
            raise RuntimeError(f"Unstructured parsing failed for {file_path}: {str(e)}")
