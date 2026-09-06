import magic
from typing import List, Dict, Any
from src.ingestion.parsers.base import AbstractParser, ParsedElement
from unstructured.partition.auto import partition

class MimeDetector:
    """Detects file types based on content using libmagic."""

    @staticmethod
    def detect(file_path: str) -> str:
        """Returns the MIME type of the file."""
        return magic.from_file(file_path, mime=True)

class UnstructuredParser(AbstractParser):
    """Enterprise-grade parser using Unstructured.io to handle diverse formats."""

    def parse(self, file_path: str) -> List[ParsedElement]:
        """
        Parses a file using Unstructured's automatic partitioning.

        This handles PDFs, DOCX, PPTX, HTML, and Markdown.
        """
        try:
            # partition() automatically detects the file type and applies the correct strategy
            elements = partition(filename=file_path)

            parsed_elements = []
            for i, el in enumerate(elements):
                # Extract metadata from Unstructured's element objects
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
