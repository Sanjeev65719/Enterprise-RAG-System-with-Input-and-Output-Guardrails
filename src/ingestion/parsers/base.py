from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ParsedElement:
    """Standardized representation of a document element."""
    text: str
    element_type: str  # e.g., 'Title', 'NarrativeText', 'Table', 'ListItem'
    page_number: int = 1
    metadata: Dict[str, Any] = None

class AbstractParser(ABC):
    """Base class for all document parsers in SentinelRAG."""

    @abstractmethod
    def parse(self, file_path: str) -> List[ParsedElement]:
        """
        Parses a file and returns a list of standardized elements.

        Args:
            file_path: Path to the source file.

        Returns:
            A list of ParsedElement objects.
        """
        pass
