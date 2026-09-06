from typing import List
from src.ingestion.parsers.base import ParsedElement

class DocumentNormalizer:
    """
    Converts a stream of ParsedElements into a standardized 'Sentinel-Common'
    Markdown representation to preserve structural hierarchy.
    """

    def normalize(self, elements: List[ParsedElement]) -> str:
        """
        Transforms parsed elements into a structured Markdown string.
        """
        markdown_output = []

        for el in elements:
            if el.element_type == "Title":
                # Treat Titles as H1 or H2
                markdown_output.append(f"# {el.text}\n")
            elif el.element_type == "ListItem":
                markdown_output.append(f"- {el.text}")
            elif el.element_type == "Table":
                # Unstructured often returns tables as text or HTML
                markdown_output.append(f"\n[TABLE]\n{el.text}\n[END TABLE]\n")
            else:
                # NarrativeText and others
                markdown_output.append(f"{el.text}\n")

        return "\n".join(markdown_output)
