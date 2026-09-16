"""Citation formatting module.
Extracts and deduplicates structured citations from retrieved document metadata.
"""

from typing import Any, Dict, List, Set, Tuple
from langchain_core.documents import Document
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Structured citation item returned to API clients."""

    document: str = Field(..., description="Source document filename")
    page: int = Field(..., description="1-indexed page number in the document")
    section: str = Field(..., description="Section header or title")


def extract_citations(documents: List[Document]) -> List[Dict[str, Any]]:
    """Extract and deduplicate citations from a list of retrieved Document objects.
    Preserves first-seen order while preventing duplicate (document, page, section) entries.
    Never invents missing citation fields.
    """
    citations: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, int, str]] = set()

    for doc in documents:
        metadata = doc.metadata or {}
        doc_name = metadata.get("source", "Unknown Document")
        page_num = metadata.get("page", 1)
        section = metadata.get("section", "General")

        try:
            page_int = int(page_num)
        except (ValueError, TypeError):
            page_int = 1

        citation_key = (str(doc_name), page_int, str(section))
        if citation_key not in seen:
            seen.add(citation_key)
            citations.append(
                {
                    "document": str(doc_name),
                    "page": page_int,
                    "section": str(section),
                }
            )

    return citations
