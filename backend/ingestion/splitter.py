"""Document chunking module using RecursiveCharacterTextSplitter.
Preserves metadata and attaches unique deterministic chunk IDs.
"""

import logging
import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.config import settings

logger = logging.getLogger(__name__)


def clean_slug(name: str) -> str:
    """Generate a clean slug for chunk IDs."""
    base = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    slug = re.sub(r"[^\w]+", "_", base).strip("_").lower()
    return slug


def split_documents(
    documents: List[Document],
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> List[Document]:
    """Split a list of page-level Document objects into smaller chunks with preserved metadata
    and unique chunk IDs.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: List[Document] = []

    # Track chunk counts per document page to generate sequential chunk IDs
    page_chunk_counters: dict[tuple[str, int], int] = {}

    for doc in documents:
        source = doc.metadata.get("source", "unknown_doc")
        page = doc.metadata.get("page", 1)
        section = doc.metadata.get("section", "General")
        doc_type = doc.metadata.get("document_type", "policy")

        splits = splitter.split_text(doc.page_content)
        doc_slug = clean_slug(source)

        for text_chunk in splits:
            if not text_chunk.strip():
                continue

            counter_key = (source, page)
            current_index = page_chunk_counters.get(counter_key, 0) + 1
            page_chunk_counters[counter_key] = current_index

            chunk_id = f"{doc_slug}_{page}_{current_index:02d}"

            chunk_metadata = {
                "source": source,
                "page": page,
                "section": section,
                "document_type": doc_type,
                "chunk_id": chunk_id,
            }

            all_chunks.append(
                Document(
                    page_content=text_chunk,
                    metadata=chunk_metadata,
                )
            )

    logger.info(
        f"Split {len(documents)} page documents into {len(all_chunks)} chunks "
        f"(chunk_size={chunk_size}, chunk_overlap={chunk_overlap})"
    )
    return all_chunks
