"""Hybrid Retriever combining Pinecone semantic vector search and BM25 keyword search.
Applies Reciprocal Rank Fusion (RRF) for deterministic result merging and deduplication.
"""

import hashlib
import logging
from typing import Dict, List, Optional
from langchain_core.documents import Document
from backend.config import settings
from backend.retrieval.bm25_retriever import BM25RetrieverService
from backend.retrieval.pinecone_retriever import PineconeRetriever

logger = logging.getLogger(__name__)


def get_document_key(doc: Document) -> str:
    """Extract a unique identifier for a Document chunk for deduplication."""
    if "chunk_id" in doc.metadata and doc.metadata["chunk_id"]:
        return str(doc.metadata["chunk_id"])
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page", "")
    content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()[:12]
    return f"{source}_{page}_{content_hash}"


class HybridRetriever:
    """Hybrid Retriever combining dense semantic vector search and sparse BM25 keyword search."""

    def __init__(
        self,
        pinecone_retriever: Optional[PineconeRetriever] = None,
        bm25_retriever: Optional[BM25RetrieverService] = None,
        rrf_k: int = 60,
    ):
        self.pinecone_retriever = pinecone_retriever or PineconeRetriever()
        self.bm25_retriever = bm25_retriever or BM25RetrieverService()
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        vector_top_k: int = settings.VECTOR_TOP_K,
        bm25_top_k: int = settings.BM25_TOP_K,
        hybrid_top_k: int = settings.HYBRID_TOP_K,
    ) -> List[Document]:
        """Perform hybrid retrieval:
        1. Fetch vector_top_k from Pinecone.
        2. Fetch bm25_top_k from BM25.
        3. Merge candidates using Reciprocal Rank Fusion (RRF).
        4. Deduplicate candidates.
        5. Return top hybrid_top_k candidates.
        """
        logger.info(f"Starting hybrid retrieval for query: '{query[:50]}'")

        # 1. Fetch from both retrievers
        semantic_docs = self.pinecone_retriever.search(query, top_k=vector_top_k)
        bm25_docs = self.bm25_retriever.search(query, top_k=bm25_top_k)

        logger.info(
            f"Retrieval counts -> Pinecone: {len(semantic_docs)} chunks, BM25: {len(bm25_docs)} chunks"
        )

        # 2. Reciprocal Rank Fusion (RRF)
        # RRF Score = sum(1 / (k + rank)) across ranking systems
        doc_store: Dict[str, Document] = {}
        rrf_scores: Dict[str, float] = {}

        # Score semantic results
        for rank, doc in enumerate(semantic_docs, start=1):
            key = get_document_key(doc)
            doc_store[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (self.rrf_k + rank))

        # Score BM25 results
        for rank, doc in enumerate(bm25_docs, start=1):
            key = get_document_key(doc)
            if key not in doc_store:
                doc_store[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (self.rrf_k + rank))

        # 3. Sort by aggregated RRF score descending
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        # 4. Extract top candidate documents and attach hybrid score to metadata
        merged_candidates: List[Document] = []
        for key in sorted_keys[:hybrid_top_k]:
            doc = doc_store[key]
            # Copy metadata to preserve immutability
            meta = dict(doc.metadata)
            meta["hybrid_score"] = round(rrf_scores[key], 6)
            merged_candidates.append(
                Document(
                    page_content=doc.page_content,
                    metadata=meta,
                )
            )

        logger.info(
            f"Hybrid retrieval merged {len(merged_candidates)} unique candidate chunks (limit: {hybrid_top_k})"
        )
        return merged_candidates
