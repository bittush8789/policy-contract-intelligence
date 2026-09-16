"""BM25 keyword retriever module for lexical matching over pre-loaded document chunks.
"""

import logging
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever as LangChainBM25Retriever
from backend.config import settings
from backend.ingestion.ingest import load_chunks_cache
from backend.ingestion.loader import load_all_documents
from backend.ingestion.splitter import split_documents

logger = logging.getLogger(__name__)


class BM25RetrieverService:
    """BM25 Keyword Retriever over pre-loaded enterprise documents."""

    def __init__(self, documents: Optional[List[Document]] = None, k: int = settings.BM25_TOP_K):
        self.k = k
        self._retriever: Optional[LangChainBM25Retriever] = None

        if documents is not None:
            self._init_with_documents(documents)
        else:
            self._load_and_init()

    def _init_with_documents(self, documents: List[Document]) -> None:
        """Initialize BM25 with an explicit list of Document chunks."""
        if not documents:
            logger.warning("BM25 initialized with an empty document list.")
            self._retriever = None
            return

        self._retriever = LangChainBM25Retriever.from_documents(documents, k=self.k)
        logger.info(f"BM25 initialized with {len(documents)} document chunks.")

    def _load_and_init(self) -> None:
        """Load chunks from local cache or directly process documents from disk."""
        cache_file = settings.CHUNKS_CACHE_FILE
        chunks: List[Document] = []

        if cache_file.exists():
            chunks = load_chunks_cache(cache_file)
            logger.info(f"Loaded {len(chunks)} chunks from cache file for BM25.")

        if not chunks and settings.DATA_DIR.exists():
            logger.info(f"Cache not found; loading and splitting documents directly from {settings.DATA_DIR}...")
            pages = load_all_documents(settings.DATA_DIR)
            if pages:
                chunks = split_documents(pages, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

        self._init_with_documents(chunks)

    def search(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """Execute BM25 keyword search and return matching Document objects."""
        if self._retriever is None:
            logger.warning("BM25 retriever is not initialized; returning empty list.")
            return []

        limit = top_k if top_k is not None else self.k
        self._retriever.k = limit

        try:
            # invoke returns List[Document]
            results = self._retriever.invoke(query)
            logger.info(f"BM25 retrieved {len(results)} candidate chunks for query: '{query[:40]}...'")
            return results
        except Exception as e:
            logger.error(f"BM25 search error: {e}", exc_info=True)
            return []
