"""Offline ingestion pipeline for Enterprise RAG Assistant.
Scans data/documents/, extracts text, splits chunks, computes BGE embeddings,
upserts into Pinecone, and caches chunks locally for BM25 keyword retrieval.

Usage:
    python -m backend.ingestion.ingest
"""

import json
import logging
import sys
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from backend.config import settings
from backend.ingestion.embeddings import get_embedding_model
from backend.ingestion.loader import load_all_documents
from backend.ingestion.splitter import split_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingestion.ingest")


def save_chunks_cache(chunks: List[Document], cache_path: Path) -> None:
    """Save parsed chunks to a local JSON file for fast BM25 loading."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [
        {
            "page_content": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(chunks)} chunks to local cache at {cache_path}")


def load_chunks_cache(cache_path: Path) -> List[Document]:
    """Load cached document chunks from JSON file."""
    if not cache_path.exists():
        return []
    with open(cache_path, "r", encoding="utf-8") as f:
        serialized = json.load(f)
    return [
        Document(
            page_content=item["page_content"],
            metadata=item["metadata"],
        )
        for item in serialized
    ]


def upsert_to_pinecone(chunks: List[Document]) -> bool:
    """Upsert document chunks to Pinecone vector store."""
    if not settings.PINECONE_API_KEY or settings.PINECONE_API_KEY.startswith("your_"):
        logger.warning(
            "PINECONE_API_KEY is not configured in .env. "
            "Skipping remote Pinecone upsert. Chunks remain cached locally for BM25."
        )
        return False

    try:
        from pinecone import Pinecone, ServerlessSpec
        from langchain_pinecone import PineconeVectorStore

        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index_name = settings.PINECONE_INDEX_NAME

        existing_indexes = [idx["name"] for idx in pc.list_indexes()]
        embedding_dim = 384  # BAAI/bge-small-en-v1.5 dimension

        if index_name not in existing_indexes:
            logger.info(f"Creating new Pinecone index '{index_name}' with dimension {embedding_dim}...")
            pc.create_index(
                name=index_name,
                dimension=embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info(f"Index '{index_name}' created successfully.")
        else:
            logger.info(f"Using existing Pinecone index '{index_name}'.")

        embeddings = get_embedding_model()
        chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]

        logger.info(f"Upserting {len(chunks)} chunks to Pinecone namespace '{settings.PINECONE_NAMESPACE}'...")
        vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            namespace=settings.PINECONE_NAMESPACE,
            pinecone_api_key=settings.PINECONE_API_KEY,
        )

        vector_store.add_documents(documents=chunks, ids=chunk_ids)
        logger.info("Successfully upserted all chunks into Pinecone!")
        return True

    except Exception as e:
        logger.error(f"Error during Pinecone upsert: {e}", exc_info=True)
        return False


def run_ingestion() -> None:
    """Execute the complete offline document ingestion workflow."""
    docs_dir = settings.DATA_DIR
    logger.info("=" * 60)
    logger.info("Starting Enterprise Document Ingestion Pipeline")
    logger.info(f"Scanning directory: {docs_dir}")
    logger.info("=" * 60)

    if not docs_dir.exists():
        logger.error(f"Documents directory not found: {docs_dir}")
        sys.exit(1)

    # 1. Load PDFs
    pages = load_all_documents(docs_dir)
    if not pages:
        logger.warning(f"No documents found or extracted in {docs_dir}. Please place PDF files in the directory.")
        return

    # 2. Split into chunks with metadata
    chunks = split_documents(
        documents=pages,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    # 3. Cache chunks locally for BM25 and fallback
    save_chunks_cache(chunks, settings.CHUNKS_CACHE_FILE)

    # 4. Upsert to Pinecone
    upsert_success = upsert_to_pinecone(chunks)

    logger.info("=" * 60)
    logger.info("Ingestion Summary:")
    logger.info(f"- Source Pages:     {len(pages)}")
    logger.info(f"- Generated Chunks: {len(chunks)}")
    logger.info(f"- Chunks Cached:    {settings.CHUNKS_CACHE_FILE}")
    logger.info(f"- Pinecone Upsert:  {'Completed' if upsert_success else 'Skipped/Deferred'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_ingestion()
