"""Tests for BM25 retrieval, mock vector retrieval, and deterministic hybrid RRF merging."""

from unittest.mock import MagicMock
from langchain_core.documents import Document
from backend.retrieval.bm25_retriever import BM25RetrieverService
from backend.retrieval.hybrid_retriever import HybridRetriever, get_document_key


def test_bm25_retrieval_ranking(sample_documents):
    """Test that BM25 accurately ranks relevant documents by keyword matches."""
    bm25 = BM25RetrieverService(documents=sample_documents, k=3)
    results = bm25.search("termination notice agreement")

    assert len(results) > 0
    top_doc = results[0]
    assert top_doc.metadata["source"] == "vendor_contract.pdf"
    assert "thirty (30) days written notice" in top_doc.page_content
    assert top_doc.metadata["chunk_id"] == "vendor_contract_3_01"


def test_bm25_empty_query_or_corpus():
    """Test BM25 behavior when initialized with empty documents."""
    bm25 = BM25RetrieverService(documents=[], k=3)
    results = bm25.search("annual leave")
    assert results == []


def test_hybrid_retrieval_rrf_and_deduplication(sample_documents):
    """Test that hybrid retriever merges results from both sources and deduplicates chunks."""
    mock_pinecone = MagicMock()
    # Pinecone returns doc 0 (annual leave) and doc 2 (termination)
    mock_pinecone.search.return_value = [sample_documents[0], sample_documents[2]]

    mock_bm25 = MagicMock()
    # BM25 returns doc 2 (termination) and doc 3 (procurement)
    # Notice doc 2 appears in BOTH lists - testing deduplication!
    mock_bm25.search.return_value = [sample_documents[2], sample_documents[3]]

    hybrid = HybridRetriever(
        pinecone_retriever=mock_pinecone,
        bm25_retriever=mock_bm25,
        rrf_k=60,
    )

    merged = hybrid.retrieve("agreement notice", hybrid_top_k=5)

    # We started with 2 + 2 = 4 results, but doc 2 was in both, so 3 unique documents
    assert len(merged) == 3

    # Document 2 appeared in BOTH Pinecone (#2) and BM25 (#1), so its RRF score should be highest!
    top_doc = merged[0]
    assert top_doc.metadata["chunk_id"] == "vendor_contract_3_01"
    assert "hybrid_score" in top_doc.metadata
    assert top_doc.metadata["hybrid_score"] > 0

    # Ensure no duplicates exist in merged results
    chunk_ids = [doc.metadata["chunk_id"] for doc in merged]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_get_document_key(sample_documents):
    """Test unique document key generation for deduplication."""
    doc = sample_documents[0]
    key = get_document_key(doc)
    assert key == "leave_policy_1_01"

    # Test fallback key when chunk_id is missing
    fallback_doc = Document(page_content="Unique test content", metadata={"source": "test.pdf", "page": 4})
    fallback_key = get_document_key(fallback_doc)
    assert fallback_key.startswith("test.pdf_4_")
