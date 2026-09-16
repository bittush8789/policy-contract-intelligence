"""Tests for document loading, section header detection, chunk splitting, and metadata preservation."""

from pathlib import Path
from langchain_core.documents import Document
from backend.ingestion.loader import (
    detect_section_header,
    infer_document_type,
    load_all_documents,
    load_pdf_file,
)
from backend.ingestion.splitter import clean_slug, split_documents
from backend.ingestion.ingest import save_chunks_cache, load_chunks_cache


def test_detect_section_header():
    """Test heuristic section header detection and safe fallback."""
    sample_text_1 = "Section 4: Termination Notice Period\nEither party may terminate..."
    assert detect_section_header(sample_text_1) == "Section 4: Termination Notice Period"

    sample_text_2 = "Article 12 - Indemnification Obligations\nVendor agrees to..."
    assert "Indemnification" in detect_section_header(sample_text_2)

    sample_text_3 = "Just regular paragraph content without any clear heading on this page."
    assert detect_section_header(sample_text_3) == "General"


def test_infer_document_type():
    """Test document type classification."""
    assert infer_document_type("vendor_contract.pdf") == "contract"
    assert infer_document_type("master_agreement.pdf") == "contract"
    assert infer_document_type("leave_policy.pdf") == "policy"
    assert infer_document_type("employee_handbook.pdf") == "policy"


def test_clean_slug():
    """Test slug generator for chunk IDs."""
    assert clean_slug("employee_handbook.pdf") == "employee_handbook"
    assert clean_slug("vendor-contract v2.0.pdf") == "vendor_contract_v2_0"


def test_split_documents_metadata_preservation():
    """Test RecursiveCharacterTextSplitter chunk generation and metadata fidelity."""
    sample_pages = [
        Document(
            page_content="Page 1 sentence one. " * 50,  # ~1150 chars, will split into >= 2 chunks
            metadata={
                "source": "leave_policy.pdf",
                "page": 1,
                "section": "Section 1: Annual Leave",
                "document_type": "policy",
            },
        ),
        Document(
            page_content="Page 2 brief content.",
            metadata={
                "source": "leave_policy.pdf",
                "page": 2,
                "section": "Section 2: Sick Leave",
                "document_type": "policy",
            },
        ),
    ]

    chunks = split_documents(sample_pages, chunk_size=400, chunk_overlap=50)
    assert len(chunks) >= 3

    # Check first chunk metadata
    first_chunk = chunks[0]
    assert first_chunk.metadata["source"] == "leave_policy.pdf"
    assert first_chunk.metadata["page"] == 1
    assert first_chunk.metadata["section"] == "Section 1: Annual Leave"
    assert first_chunk.metadata["document_type"] == "policy"
    assert first_chunk.metadata["chunk_id"] == "leave_policy_1_01"

    # Check second chunk of page 1
    second_chunk = chunks[1]
    assert second_chunk.metadata["chunk_id"] == "leave_policy_1_02"

    # Check page 2 chunk
    page_2_chunk = [c for c in chunks if c.metadata["page"] == 2][0]
    assert page_2_chunk.metadata["chunk_id"] == "leave_policy_2_01"
    assert page_2_chunk.metadata["section"] == "Section 2: Sick Leave"


def test_chunks_cache_roundtrip(tmp_path: Path):
    """Test saving and loading chunks from JSON cache."""
    test_chunks = [
        Document(
            page_content="Test content chunk 1",
            metadata={"source": "test.pdf", "page": 1, "chunk_id": "test_1_01"},
        ),
        Document(
            page_content="Test content chunk 2",
            metadata={"source": "test.pdf", "page": 2, "chunk_id": "test_2_01"},
        ),
    ]
    cache_file = tmp_path / "test_chunks.json"
    save_chunks_cache(test_chunks, cache_file)
    assert cache_file.exists()

    loaded = load_chunks_cache(cache_file)
    assert len(loaded) == 2
    assert loaded[0].page_content == test_chunks[0].page_content
    assert loaded[0].metadata["chunk_id"] == "test_1_01"
