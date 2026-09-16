"""Document loader using PyMuPDF (fitz) for page-level text extraction
and safe section header detection.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional
import fitz  # PyMuPDF
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Heuristic patterns for common enterprise sections (e.g. "Section 1: ...", "Article IV", "1.2 Scope")
SECTION_PATTERNS = [
    re.compile(r"^(?:Section|Article|Clause)\s+[\d\wIVXLCDM]+(?::|\s*[-–—])?\s*[A-Za-z0-9\s,&/'-]+", re.IGNORECASE),
    re.compile(r"^\d+\.\d+(?:\.\d+)?\s+[A-Z][A-Za-z0-9\s,&/'-]+"),
    re.compile(r"^[A-Z\s]{4,40}$"),  # Short UPPERCASE titles
]


def detect_section_header(text: str) -> str:
    """Detect section header from the first few lines of page text.
    Returns safe fallback 'General' if no reliable section is found.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:8]:  # Inspect header lines
        # Skip generic top headers or footers
        if "ENTERPRISE CORP" in line.upper() or "CONFIDENTIAL" in line.upper():
            continue
        if re.match(r"^Document:.*Page \d+", line, re.IGNORECASE):
            continue

        for pattern in SECTION_PATTERNS:
            if pattern.match(line):
                cleaned = line.strip(" -–—:#")
                if len(cleaned) <= 80:
                    return cleaned

    return "General"


def infer_document_type(filename: str) -> str:
    """Infer document type based on filename."""
    lower_name = filename.lower()
    if "contract" in lower_name or "agreement" in lower_name or "msa" in lower_name:
        return "contract"
    return "policy"


def load_pdf_file(file_path: Path) -> List[Document]:
    """Load a single PDF file using PyMuPDF and extract page-level text with metadata."""
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    documents: List[Document] = []
    filename = file_path.name
    doc_type = infer_document_type(filename)

    try:
        pdf_doc = fitz.open(file_path)
    except Exception as e:
        logger.error(f"Failed to open PDF {file_path}: {e}")
        raise ValueError(f"Invalid or corrupted PDF file: {filename}") from e

    try:
        for page_idx in range(len(pdf_doc)):
            page = pdf_doc[page_idx]
            page_text = page.get_text("text").strip()

            if not page_text:
                logger.warning(f"Empty text on page {page_idx + 1} of {filename}")
                continue

            section = detect_section_header(page_text)
            page_number = page_idx + 1

            metadata = {
                "source": filename,
                "page": page_number,
                "section": section,
                "document_type": doc_type,
            }

            documents.append(Document(page_content=page_text, metadata=metadata))

    finally:
        pdf_doc.close()

    logger.info(f"Loaded {len(documents)} pages from {filename}")
    return documents


def load_all_documents(directory: Path) -> List[Document]:
    """Scan directory for all PDF files and extract page-level Documents."""
    if not directory.exists() or not directory.is_dir():
        logger.warning(f"Document directory does not exist: {directory}")
        return []

    pdf_files = sorted(list(directory.glob("*.pdf")))
    all_pages: List[Document] = []

    for pdf_file in pdf_files:
        try:
            pages = load_pdf_file(pdf_file)
            all_pages.extend(pages)
        except Exception as e:
            logger.error(f"Error loading {pdf_file.name}: {e}")

    logger.info(f"Loaded a total of {len(all_pages)} pages from {len(pdf_files)} PDFs in {directory}")
    return all_pages
