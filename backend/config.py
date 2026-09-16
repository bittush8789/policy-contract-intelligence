"""Configuration module for Enterprise RAG Assistant.
Loads environment variables safely using Pydantic Settings and enforces
environment guards for machine learning runtimes.
"""

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Guard machine learning backends on Windows / multi-framework setups
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Centralized application settings."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Pinecone
    PINECONE_API_KEY: str = Field(default="", description="Pinecone API key")
    PINECONE_INDEX_NAME: str = Field(default="enterprise-rag", description="Pinecone index name")
    PINECONE_NAMESPACE: str = Field(default="policies-contracts", description="Pinecone namespace")

    # LLM (Groq)
    GROQ_API_KEY: str = Field(default="", description="Groq API key")
    GROQ_MODEL: str = Field(default="openai/gpt-oss-120b", description="Groq model ID")

    # Embedding & Reranker Models
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-small-en-v1.5", description="BGE embedding model")
    RERANKER_MODEL: str = Field(default="BAAI/bge-reranker-v2-m3", description="BGE CrossEncoder reranker")

    # Ingestion Parameters
    CHUNK_SIZE: int = Field(default=800, description="Recursive character text splitter chunk size")
    CHUNK_OVERLAP: int = Field(default=150, description="Recursive character text splitter chunk overlap")

    # Retrieval Limits
    VECTOR_TOP_K: int = Field(default=20, description="Pinecone semantic search top candidates")
    BM25_TOP_K: int = Field(default=20, description="BM25 keyword search top candidates")
    HYBRID_TOP_K: int = Field(default=20, description="Merged hybrid candidates before reranking")
    RERANK_TOP_K: int = Field(default=5, description="Final top context chunks after reranking")

    # File Paths
    DATA_DIR: Path = Field(default=BASE_DIR / "data" / "documents", description="Pre-loaded documents directory")
    CHUNKS_CACHE_FILE: Path = Field(default=BASE_DIR / "data" / "processed_chunks.json", description="Ingested chunks cache")
    FRONTEND_DIR: Path = Field(default=BASE_DIR / "frontend", description="Frontend static files directory")

    # Server
    HOST: str = Field(default="0.0.0.0", description="API server host")
    PORT: int = Field(default=8000, description="API server port")

    # AI Guardrails
    ENABLE_GUARDRAILS: bool = Field(default=True, description="Master toggle for AI Guardrails subsystem")
    ENABLE_INPUT_GUARDRAILS: bool = Field(default=True, description="Toggle for pre-retrieval input guardrails")
    ENABLE_OUTPUT_GUARDRAILS: bool = Field(default=True, description="Toggle for post-generation output guardrails")
    ENABLE_PII_REDACTION: bool = Field(default=True, description="Toggle for PII detection and masking")
    BLOCK_PROMPT_INJECTION: bool = Field(default=True, description="Block adversarial prompt injection attempts")

    # LangSmith Observability
    LANGCHAIN_TRACING_V2: bool = Field(default=False, description="Enable LangSmith tracing (set True + API key to activate)")
    LANGCHAIN_API_KEY: str = Field(default="", description="LangSmith API key from smith.langchain.com")
    LANGCHAIN_PROJECT: str = Field(default="enterprise-rag-assistant", description="LangSmith project name for trace grouping")
    LANGSMITH_ENDPOINT: str = Field(default="https://api.smith.langchain.com", description="LangSmith API endpoint")


# Reusable singleton instance
settings = Settings()
