"""FastAPI application entrypoint for Enterprise RAG Assistant.
Provides health check, chat query endpoint, CORS configuration, and static frontend hosting.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from backend.api.chat import router as chat_router
from backend.api.evaluation import router as evaluation_router
from backend.config import settings

# Setup standard structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("enterprise_rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler for application startup and shutdown."""
    logger.info("=" * 60)
    logger.info("Initializing Enterprise RAG Assistant Application")
    logger.info(f"Model: {settings.GROQ_MODEL} (Provider: Groq)")
    logger.info(f"Embedding: {settings.EMBEDDING_MODEL}")
    logger.info(f"Reranker: {settings.RERANKER_MODEL}")
    logger.info(f"Documents Directory: {settings.DATA_DIR}")
    logger.info(f"Chunks Cache: {settings.CHUNKS_CACHE_FILE}")
    logger.info("=" * 60)
    yield
    logger.info("Enterprise RAG Assistant Application shut down gracefully.")


app = FastAPI(
    title="Enterprise RAG Assistant — Policy & Contract Intelligence",
    description="Enterprise-grade RAG service querying pre-loaded corporate policies and contracts.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Configuration for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(chat_router)
app.include_router(evaluation_router)


@app.get(
    "/api/health",
    tags=["System"],
    summary="Application Health Check",
    response_description="System health status and configuration",
)
async def health_check() -> JSONResponse:
    """Return health status and operational telemetry."""
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "enterprise-rag-assistant",
            "version": "1.0.0",
            "models": {
                "llm": settings.GROQ_MODEL,
                "embedding": settings.EMBEDDING_MODEL,
                "reranker": settings.RERANKER_MODEL,
            },
            "retrieval": {
                "vector_top_k": settings.VECTOR_TOP_K,
                "bm25_top_k": settings.BM25_TOP_K,
                "hybrid_top_k": settings.HYBRID_TOP_K,
                "rerank_top_k": settings.RERANK_TOP_K,
            },
        }
    )


# Mount frontend static files if directory exists
if settings.FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(settings.FRONTEND_DIR), html=True), name="frontend")
else:
    logger.warning(f"Frontend directory not found at {settings.FRONTEND_DIR}; static UI will not be served.")
