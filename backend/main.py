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
from backend.observability.langsmith import setup_langsmith, get_run_url, is_tracing_enabled

# Setup standard structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("enterprise_rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler for application startup and shutdown."""
    # ── LangSmith must be initialized FIRST before any LangChain component ──
    setup_langsmith(
        api_key=settings.LANGCHAIN_API_KEY,
        project=settings.LANGCHAIN_PROJECT,
        endpoint=settings.LANGSMITH_ENDPOINT,
        enabled=settings.LANGCHAIN_TRACING_V2,
    )

    logger.info("=" * 60)
    logger.info("Initializing Enterprise RAG Assistant Application")
    logger.info(f"Model: {settings.GROQ_MODEL} (Provider: Groq)")
    logger.info(f"Embedding: {settings.EMBEDDING_MODEL}")
    logger.info(f"Reranker: {settings.RERANKER_MODEL}")
    logger.info(f"Documents Directory: {settings.DATA_DIR}")
    logger.info(f"Chunks Cache: {settings.CHUNKS_CACHE_FILE}")
    logger.info(f"LangSmith Tracing: {'ENABLED' if is_tracing_enabled() else 'DISABLED'}")
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


@app.get(
    "/api/observability",
    tags=["System"],
    summary="LangSmith Observability Status",
    response_description="LangSmith tracing configuration and dashboard URL",
)
async def observability_status() -> JSONResponse:
    """Return the current LangSmith tracing status and project dashboard URL."""
    tracing_on = is_tracing_enabled()
    return JSONResponse(
        content={
            "langsmith_tracing": tracing_on,
            "project": settings.LANGCHAIN_PROJECT if tracing_on else None,
            "dashboard_url": get_run_url(),
            "endpoint": settings.LANGSMITH_ENDPOINT if tracing_on else None,
            "note": (
                "Tracing active — all RAG pipeline runs are being recorded in LangSmith."
                if tracing_on
                else "Tracing disabled. Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY in .env to enable."
            ),
        }
    )


# Mount frontend static files if directory exists
if settings.FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(settings.FRONTEND_DIR), html=True), name="frontend")
else:
    logger.warning(f"Frontend directory not found at {settings.FRONTEND_DIR}; static UI will not be served.")
