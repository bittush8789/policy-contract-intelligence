# ==============================================================================
# Enterprise RAG Assistant — Multi-Stage Dockerfile
# Stage 1: Builder  - installs all Python dependencies
# Stage 2: Runtime  - slim production image
# ==============================================================================

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System-level build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only the requirements file first (leverages Docker layer cache)
COPY requirements.txt .

# Install all dependencies into a dedicated prefix so the runtime stage can
# copy just the installed packages without the build tools.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Metadata labels
LABEL org.opencontainers.image.title="Enterprise RAG Assistant"
LABEL org.opencontainers.image.description="Policy & Contract Intelligence — RAG API + Frontend"
LABEL org.opencontainers.image.version="1.0.0"

# ── Environment Hardening ──────────────────────────────────────────────────────
# Prevent Python from buffering stdout/stderr (important for log visibility)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # ML framework guards (keeps startup lean — no TensorFlow/JAX side-effects)
    USE_TF=0 \
    TRANSFORMERS_NO_TF=1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TOKENIZERS_PARALLELISM=false \
    # Application defaults (overridable via docker-compose / -e flags)
    HOST=0.0.0.0 \
    PORT=8000

# Create a non-root user for security
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source code
COPY backend/   ./backend/
COPY frontend/  ./frontend/
COPY scripts/   ./scripts/
COPY data/      ./data/

# Transfer ownership to non-root user
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# Health-check: polls the /api/health endpoint every 30 s
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Default entrypoint — starts the Uvicorn ASGI server
CMD ["python", "-m", "uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
