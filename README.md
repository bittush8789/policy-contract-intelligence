# Enterprise RAG Assistant
**Policy & Contract Intelligence** — Ask questions about your corporate documents and get cited, accurate answers.

---

## What It Does
- Answers questions from pre-loaded policy and contract PDFs
- Returns answers with **document, page, and section citations**
- Uses **hybrid search** (Pinecone + BM25) with AI reranking
- Blocks harmful inputs and hallucinated outputs via **AI Guardrails**
- Includes a built-in **RAGAS quality scorecard**

---

## Tech Stack
| Layer | Technology |
|---|---|
| LLM | Groq (`openai/gpt-oss-120b`) |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Vector DB | Pinecone |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML / CSS / Vanilla JS |

---

## Quick Start

### 1. Install
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Fill in PINECONE_API_KEY and GROQ_API_KEY
```

### 3. Generate & Ingest Documents
```bash
python scripts/generate_sample_docs.py
python -m backend.ingestion.ingest
```

### 4. Run
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Open **http://localhost:8000**

---

## Docker
```bash
cp .env.example .env          # fill in API keys
docker compose up --build     # first run downloads ~500 MB models

# One-time ingestion
docker compose exec app python -m backend.ingestion.ingest
```

---

## CI/CD (GitHub Actions → AWS ECR)

| Stage | Job |
|---|---|
| Lint | Ruff code quality checks |
| Unit Tests | 34 offline pytest tests |
| RAG Tests | Retrieval, reranker, guardrails |
| Build Docker | Multi-stage BuildX |
| Trivy Scan | CVE scan → GitHub Security tab |
| Evaluation | RAGAS offline metrics |
| Push to ECR | `main`/`master` branch only |

**Required GitHub Secrets:**
```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
ECR_REPOSITORY
```

---

## Environment Variables
| Variable | Required | Default |
|---|---|---|
| `PINECONE_API_KEY` | Yes | — |
| `GROQ_API_KEY` | Yes | — |
| `PINECONE_INDEX_NAME` | No | `enterprise-rag` |
| `GROQ_MODEL` | No | `openai/gpt-oss-120b` |
| `ENABLE_GUARDRAILS` | No | `true` |

---

## Tests
```bash
python -m pytest tests/ -v
```

## API
| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check |
| `/api/chat` | POST | Ask a question |
| `/api/evaluation/benchmark` | GET | RAGAS scores |
