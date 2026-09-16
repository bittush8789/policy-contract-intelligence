# Enterprise RAG Assistant
### Policy & Contract Intelligence

> Ask questions about corporate policy and contract documents and get **cited, accurate, business-friendly answers** — powered by Hybrid RAG, AI Guardrails, and LLM Observability.

---

## Features

| Feature | Details |
|---|---|
| **Hybrid Search** | Pinecone semantic + BM25 keyword search fused via RRF |
| **AI Reranking** | BGE CrossEncoder reranks top candidates to best 5 context chunks |
| **Document Citations** | Every answer includes document, page, and section reference |
| **AI Guardrails** | Blocks prompt injection, PII leakage, and hallucinated outputs |
| **RAGAS Evaluation** | Built-in quality scorecard (Faithfulness, Relevancy, Precision, Recall) |
| **LangSmith Tracing** | Full LLM observability — traces every RAG run to a dashboard |
| **Zero-Upload Design** | Documents are pre-loaded; no file upload UI exists |
| **Docker Ready** | Multi-stage Docker image + Compose with volume persistence |
| **CI/CD Pipeline** | GitHub Actions → Lint → Tests → Trivy Scan → AWS ECR |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq — `openai/gpt-oss-120b` |
| **Embeddings** | `BAAI/bge-small-en-v1.5` (384-dim) |
| **Reranker** | `BAAI/bge-reranker-v2-m3` (CrossEncoder) |
| **Vector DB** | Pinecone |
| **Keyword Search** | `rank-bm25` |
| **Backend** | FastAPI + Uvicorn + Pydantic v2 |
| **Frontend** | HTML5 + CSS3 + Vanilla JS |
| **Observability** | LangSmith |
| **Evaluation** | RAGAS (custom offline implementation) |
| **Containerization** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions + AWS ECR |

---

## Project Structure

```
enterprise-rag-assistant/
├── backend/
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # All environment settings (Pydantic)
│   ├── api/
│   │   ├── chat.py                # POST /api/chat
│   │   └── evaluation.py          # GET /api/evaluation/benchmark
│   ├── chains/
│   │   ├── rag_chain.py           # End-to-end RAG orchestration
│   │   └── prompts.py             # System & human prompt templates
│   ├── retrieval/
│   │   ├── pinecone_retriever.py  # Semantic vector search
│   │   ├── bm25_retriever.py      # Keyword BM25 search
│   │   ├── hybrid_retriever.py    # RRF fusion of both
│   │   └── reranker.py            # BGE CrossEncoder reranking
│   ├── ingestion/
│   │   ├── loader.py              # PyMuPDF PDF loader
│   │   ├── splitter.py            # Recursive chunker
│   │   ├── embeddings.py          # BGE embedding generator
│   │   └── ingest.py              # Full ingestion pipeline runner
│   ├── guardrails/
│   │   ├── input_guardrails.py    # Injection + PII detection
│   │   ├── output_guardrails.py   # Leakage + hallucination filter
│   │   └── service.py             # Guardrails pipeline coordinator
│   ├── evaluation/
│   │   ├── dataset.py             # 7-query golden dataset
│   │   ├── evaluator.py           # RAGAS metrics engine
│   │   └── benchmark.py           # CLI benchmark runner
│   ├── observability/
│   │   └── langsmith.py           # LangSmith tracing setup
│   └── citations/
│       └── formatter.py           # Citation extractor
├── frontend/
│   ├── index.html                 # Query UI + RAGAS scorecard modal
│   ├── css/style.css              # Light theme + responsive layout
│   └── js/app.js                  # Vanilla JS client
├── data/
│   ├── documents/                 # Pre-loaded PDFs (7 enterprise docs)
│   ├── evaluation/                # Golden dataset + benchmark report
│   └── processed_chunks.json      # BM25 chunk cache
├── docs/
│   └── system_design.md           # System architecture diagrams
├── tests/                         # 34 unit tests (100% offline)
├── scripts/
│   └── generate_sample_docs.py    # Generate sample enterprise PDFs
├── .env.example                   # Environment variable template
├── requirements.txt
├── pyproject.toml                 # Ruff linter config
├── Dockerfile                     # Multi-stage Docker build
├── docker-compose.yml
└── .github/workflows/ci-cd.yml   # GitHub Actions pipeline
```

---

## Quick Start

### 1. Setup Environment
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
cp .env.example .env
```
Edit `.env` and fill in:
```ini
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=enterprise-rag
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-120b
```

### 3. Generate & Ingest Documents
```bash
# Generate sample enterprise policy PDFs
python scripts/generate_sample_docs.py

# Chunk, embed and index into Pinecone
python -m backend.ingestion.ingest
```

### 4. Run the Application
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Open **http://localhost:8000**

---

## Sample Questions

| Document | Example Question |
|---|---|
| Leave Policy | What is the annual leave entitlement for full-time employees? |
| Leave Policy | How many sick days do employees get per year? |
| Remote Work Policy | How many days per week can employees work from home? |
| Remote Work Policy | Does remote work require manager approval? |
| Info Security Policy | What is the minimum password length required? |
| Info Security Policy | Is Multi-Factor Authentication mandatory? |
| Procurement Policy | What is the approval threshold for large purchases? |
| Vendor Contract | What is the termination notice period? |
| Employee Handbook | What is the code of conduct policy? |
| Privacy Policy | How is employee personal data protected? |

> **Cross-document:** *"What security rules apply to employees working remotely?"*

---

## Docker

```bash
# 1. Configure keys
cp .env.example .env

# 2. Build and start
docker compose up --build

# 3. One-time document ingestion
docker compose exec app python -m backend.ingestion.ingest
```

Open **http://localhost:8000**

### Useful Docker Commands
```bash
docker compose up -d           # Run in background
docker compose logs -f         # Live logs
docker compose exec app bash   # Shell into container
docker compose down            # Stop (volumes preserved)
```

---

## LangSmith Observability

Enable full LLM tracing in `.env`:
```ini
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_key_here
LANGCHAIN_PROJECT=enterprise-rag-assistant
```

Get your key at: **https://smith.langchain.com → Settings → API Keys**

Check tracing status:
```
GET /api/observability
```

Every RAG pipeline run is traced — including prompt, retrieval, LLM call, token usage, latency, and guardrails result.

---

## RAGAS Evaluation

Run the benchmark CLI:
```bash
python -m backend.evaluation.benchmark
```

Or click the **Quality Scorecard** button in the web UI.

| Metric | Measures |
|---|---|
| **Faithfulness** | Are all claims grounded in retrieved context? |
| **Answer Relevancy** | Does the answer address the question? |
| **Context Precision** | Is the top-ranked chunk from the right document? |
| **Context Recall** | Does context cover all ground truth facts? |

---

## Running Tests
```bash
python -m pytest tests/ -v
```
**34 tests** — all run 100% offline, no real API keys needed.

---

## CI/CD Pipeline

```
Git Push → GitHub Actions
   ├── Lint (Ruff)
   ├── Unit Tests (34 offline)
   ├── RAG Tests (retrieval, reranker, guardrails)
   ├── Build Docker (multi-stage BuildX)
   ├── Trivy CVE Scan (→ GitHub Security tab)
   └── RAGAS Evaluation Tests
             ↓ (main/master only)
         Push to AWS ECR (:sha + :latest)
```

**Required GitHub Secrets:**
```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
ECR_REPOSITORY
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check + model config |
| `/api/chat` | POST | Ask a question → answer + citations |
| `/api/evaluation/benchmark` | GET | RAGAS scorecard |
| `/api/evaluation/run` | POST | Run fresh evaluation |
| `/api/observability` | GET | LangSmith tracing status |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `PINECONE_API_KEY` | ✅ | — | Pinecone API key |
| `GROQ_API_KEY` | ✅ | — | Groq LLM API key |
| `PINECONE_INDEX_NAME` | No | `enterprise-rag` | Pinecone index name |
| `GROQ_MODEL` | No | `openai/gpt-oss-120b` | LLM model ID |
| `ENABLE_GUARDRAILS` | No | `true` | AI safety guardrails toggle |
| `LANGCHAIN_TRACING_V2` | No | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | — | LangSmith API key |
| `LANGCHAIN_PROJECT` | No | `enterprise-rag-assistant` | LangSmith project name |
| `PORT` | No | `8000` | Server port |

> See [`.env.example`](.env.example) for the full list.

---

## System Design

Detailed architecture diagrams (sequence flows, guardrails, CI/CD, observability) are in [`docs/system_design.md`](docs/system_design.md).
