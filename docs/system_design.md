# System Design — Enterprise RAG Assistant
## Policy & Contract Intelligence

---

## 1. High-Level Architecture

```mermaid
flowchart TD
    subgraph Client["Client Layer"]
        U["👤 Business User\n(Browser)"]
        UI["Frontend\nHTML / CSS / JS"]
    end

    subgraph API["API Layer\n(FastAPI + Uvicorn)"]
        CHAT["/api/chat"]
        HEALTH["/api/health"]
        EVAL["/api/evaluation"]
        OBS["/api/observability"]
    end

    subgraph RAG["RAG Pipeline"]
        IG["Input Guardrails\n(Injection + PII Filter)"]
        HR["Hybrid Retriever\n(Pinecone + BM25 + RRF)"]
        RR["BGE Reranker\n(CrossEncoder)"]
        CTX["Context Builder"]
        PROMPT["Prompt Assembly\n(LangChain)"]
        LLM["Groq LLM\nopenai/gpt-oss-120b"]
        OG["Output Guardrails\n(Leakage + Hallucination)"]
        CIT["Citation Extractor"]
    end

    subgraph Storage["Storage & Index"]
        PINE[("Pinecone\nVector DB")]
        BM25[("BM25\nIn-Memory Index")]
        CACHE[("processed_chunks.json\nLocal Cache")]
    end

    subgraph Observability["Observability"]
        LS["LangSmith\nTracing Dashboard"]
    end

    subgraph Offline["Offline Ingestion Pipeline"]
        PDF["📄 PDF Documents\ndata/documents/"]
        LOAD["PyMuPDF Loader"]
        SPLIT["Recursive Chunker"]
        EMBED["BGE Embeddings\nbge-small-en-v1.5"]
        INDEX["Pinecone Indexer"]
    end

    U --> UI --> CHAT
    CHAT --> IG --> HR
    HR --> PINE
    HR --> BM25
    BM25 --> CACHE
    HR --> RR --> CTX --> PROMPT --> LLM --> OG --> CIT
    CIT --> CHAT
    CHAT --> UI

    PDF --> LOAD --> SPLIT --> EMBED --> INDEX --> PINE
    SPLIT --> CACHE

    LLM --> LS
    PROMPT --> LS
    HR --> LS
```

---

## 2. Offline Ingestion Pipeline

```mermaid
sequenceDiagram
    participant PDF as 📄 PDF Files
    participant Loader as PyMuPDF Loader
    participant Splitter as Recursive Chunker
    participant Embedder as BGE Embeddings
    participant Pinecone as Pinecone Vector DB
    participant Cache as processed_chunks.json

    PDF->>Loader: Load pages + extract text & metadata
    Loader->>Splitter: Raw text + (source, page, section) metadata
    Splitter->>Splitter: Chunk (size=800, overlap=150)
    Splitter->>Cache: Persist chunks for BM25 indexing
    Splitter->>Embedder: Chunk texts
    Embedder->>Embedder: Generate 384-dim dense vectors
    Embedder->>Pinecone: Upsert (chunk_id, vector, metadata)
    Note over Cache: BM25 loads from here at runtime
    Note over Pinecone: Indexed & searchable immediately
```

---

## 3. Online Query Pipeline (Per Request)

```mermaid
sequenceDiagram
    participant User as 👤 User
    participant API as FastAPI /api/chat
    participant IG as Input Guardrails
    participant VEC as Pinecone Retriever
    participant BM25 as BM25 Retriever
    participant RRF as RRF Fusion
    participant RR as BGE Reranker
    participant LLM as Groq LLM
    participant OG as Output Guardrails
    participant LS as LangSmith

    User->>API: POST {"question": "..."}
    API->>IG: Validate & sanitize input
    IG-->>API: BLOCKED (injection/PII) or PASS + sanitized query

    API->>VEC: Semantic search (top_k=20)
    API->>BM25: Keyword search (top_k=20)
    VEC-->>RRF: 20 vector candidates
    BM25-->>RRF: 20 keyword candidates
    RRF->>RRF: Reciprocal Rank Fusion → top 20 merged
    RRF->>RR: 20 hybrid candidates
    RR->>RR: CrossEncoder rerank → top 5 chunks
    RR->>LLM: Formatted context + question prompt
    LLM-->>OG: Raw answer
    OG->>OG: Check for leakage / hallucination
    OG-->>API: Sanitized answer + citations
    API-->>User: {"answer", "citations", "retrieval", "guardrails"}

    Note over LS: Full trace recorded automatically
    LLM-->LS: Token usage, latency, prompt/response
```

---

## 4. Hybrid Retrieval & RRF Fusion

```mermaid
flowchart LR
    Q["User Query"]

    Q --> VEC["Pinecone\nSemantic Search\nBGE Embeddings\ntop_k = 20"]
    Q --> BM25["BM25\nKeyword Search\nTF-IDF Scoring\ntop_k = 20"]

    VEC --> RRF["Reciprocal Rank Fusion\nscore = Σ 1/(k + rank_i)\nk = 60"]
    BM25 --> RRF

    RRF --> MERGE["Merged & Deduplicated\nTop 20 candidates"]
    MERGE --> RERANK["BGE CrossEncoder\nbge-reranker-v2-m3\nCross-attention scoring"]
    RERANK --> TOP5["Top 5 Context Chunks\n→ LLM Prompt"]
```

---

## 5. AI Guardrails Architecture

```mermaid
flowchart TD
    subgraph Input Guardrails
        I1["Prompt Injection Detector\n(adversarial pattern matching)"]
        I2["PII Redactor\n(SSN, email, phone, credit card)"]
        I3["Harmful Intent Blocker\n(violence, illegal activity)"]
        I4["Query Length Validator\n(max 2000 chars)"]
    end

    subgraph Output Guardrails
        O1["System Prompt Leakage Detector\n(prevents instruction disclosure)"]
        O2["API Key / Secret Leakage Detector\n(regex-based credential scan)"]
        O3["Safe Refusal Enforcer\n(no context = safe refusal)"]
    end

    Q["Raw User Input"] --> I1 --> I2 --> I3 --> I4
    I4 -->|"BLOCKED"| BLOCK["Return security\nblock message"]
    I4 -->|"ALLOWED"| RAG["RAG Pipeline"]
    RAG --> LLM["LLM Raw Answer"]
    LLM --> O1 --> O2 --> O3
    O3 -->|"CLEAN"| FINAL["Final Answer\n+ Citations"]
    O3 -->|"FLAGGED"| SANITIZE["Sanitized / Redacted\nAnswer"]
```

---

## 6. RAGAS Evaluation Framework

```mermaid
flowchart TD
    GD["Golden Dataset\n7 Enterprise Queries\nwith ground truth answers"]
    GD --> E1["Faithfulness\nAre all claims grounded\nin retrieved context?"]
    GD --> E2["Answer Relevancy\nDoes the answer address\nthe question?"]
    GD --> E3["Context Precision\nIs the top-ranked chunk\nfrom the right document?"]
    GD --> E4["Context Recall\nDoes retrieved context cover\nthe ground truth facts?"]

    E1 & E2 & E3 & E4 --> SCORE["RAG Score\nWeighted Average\n0.0 → 1.0"]
    SCORE --> REPORT["benchmark_report.json\n+ UI Scorecard Modal"]
```

---

## 7. CI/CD Pipeline (GitHub Actions → AWS ECR)

```mermaid
flowchart LR
    GIT["Git Push\nmain / master"] --> GHA["GitHub Actions"]

    GHA --> L["Lint\nRuff E/W/F/I"]
    L --> UT["Unit Tests\n34 offline pytest"]
    L --> RT["RAG Tests\nRetrieval + Reranker\n+ Guardrails"]

    UT & RT --> BD["Build Docker\nMulti-stage BuildX\n+ GHA layer cache"]

    BD --> TS["Trivy Scan\nCVE CRITICAL/HIGH\n→ SARIF → GitHub Security"]
    BD --> ET["Evaluation Tests\nRAGAS offline metrics"]

    TS & ET --> ECR["Push to AWS ECR\n:short-sha + :latest\nmain/master only"]
```

---

## 8. LangSmith Observability Flow

```mermaid
flowchart TD
    APP["Application Startup\nmain.py lifespan"]
    APP --> ENV["Set Environment Variables\nLANGCHAIN_TRACING_V2=true\nLANGCHAIN_API_KEY\nLANGCHAIN_PROJECT"]

    ENV --> LC["LangChain SDK\nAuto-instruments all calls"]

    LC --> T1["Trace: Prompt Assembly"]
    LC --> T2["Trace: Groq LLM Call\n(tokens, latency, cost)"]
    LC --> T3["Trace: Retriever Calls"]
    LC --> T4["Trace: Full RAG Chain\n@traceable decorator"]

    T1 & T2 & T3 & T4 --> DASH["LangSmith Dashboard\nsmith.langchain.com\n/api/observability endpoint"]
```

---

## 9. Data Flow Summary

| Stage | Input | Process | Output |
|---|---|---|---|
| **Ingestion** | PDF files | PyMuPDF → Chunk → Embed | Pinecone index + BM25 cache |
| **Input Guard** | Raw query | Injection / PII detection | Sanitized query or block |
| **Retrieval** | Query | Pinecone + BM25 + RRF | 20 merged candidates |
| **Reranking** | 20 candidates | BGE CrossEncoder | Top 5 context chunks |
| **Generation** | Context + query | Groq LLM | Raw answer |
| **Output Guard** | Raw answer | Leakage / hallucination check | Clean final answer |
| **Citation** | Top 5 chunks | Metadata extraction | Document + page + section |
| **Evaluation** | 7 golden queries | RAGAS metrics | Score 0.0–1.0 |
| **Observability** | All LangChain calls | LangSmith tracing | Dashboard traces |

---

## 10. Technology Decisions

| Decision | Choice | Reason |
|---|---|---|
| Vector DB | Pinecone | Managed, scalable, low-latency ANN search |
| Keyword Search | BM25 | Handles exact terms, acronyms, IDs — complements semantic |
| Fusion | RRF | Parameter-free, robust rank merging — no score normalization needed |
| Reranker | BGE CrossEncoder | Token-level cross-attention far outperforms bi-encoder re-scoring |
| LLM | Groq / openai/gpt-oss-120b | Low-latency inference, deterministic (temp=0) for grounded answers |
| Embeddings | BGE-small-en-v1.5 | 384-dim, fast, strong English semantic accuracy |
| Observability | LangSmith | Native LangChain integration, zero-instrumentation auto-tracing |
| Evaluation | RAGAS | Industry-standard RAG metrics, fully offline capable |
| Container | Docker multi-stage | Lean runtime image, non-root user, secrets never baked into layers |
| CI/CD | GitHub Actions + ECR | Native GH integration, Trivy CVE scanning, immutable SHA tags |
