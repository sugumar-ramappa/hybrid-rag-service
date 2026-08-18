# Hybrid RAG Service

A Smart Document Q&A System that combines RAG (Retrieval Augmented Generation), MCP (Model Context Protocol), and Google ADK Agent.

Put documents in a folder → system indexes them → ask questions in plain English → get accurate answers with source citations.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                          │
├──────────┬───────────┬──────────────┬──────────────┬────────────┤
│Terminal  │ ADK Agent │  Streamlit   │   Claude     │  FastAPI   │
│  CLI     │  CLI      │  (browser)   │  Desktop     │  (REST)    │
│query.py  │run_agent  │  app.py      │  (MCP)       │  api.py    │
└────┬─────┴─────┬─────┴──────┬───────┴──────┬───────┴─────┬──────┘
     │           │            │              │             │
     │           ▼            ▼              ▼             │
     │    ┌────────────┐ ┌────────────┐ ┌──────────┐     │
     │    │ ADK Agent  │ │ ADK Agent  │ │ MCP      │     │
     │    │ (Gemini    │ │ (Gemini    │ │ Server   │     │
     │    │  decides)  │ │  decides)  │ │server.py │     │
     │    └─────┬──────┘ └─────┬──────┘ └────┬─────┘     │
     │          │              │              │           │
     └──────────┴──────────────┴──────────────┴───────────┘
                               │
                        ┌──────┴──────┐
                        │ RAG Pipeline │
                        │ pipeline.py  │
                        └──────┬──────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
       ┌────────────┐  ┌────────────┐   ┌──────────┐
       │ ChromaDB   │  │  Gemini    │   │ Gemini   │
       │ (search)   │  │ (embedding)│   │ (answer) │
       └────────────┘  └────────────┘   └──────────┘
```

## 5 Ways to Use This System

### 1. Terminal CLI (direct pipeline)
```bash
python -m scripts.query "What is Python?"
```
| Flow | `scripts/query.py` → `RAGPipeline.query()` → ChromaDB → Gemini |
|------|------------------------------------------------------------------|
| AI used | Gemini (RAG only — generates answer from chunks) |
| Agent? | No — no decision-making, always searches |
| Memory? | No — each question is independent |
| Best for | Quick testing |

### 2. ADK Agent CLI (interactive chat)
```bash
python -m scripts.run_agent
```
| Flow | `scripts/run_agent.py` → `ADK Agent (Gemini)` → decides tool → `search_knowledge_base()` → `RAGPipeline.query()` |
|------|------|
| AI used | Gemini x2 (agent decides + pipeline answers) |
| Agent? | Yes — Gemini decides which tool to call |
| Memory? | Yes — remembers previous questions in session |
| Best for | Multi-turn conversations |

### 3. Streamlit (browser chat UI)
```bash
streamlit run app.py
```
| Flow | `app.py` → `run_agent_query()` → ADK Agent → `RAGPipeline.query()` |
|------|------|
| AI used | Gemini x2 (same as ADK Agent) |
| Agent? | Yes |
| Memory? | Yes — UI memory (chat bubbles) + agent memory (context) |
| Best for | Demos, internal tools |

### 4. Claude Desktop (MCP)
Configure in `~/Library/Application Support/Claude/claude_desktop_config.json`
| Flow | Claude → MCP protocol (stdio) → `server.py` → `search_documents()` → `RAGPipeline.query()` |
|------|------|
| AI used | Claude (decides tool) + Gemini (generates answer from chunks) |
| Agent? | Claude IS the agent — your server just provides tools |
| Memory? | Claude manages its own memory |
| Best for | Developers who already use Claude |

### 5. FastAPI (REST API)
```bash
uvicorn api:app --reload
# Open http://localhost:8000/docs
```
| Flow | HTTP POST `/query` → `api.py` → `RAGPipeline.query()` |
|------|------|
| AI used | Gemini (RAG only — no agent) |
| Agent? | No — direct pipeline call |
| Memory? | No — stateless REST |
| Best for | Mobile apps, Slack bots, other services calling your system |

## Tech Stack & Trade-offs

| Technology | What it does | Why we chose it | Trade-off |
|-----------|-------------|----------------|-----------|
| **Python 3.11** | Language | AI/ML ecosystem, library support | Slower than Java/Go at runtime |
| **Google Gemini** | LLM (answers + embeddings) | Free tier (15 req/min), good quality | Rate limits, vendor lock-in |
| **ChromaDB** | Vector database | Free, local, zero setup, pip install | Not for billions of vectors (use Vertex AI Search) |
| **MCP SDK 2.0** | AI-to-code protocol | Standard for Claude/VS Code integration | Only useful if user has Claude Desktop |
| **Google ADK** | Agent framework | Native Gemini support, sessions built-in | Gemini-only (unlike LangChain which supports any LLM) |
| **Streamlit** | Web UI | ~35 lines for a full chat interface | Not for millions of users, limited customization |
| **FastAPI** | REST API | Auto Swagger docs, async, fast | Needs separate server process |
| **Docker** | Containerization | Reproducible builds, deploy anywhere | Adds build step, image size |
| **Pydantic** | Data validation | Auto-validates API input, type-safe | Learning curve for complex nested models |

### Alternative choices (and when to switch)

| Current | Alternative | Switch when |
|---------|-------------|-------------|
| ChromaDB (local) | Pinecone / Vertex AI Search | Need cloud scale, >1M vectors |
| Gemini (free) | GPT-4 / Claude API | Need better reasoning or different pricing |
| Google ADK | LangChain / CrewAI | Need multi-provider LLM support |
| Streamlit | React + Next.js | Need custom design, millions of users |
| Google AI Studio | Vertex AI | Need production SLA, VPC isolation |
| InMemorySession | PostgreSQL / Redis | Need persistence across restarts |
| `time.sleep()` retry | Tenacity library | Need complex retry policies |

## Quick Start

### 1. Clone and setup
```bash
git clone https://github.com/sugumar-ramappa/hybrid-rag-service.git
cd hybrid-rag-service
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env and add your Google AI Studio API key
# Get one free at https://aistudio.google.com/apikey
```

### 3. Ingest documents
```bash
python -m scripts.ingest
```

### 4. Run (pick any interface)
```bash
python -m scripts.query "What is Python?"           # Terminal
python -m scripts.run_agent                          # Agent chat
streamlit run app.py                                 # Browser UI
uvicorn api:app --reload                             # REST API
```

### 5. Docker
```bash
docker build -t hybrid-rag-service .
docker run --env-file .env \
  -e DOCUMENTS_DIR=./documents \
  -e CHROMA_PERSIST_DIR=./data/chroma_db \
  hybrid-rag-service sh -c "python -m scripts.ingest && python -m scripts.query 'What is Python?'"
```

## Project Structure

```
├── app.py                        # Streamlit chat UI
├── api.py                        # FastAPI REST API
├── Dockerfile                    # Container build
├── .dockerignore                 # Exclude .env, .venv, data/
├── src/
│   ├── config.py                 # Configuration (reads .env)
│   ├── logging_config.py         # JSON/text logging
│   ├── rag/
│   │   ├── document_loader.py    # Loads .txt/.md/.pdf
│   │   ├── text_splitter.py      # Splits docs into chunks
│   │   ├── embeddings.py         # Text → vectors (Gemini API)
│   │   ├── vector_store.py       # ChromaDB storage + search
│   │   └── pipeline.py           # Orchestrates RAG flow
│   ├── mcp_server/
│   │   └── server.py             # MCP tools for Claude
│   └── agent/
│       └── rag_agent.py          # Google ADK agent
├── scripts/
│   ├── ingest.py                 # CLI: ingest documents
│   ├── query.py                  # CLI: ask questions
│   └── run_agent.py              # CLI: agent chat
└── tests/                        # pytest unit tests
```

## Tests

```bash
pytest tests/ -v
```

## Security

- API keys in `.env` (gitignored, never in code)
- Input validation: query length limits, `top_k` clamping
- Path traversal protection in document loader
- Non-root Docker user
- Structured logging for audit trail
