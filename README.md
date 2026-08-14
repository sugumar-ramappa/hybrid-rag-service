# RAG MCP Gemini

A Smart Document Q&A System that combines RAG (Retrieval Augmented Generation), MCP (Model Context Protocol), and Google ADK Agent.

Put documents in a folder → system indexes them → ask questions in plain English → get accurate answers with source citations.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│ Layer 3: Google ADK Agent                           │
│   Gemini-powered assistant that decides what to do  │
├─────────────────────────────────────────────────────┤
│ Layer 2: MCP Server                                 │
│   Exposes RAG as tools for Claude Desktop / VS Code │
├─────────────────────────────────────────────────────┤
│ Layer 1: RAG Pipeline                               │
│   Load → Chunk → Embed → Store → Search → Answer   │
│   ChromaDB + Google Gemini                          │
└─────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| LLM | Google Gemini (via AI Studio) |
| Embeddings | gemini-embedding-001 |
| Vector DB | ChromaDB |
| MCP | MCP SDK 2.0 |
| Agent | Google ADK |
| Container | Docker |

## Quick Start

### 1. Clone and setup
```bash
git clone https://github.com/sugumar-ramappa/rag-mcp-gemini.git
cd rag-mcp-gemini
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
# Add .txt, .md, or .pdf files to the documents/ folder, then:
python -m scripts.ingest
```

### 4. Query (3 ways)

**Terminal CLI:**
```bash
python -m scripts.query "What is Python?"
```

**ADK Agent (interactive chat):**
```bash
python -m scripts.run_agent
```

**Claude Desktop (MCP):**
Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "rag-mcp-server": {
      "command": "/path/to/rag-mcp-gemini/.venv/bin/python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "/path/to/rag-mcp-gemini",
      "env": {"PYTHONPATH": "/path/to/rag-mcp-gemini"}
    }
  }
}
```

### 5. Docker
```bash
docker build -t rag-mcp-gemini .
docker run --env-file .env \
  -e DOCUMENTS_DIR=./documents \
  -e CHROMA_PERSIST_DIR=./data/chroma_db \
  rag-mcp-gemini sh -c "python -m scripts.ingest && python -m scripts.query 'What is Python?'"
```

## Project Structure

```
src/
├── config.py                 # Configuration (reads .env)
├── logging_config.py         # JSON/text logging setup
├── rag/
│   ├── document_loader.py    # Loads .txt/.md/.pdf files
│   ├── text_splitter.py      # Splits docs into chunks with overlap
│   ├── embeddings.py         # Text → vectors via Gemini API
│   ├── vector_store.py       # ChromaDB storage and search
│   └── pipeline.py           # Orchestrates the RAG flow
├── mcp_server/
│   └── server.py             # MCP tools for Claude Desktop
└── agent/
    └── rag_agent.py           # Google ADK agent with Gemini
```

## How It Works

**Ingestion:**
```
Documents → Load → Split into chunks → Generate embeddings → Store in ChromaDB
```

**Query:**
```
Question → Embed → Search ChromaDB → Build context → Gemini generates answer
```

## Tests

```bash
pytest tests/ -v
```

## Key Features

- **RAG Pipeline**: Document ingestion with chunking, embedding, and vector search
- **MCP Server**: Connect to Claude Desktop for AI-powered document Q&A
- **ADK Agent**: Self-contained Gemini-powered assistant with session memory
- **Error Handling**: Retry with exponential backoff for API rate limits
- **Security**: Input validation, path traversal protection, secrets management
- **Docker**: Containerized with non-root user
- **Structured Logging**: JSON output for production, human-readable for dev
