# RAG MCP Gemini - Learning Notes
> Java developer learning Python + RAG + MCP + Google ADK
> Started: Aug 11, 2026 | Schedule: 2-3 hrs/day, 3 weeks

---

## Project Overview
**What we're building:** A "Smart Document Q&A System"
- Put documents (PDF/TXT) into a folder
- System reads, understands, and indexes them
- Ask questions in plain English → get accurate answers with source citations

**Architecture (3 layers):**
```
Layer 3: Google ADK Agent     → "Smart assistant that DECIDES what to do"    (Week 2)
Layer 2: MCP Server           → "API that exposes search to AI tools"       (Week 2)
Layer 1: RAG Pipeline         → "Load → chunk → embed → store → search"    (Week 1)
```

---

## 3-Week Plan (2-3 hrs/day)

### Week 1: Foundation (Python + RAG Core)
| Day | Topic | Status |
|-----|-------|--------|
| 1 | Setup + config.py + install deps + tests passing | ✅ Done |
| 2 | document_loader.py | ✅ Done |
| 3 | text_splitter.py + tests | ✅ Done |
| 4 | embeddings.py (Vertex AI) | ✅ Done |
| 5 | vector_store.py (ChromaDB) | ✅ Done |
| 6 | pipeline.py (end-to-end RAG) | ✅ Done |
| 7 | Write & run all tests | ✅ Done |

### Week 2: MCP Server + Google ADK Agent
| Day | Topic | Status |
|-----|-------|--------|
| 8 | Learn MCP protocol concepts | ✅ Done |
| 9 | Build MCP server (server.py) | ✅ Done |
| 10 | Test MCP server with Claude Desktop | ✅ Done |
| 11 | Google ADK Agent basics | ✅ Done |
| 12 | ADK Agent + RAG integration | ✅ Done |
| 13 | ADK Agent + MCP bridge | ✅ Done |
| 14 | Error handling + logging | ✅ Done |

### Week 3: Production + Deploy
| Day | Topic | Status |
|-----|-------|--------|
| 15 | Structured logging + monitoring | ✅ Done |
| 16 | Input validation + security | ✅ Done |
| 17 | Docker containerization | ✅ Done |
| 18 | Cloud Run deployment (switch to Vertex AI) | ⏭ Skipped (needs GCP account) |
| 19 | CI/CD with GitHub Actions | ⏭ Skipped (no deploy target) |
| 20 | Load testing + optimization | ⏭ Skipped |
| 21 | Documentation + final review | ✅ Done |

### Bonus (beyond 3-week plan)
| Topic | Status |
|-------|--------|
| Streamlit chat UI (app.py) | ✅ Done |
| FastAPI REST API (api.py) | ✅ Done |
| GitHub repo setup (personal account) | ✅ Done |

---

## Setup Status
- [x] Python 3.11.6 installed
- [x] pip 25.2 installed
- [x] Google AI Studio API key created
- [x] .env file configured with API key
- [x] Virtual environment created (.venv)
- [x] Dependencies installed (pip install -r requirements.txt)
- [x] All 17 tests passing

**Setup commands (run when ready):**
```bash
cd ~/Learning\ Project/rag-mcp-gemini
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

---

## Google AI Studio vs Vertex AI
- **Weeks 1-2:** Google AI Studio (free API key, no credit card)
- **Week 3:** Switch to Vertex AI for deployment (needs Google Cloud account, $300 free credits)
- **Change is small:** swap authentication method + a few imports, RAG logic stays same

---

## Project Files Explained

### Config & Setup
| File | Java Equivalent | Purpose |
|------|----------------|---------|
| pyproject.toml | pom.xml | Project metadata + dependencies |
| requirements.txt | dependency list | Flat list of packages to install |
| .env | application.properties | API key + settings (gitignored) |

### Layer 1: RAG Pipeline (Week 1) — Read in this order
| # | File | Java Equivalent | Purpose |
|---|------|----------------|---------|
| 1 | src/config.py | @Configuration class | Reads .env, creates config objects |
| 2 | src/rag/document_loader.py | FileReaderService | Loads .txt/.md/.pdf files |
| 3 | src/rag/text_splitter.py | TextChunkerService | Splits docs into overlapping chunks |
| 4 | src/rag/embeddings.py | EmbeddingClient | Calls Google API: text → numbers |
| 5 | src/rag/vector_store.py | VectorRepository (JPA) | Stores/searches vectors in ChromaDB |
| 6 | src/rag/pipeline.py | RAGService (orchestrator) | Ties 2-5 together |

### Layer 2 & 3 (Week 2 — ignore for now)
| File | Purpose |
|------|---------|
| src/mcp_server/server.py | Exposes RAG as MCP tools |
| src/agent/rag_agent.py | Google ADK agent |

---

## Python vs Java Cheat Sheet

| Java | Python | Example |
|------|--------|---------|
| `public class Foo {}` | `class Foo:` | Classes |
| `this.field` | `self.field` | Instance access |
| `new Foo()` | `Foo()` | No `new` keyword |
| `List<String>` | `list[str]` | Type hints |
| `Map<K,V>` | `dict[K, V]` | Type hints |
| `try/catch` | `try/except` | Error handling |
| `record Foo(...)` | `@dataclass class Foo` | Data classes |
| `@Test` | `def test_xxx()` | Testing |
| `Optional<T>` | `T \| None` | Nullable types |
| `System.getenv()` | `os.getenv()` | Environment vars |
| `void` | `-> None` | Return type |
| `final` | `frozen=True` | Immutability |
| Semicolons `;` | No semicolons | Syntax |
| Curly braces `{}` | Indentation | Code blocks |
| `import x.y.Z` | `from x.y import Z` | Imports |
| Maven/Gradle | pip + pyproject.toml | Package manager |
| JUnit | pytest | Testing |
| application.properties | .env + load_dotenv() | Config |

---

## Key Concepts Learned

### pipeline.py (Day 6)
- `RAGPipeline` = orchestrator (like Spring `@Service`). Constructor injects all 4 services.
- Two operations: `ingest_documents()` (store) and `query()` (search + answer)
- Ingest flow: load → chunk → embed → store in ChromaDB
- Query flow: embed question → search ChromaDB → build context → ask Gemini
- `[chunk.content for chunk in chunks]` = list comprehension = Java's `stream().map().collect()`
- `_build_context()` formats SearchResult objects into labeled text blocks for Gemini
- `_build_prompt()` wraps context + question with instructions ("answer using ONLY this context")
- RAG = Retrieval (found chunks) + Augmented (added to prompt) + Generation (Gemini answers)
- `self._llm = genai.GenerativeModel(...)` = Gemini chat model (different from embedding model)
- `get_stats()` = health check, called by scripts/query.py, MCP server, and ADK agent

### vector_store.py (Day 5)
- `VectorStore` = Spring `@Repository`. ChromaDB stores chunks + embeddings + metadata.
- `PersistentClient` saves to `./data/chroma_db/` (survives restarts). `EphemeralClient` = in-memory (for tests).
- `collection` = like a database table. `get_or_create_collection` = `CREATE TABLE IF NOT EXISTS`.
- `hnsw:space: cosine` = distance metric. Cosine measures vector direction, not length (ideal for text).
- Score = cosine distance: 0.0 = identical, 1.0 = completely different. Lower = better match.
- `upsert` = insert or update (like SQL MERGE). Same ID = update, new ID = insert. No duplicates.
- `collection.query()` returns nested lists `[[...]]` because it supports batch queries. We always use `[0]` (single query).
- `SearchResult(content, metadata, score)` = clean wrapper for ChromaDB raw results.
- `**query_params` = dict unpacking into keyword arguments (no Java equivalent).
- `clear()` = drop + recreate collection (ChromaDB has no 'delete all' method).
- ChromaDB methods are different from Vertex AI Search, but concepts are identical: connect → store → search → return.
- For interviews: know the pattern (Load → Chunk → Embed → Store → Search), not API method names.

### MCP Server - server.py (Days 8-10)
- **MCP = AI calling YOUR code** (not your code calling AI). Opposite direction from traditional APIs.
- `MCPServer` (MCP 2.0) replaced `FastMCP` (MCP 1.x) — like Spring Boot 2→3 migration.
- `@mcp.tool()` = `@PostMapping` — registers a function the AI can call. AI reads the docstring to decide when to use it.
- `@mcp.resource("rag://config")` = `@GetMapping` — read-only data endpoint, no parameters.
- `Annotated[str, "description"]` = `@RequestParam(description="...")` — tells AI what to pass.
- `mcp.run()` = `SpringApplication.run()` — starts event loop reading stdin, routes JSON requests to registered tools.
- Internally: `@mcp.tool()` stores functions in a `dict`. `run()` loops: read request → lookup tool by name → call it → return result.
- Communication: stdio (stdin/stdout JSON), not HTTP. No port conflicts, AI launches server as subprocess.
- For remote deployment: `mcp.run(transport="sse")` switches to HTTP.
- Lazy singleton `_get_pipeline()` with `global` keyword — Python needs `global` to reassign module-level variables.
- Tools return errors as text (not exceptions) so AI can report the error to the user.
- Claude Desktop config: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Config has: `command` (python path), `args` (module), `cwd` (working dir), `env` (PYTHONPATH)
- Use absolute paths in `.env` for `CHROMA_PERSIST_DIR` and `DOCUMENTS_DIR` — Claude Desktop may not respect `cwd`.
- Claude decides whether to use tools: generic questions → answers from memory; document-specific questions → calls your tool.
- MCP wraps traditional APIs so AI can use them (e.g., GitHub MCP wraps github.com REST API).

### ADK Agent - rag_agent.py (Days 11-12)
- **MCP Server** = give YOUR tools to SOMEONE ELSE'S AI (Claude). **ADK Agent** = build YOUR OWN AI with your tools.
- ADK Agent and MCP Server are independent paths — both call RAGPipeline, never each other.
- `Agent(name, model, instruction, tools=[...])` = the AI brain. `instruction` = system prompt with rules.
- ADK reads function name + docstring to understand tools (no decorators needed, unlike MCP).
- ADK tools return `dict` (structured data for Gemini). MCP tools return `str` (text for Claude).
- `Runner` = execution engine (like Tomcat). `InMemorySessionService` = conversation memory (like HttpSession).
- `Session` tracks conversation history. Same session_id = agent remembers. New session = fresh start.
- `async for event in runner.run_async()` = streams events: thinking → tool_call → tool_result → final_response.
- `async def` = Java's `CompletableFuture`. `await` = `.get()` but non-blocking. `async for` = reactive stream.
- Two Gemini calls per question: #1 Agent decides which tool → #2 RAG pipeline generates answer from chunks.
- Model name format: ADK uses `gemini-3.1-flash-lite` (no prefix). `gemini/` prefix is for litellm.
- Original code created new session per call (no memory). Fixed to reuse session across turns.
- `asyncio.new_event_loop()` + `loop.run_until_complete()` keeps same event loop across calls in CLI.
- `python -m` = run as module (package-aware). Without `-m`, `from src.xxx import` breaks.

### ADK + MCP Bridge (Day 13)
- ADK agents CAN connect to external MCP servers as tools (for GitHub, Slack, etc.)
- Not needed when agent calls your own functions directly — adds unnecessary complexity.
- Bridge only useful when mixing your tools with someone else's MCP server.

### Error Handling + Logging (Day 14)
- Retry with exponential backoff: `wait = 2 ** attempt * 5` → 5s, 10s, 20s. Java: `@Retryable` + `@Backoff`.
- `_generate_with_retry()` catches `genai_errors.ServerError` (503/429), retries, then raises on last attempt.
- Agent tools: `try/except` → return `{"error": str(e)}` so Gemini can tell the user what went wrong.
- MCP tools: `try/except` → return `f"Error: {e}"` as text so Claude can report the error.
- Pipeline logging: `logger.info("Query: %s", question[:100])` at entry and exit of query method.
- Python log levels: `DEBUG < INFO < WARNING < ERROR < CRITICAL` (Java: `TRACE < DEBUG < INFO < WARN < ERROR`).

### embeddings.py (Day 4)
- `EmbeddingService` calls Google AI's `text-embedding-004` to convert text → 768 numbers (vector)
- Similar texts → similar vectors; different texts → different vectors
- `genai.embed_content()` = SDK method that sends text to Google API, returns numbers
- `embed_texts()` = batch (many chunks), uses `task_type="retrieval_document"` (for storing)
- `embed_query()` = single query, uses `task_type="retrieval_query"` (for searching)
- Different task types improve search accuracy by ~5-10% — Google tunes vectors differently
- 5 task types: `retrieval_document`, `retrieval_query`, `semantic_similarity`, `classification`, `clustering`
- We only use retrieval_document + retrieval_query; others are for different AI use cases
- `semantic_similarity` = compare two texts directly (no DB). Use: duplicate detection, plagiarism
- `classification` = assign text to predefined categories. `clustering` = discover groups automatically
- Batch limit: 100 texts per API call (`_MAX_BATCH_SIZE = 100`)
- Free tier: 1,500 requests/day via Google AI Studio

### text_splitter.py (Day 3)
- **Why split:** whole files are too big for precise search. Chunks return the exact paragraph that answers the question.
- **Overlap (200 chars):** prevents losing sentences split at chunk boundaries
- `RecursiveCharacterTextSplitter` from `langchain-text-splitters` does the actual splitting (library method)
- Separators tried in order: `\n\n` → `\n` → `. ` → ` ` → `""` (prefers natural breaks)
- `{**doc.metadata, "chunk_index": i}` = dict spread = creates new dict merging existing + new key
- `chunk_index` = global sequential counter (0,1,2...) across ALL documents — used as primary key in ChromaDB
- ChromaDB stores: ID + document text + embedding vector + metadata
- Vector search uses ONLY embeddings (numbers) — chunk_index is just a database record ID
- ChromaDB = local embedded DB (like SQLite for vectors). No server setup needed.
- Vertex AI Search = same concept but hosted on Google Cloud (Week 3)

### document_loader.py (Day 2)
- `@dataclass` = Java record/POJO; `content: str` and `metadata: dict` are fields (like instance variables)
- `field(default_factory=dict)` = creates a NEW empty dict per instance (avoids Python's shared mutable default gotcha)
- `Path` from `pathlib` = Java's `java.nio.file.Path`; use `/` operator to join: `Path("dir") / "file.txt"`
- `self` = Java's `this` but must be explicit as first param in every instance method
- `_method_name` = private by convention (Python doesn't enforce like Java's `private`)
- `def method(self, param: Type) -> ReturnType:` = method signature; types are hints, not enforced
- `with open(file) as f:` = Java's try-with-resources (auto-closes file)
- `SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}` = Python `set` literal, O(1) `in` check
- Entry point: `scripts/ingest.py` → `RAGPipeline` → `DocumentLoader.load_all()`
- One Document per text file, one Document per PDF page — smaller units = better retrieval

### config.py (Day 1)
- `@dataclass(frozen=True)` = Java record (immutable data class)
- `field(default_factory=lambda: ...)` = lazy default, runs at object creation time
- `lambda: expr` = Java's `() -> expr` (Supplier<T>)
- `load_dotenv()` = reads .env file into environment (like Spring reads application.properties)
- `os.getenv("KEY", "default")` = `System.getenv().getOrDefault("KEY", "default")`
- AppConfig aggregates GeminiConfig + RAGConfig (like Spring @Configuration with @Bean methods)

### Embedding Models
- **text-embedding-004** (what we use): Google's model, converts text → 768-number vector
- Free via Google AI Studio, good for document Q&A
- Each dimension captures an "aspect" of meaning (not human-readable labels)

### Dense vs Sparse Embeddings
- **Dense (text-embedding-004):** 768 dimensions, ALL non-zero
  - Each dimension = blend of many concepts
  - Great at understanding MEANING ("garbage collection" matches "memory cleanup")
  - Used for: document Q&A, semantic search
  
- **Sparse (SPLADE):** ~30,000 dimensions, MOSTLY zeros (~50 non-zero)
  - Each dimension = a specific WORD in vocabulary
  - Great at exact KEYWORD matching
  - Used for: legal docs, product search, error code lookup

- **Why dense wins for our use case:** Users ask questions in natural language with synonyms.
  Dense understands "How does Python handle memory?" matches "reference counting" even though
  no words overlap. Sparse would miss this.

- **Key insight:** More dimensions ≠ more accuracy. It depends on WHAT they measure.
  768 meaning-dimensions beat 50 keyword-dimensions for Q&A because meaning > keywords.

- **Hybrid search** (production): combines both: `Score = 0.7 × Dense + 0.3 × Sparse`

### What Are Dimensions?
- Like coordinates in space: 2D = (lat, long), 3D = (lat, long, altitude)
- 768D = 768 different "aspects" of text meaning
- Zero = "this aspect is not relevant"
- Non-zero = "this aspect IS relevant, and here's how much"

---

## Useful Links (Reference Only)
- Google AI Studio API Key: https://aistudio.google.com/apikey
- Google Cloud Free Tier: https://cloud.google.com/free
- Google ADK Docs: https://google.github.io/adk-docs/
- MCP Specification: https://modelcontextprotocol.io
- Python for Java Devs: https://docs.python.org/3/tutorial/

---

## Cost Summary
| Component | Cost |
|-----------|------|
| Gemini 2.0 Flash | Free (15 RPM, 1M tokens/day) |
| text-embedding-004 | Free via Google AI Studio |
| ChromaDB | Free (runs locally) |
| Google AI Studio API key | Free (never expires) |
| Cloud Run (Week 3) | Free tier: 2M requests/month |
