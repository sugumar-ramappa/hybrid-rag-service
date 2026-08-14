# Q&A Reference - Day 1-10 Sessions (Aug 11-14, 2026)

## Day 1: Setup & Core Concepts

### Google AI Studio vs Vertex AI?
- **Google AI Studio**: Free, API key auth, no credit card, runs code locally
- **Vertex AI**: Google Cloud managed, needs project + billing, for production
- Same Gemini models, same quality answers
- We use AI Studio for weeks 1-2, switch to Vertex AI in week 3 for deployment
- The code change is small: swap auth method + a few imports

### How long does the API key last?
- **Never expires** — permanent until you manually delete it
- Free limits: 15 requests/min, 1,500 requests/day, 1M tokens/day
- Rate limits reset every 24 hours
- Can create multiple keys, can revoke anytime

### Why both pyproject.toml AND requirements.txt?
- `pyproject.toml` = full project definition (like `pom.xml`), modern standard
- `requirements.txt` = simple flat dependency list, legacy but widely used
- `pyproject.toml` used for: `pip install .` (package install)
- `requirements.txt` used for: Docker, CI/CD, quick setup
- Can use just `pyproject.toml` — requirements.txt is a convenience

### What is text-embedding-004?
- Google's model that converts text → 768-number vector
- Captures MEANING not keywords
- Free via Google AI Studio
- Max input: ~2,048 tokens (~1,500 words)
- "004" = version 4 (latest stable)
- Used twice: once for documents (ingest), once for queries (search)

### What is SPLADE?
- Hybrid embedding: keyword matching + semantic expansion
- Sparse vector: ~30,000 dimensions, mostly zeros
- Keeps keywords AND expands with related terms
- Better for exact-term tasks (legal, product search, error codes)
- NOT needed for our project — dense embeddings work better for Q&A

### What are dimensions?
- Like coordinates in space: 2D=(lat,long), 3D=(lat,long,alt), 768D=(meaning...)
- Each dimension captures one "aspect" of text meaning
- More dimensions = more detail, but not always more accuracy
- The model learns what each dimension represents (not human-readable)

### What are zeros in embeddings?
- Zero = "this aspect/word is not relevant to this text"
- Dense (ours): almost NO zeros — all 768 dimensions have values
- Sparse (SPLADE): MOSTLY zeros — only ~50 out of 30,000 are non-zero
- Each non-zero = "this word IS relevant, here's how much"

### Dense 768 vs Sparse 50: which is more accurate?
- NOT directly comparable — they measure different things
- Dense 768 = 768 active signals about MEANING (holistic understanding)
- Sparse 50 active = 50 specific KEYWORDS (exact word matching)
- Dense wins for Q&A: understands synonyms ("garbage collection" ≈ "memory cleanup")
- Sparse wins for exact terms: "NullPointerException" matches only that exact term
- For our use case (natural language Q&A): **dense wins**

### What is ChromaDB?
- Local vector database — stores and searches embeddings
- Like MySQL but for similarity search instead of SQL queries
- Runs on your Mac, persists to `./data/chroma_db/`
- Free, open-source, no cloud needed
- Java equivalent: embedded H2 database but for vectors

### ChromaDB vs Vertex AI Vector Search?
- ChromaDB: local, free, for learning/prototyping, millions of vectors
- Vertex AI Vector Search: Google Cloud, ~$360+/month, billions of vectors
- Same concept, different infrastructure
- Like H2 (local) vs Cloud SQL (production) in Java world

### Virtual environment — how often?
- `python3 -m venv .venv`: ONCE per project (creates it)
- `source .venv/bin/activate`: EVERY new terminal session
- `pip install -r requirements.txt`: ONCE (or when deps change)
- VS Code auto-activates if you select the .venv interpreter
- Prompt shows `(.venv)` when activated

---

## Day 1: config.py

### What is `@dataclass(frozen=True)`?
- Java's `record` — immutable data class
- Auto-generates `__init__`, `__eq__`, `__repr__`
- `frozen=True` = all fields are read-only after creation

### What is `field(default_factory=lambda: ...)`?
- Lazy default — runs at object creation time, not class definition time
- `lambda: expr` = Java's `() -> expr` (a Supplier<T>)
- Used when default value needs computation (reading env vars)

### What is `load_dotenv()`?
- Reads `.env` file and puts values into environment variables
- Like Spring reading `application.properties`
- Must be called before `os.getenv()` reads those values

### What is `os.getenv("KEY", "default")`?
- Java's `System.getenv().getOrDefault("KEY", "default")`
- Reads environment variable, returns default if not set

### How does AppConfig work?
- Root config that aggregates sub-configs (GeminiConfig + RAGConfig)
- Like Spring's main `@Configuration` class with `@Bean` methods
- `get_config()` = factory function (like `@Bean` method)
- `validate()` = checks required fields are present

### What is `lambda`?
- Anonymous function: `lambda: os.getenv("KEY")` = Java's `() -> System.getenv("KEY")`
- Used here because dataclass defaults can't be function calls directly

---

## Python Basics (from Java perspective)

### What is `dict`?
- Python's `dict` = Java's `HashMap<String, Object>`
- Heterogeneous values: `{"source": "file.txt", "page": 3, "is_valid": True}`
- Access: `d["key"]` (throws KeyError) or `d.get("key")` (returns None)
- Check: `"key" in d` = Java's `containsKey()`
- Merge: `{**d1, **d2}` = Java's `putAll()`

### What is `self`?
- Java's `this` but must be **explicit** as first parameter
- Required to access instance variables: `self.field`
- Required to call other methods: `self.method()`
- Convention name (not a keyword) — could be `this` but nobody does that

### What is `Path`?
- From `pathlib` module = Java's `java.nio.file.Path`
- Create: `Path("./documents")` = `Paths.get("./documents")`
- Join: `path / "file.txt"` = `path.resolve("file.txt")`
- Properties: `.suffix` (extension), `.name`, `.parent`
- Actions: `.exists()`, `.mkdir()`, `.iterdir()`

### How to read a method signature?
```python
def _load_file(self, file_path: Path) -> list[Document]:
```
- `def` = define a function
- `_load_file` = name, `_` prefix = private by convention
- `self` = instance reference (Java's `this`)
- `file_path: Path` = parameter with type hint (types NOT enforced)
- `-> list[Document]` = return type hint
- `:` = starts method body (next line, indented)

---

## document_loader.py

### What are `content` and `metadata`?
- `content: str` = the actual file text (goes into embedding/search)
- `metadata: dict` = info ABOUT the file (shown in citations)
- Example metadata: `{"source": "intro.txt", "type": ".txt", "page": 3}`

### Why `field(default_factory=dict)` instead of `= {}`?
- `= {}` creates ONE shared dict for ALL instances (Python gotcha!)
- `field(default_factory=dict)` creates a NEW dict for each instance
- Same as Java: never use `static Map` when you want instance-level state

### Why three file types (.txt, .md, .pdf)?
- `.txt/.md` = plain text, read with `open()`
- `.pdf` = binary format, needs `pypdf` library to extract text
- `.md` and `.txt` share same method (markdown IS plain text)
- Covers ~90% of knowledge base documents

### What does `sorted()` do?
- Sorts **file names** alphabetically (NOT file contents)
- Ensures reproducible order on every run
- Without it, `iterdir()` returns OS-dependent order

### What does DocumentLoader actually do?
- ONLY reads files from disk into memory
- Packages each as `Document(content=..., metadata=...)`
- Does NOT analyze or transform content
- Pure I/O layer — "get text off disk"

### Where does `documents_dir` come from?
- Constructor parameter — passed when creating the object
- `DocumentLoader("./documents")` → `documents_dir` = `"./documents"`
- Stored as `self._documents_dir` for use in other methods

---

## text_splitter.py

### Why split documents?
- Embedding models have token limits
- Whole files are too big for precise search
- Splitting returns the EXACT paragraph that answers the question
- Smaller chunks = more precise vectors = better search results

### What is overlap?
- Last 200 chars of chunk N repeated at start of chunk N+1
- Prevents losing sentences split at chunk boundaries
- Default: 200 chars overlap, 1000 chars chunk size

### How does `RecursiveCharacterTextSplitter` work?
- Library from `langchain-text-splitters` (we just configure it)
- Tries separators in order: `\n\n` → `\n` → `. ` → ` ` → `""`
- Falls to next separator only when chunk still too big
- Prefers natural text breaks over hard cuts

### What is `{**doc.metadata, "chunk_index": chunk_index}`?
- Dict spread operator — creates a NEW dict merging existing + new key
- Java equivalent: `Map.putAll()` + `Map.put()`
- Result: `{"source": "a.txt", "type": ".txt", "chunk_index": 0}`

### What is `chunk_index`?
- Global sequential counter (0, 1, 2...) across ALL documents
- NOT per-document — continues counting across files
- Used as primary key (ID) in ChromaDB

### Why chunk_index in both TextChunk field AND metadata?
- `TextChunk.chunk_index` = convenient code access
- `metadata["chunk_index"]` = survives into ChromaDB (DB only stores metadata dict)
- ChromaDB doesn't know about Python objects — only stores dict

---

## Vector Search & ChromaDB

### How is data stored in ChromaDB?
```
| ID        | Document (text)         | Embedding (768 numbers) | Metadata              |
|-----------|-------------------------|-------------------------|-----------------------|
| "chunk_0" | "Python is a lang..."   | [0.12, 0.45, ...]       | {source: "a.txt"...}  |
| "chunk_1" | "Functions use def..."  | [0.33, 0.11, ...]       | {source: "a.txt"...}  |
```
- SQLite stores: IDs, documents, metadata
- HNSW Index stores: embeddings (for fast similarity search)

### How does vector search work?
1. Query text → convert to embedding (768 numbers)
2. Compare query embedding to ALL stored embeddings (cosine similarity)
3. Find closest matches (nearest neighbors)
4. Return those chunks' text + metadata

### What's used during search?
- **Embedding** = the ONLY thing compared during search
- **ID (chunk_index)** = NOT used for search, just database housekeeping
- **Document text** = returned in results for display
- **Metadata** = returned for citation (source file name)

### Is chunk_index specific to ChromaDB?
- No — any vector DB needs a unique ID per record
- Could use content hash, UUID, or sequential number
- `chunk_index` is simplest choice for learning

### ChromaDB vs Vertex AI Search?
- ChromaDB: local, free, embedded (like SQLite), `pip install chromadb`
- Vertex AI Search: Google Cloud, pay per query, managed service
- Same concept: text + embedding + metadata + ID
- Same code for DocumentLoader and TextSplitter — only VectorStore changes

---

## Application Flow

```
scripts/ingest.py (entry point = Java's main())
    ↓
RAGPipeline.__init__() (creates all services)
    ↓
pipeline.ingest_documents()
    ├── DocumentLoader.load_all()      → list[Document]     (Day 2)
    ├── TextSplitter.split_documents() → list[TextChunk]    (Day 3)
    ├── EmbeddingService.embed_texts() → list[list[float]]  (Day 4)
    └── VectorStore.add_chunks()       → stored in ChromaDB  (Day 5)
```

---

## Real Data Example (from your documents/ folder)
- 2 sample files → loaded as 2 Documents
- Split into 13 chunks (at chunk_size=500)
- Each chunk gets unique ID: chunk_0 through chunk_12
- Metadata tracks source file for citation

---

## Days 8-10: MCP Server (Aug 14, 2026)

### What is MCP?
- Model Context Protocol — standard protocol for AI to call your code
- Like a REST API, but the client is an AI (Claude, Gemini) instead of a browser
- Direction: **AI → your server** (not your server → AI)
- Your `pipeline.py` calling Gemini = traditional API call (you → AI)
- MCP flips it: Claude calling `search_documents()` = AI → you

### Is MCP always "AI calls your code"?
- Yes — that's the entire point of the protocol
- Every MCP server works this way: GitHub MCP, Slack MCP, database MCP
- MCP doesn't replace traditional APIs — it **wraps** them so AI can use them
- Example: GitHub MCP server internally calls github.com REST API

### Why does AI need to call our service?
- AI doesn't know everything — it only knows its training data
- Your documents, your database, your company data = AI has no idea
- MCP gives AI the ability to DO things (create PR) and KNOW things (search your docs) beyond its training
- Without MCP: "I don't have access to your documents"
- With MCP: calls `search_documents` → returns answer with sources

### Where is the MCP server created?
- `MCPServer(name=..., version=..., description=...)` on line 31 of server.py
- `MCPServer` is from `mcp.server` package (MCP 2.0)
- Previously was `FastMCP` from `mcp.server.fastmcp` (MCP 1.x — removed in 2.0)
- Java equivalent: `@SpringBootApplication`

### Is FastMCP/MCPServer production-ready?
- Yes — officially recommended by MCP SDK
- Low-level `Server` class exists for full protocol control but rarely needed
- Like Spring Boot (high-level) vs raw Servlet (low-level)

### What do `@mcp.tool()` and `@mcp.resource()` do?
- `@mcp.tool()` = `@PostMapping` — AI can call with parameters, does something
- `@mcp.resource("rag://config")` = `@GetMapping` — AI reads data, no parameters
- The decorator registers the function in an internal dictionary (like Spring component scanning)
- The **docstring** is critical — AI reads it to decide WHEN to use the tool
- `Annotated[str, "description"]` = `@RequestParam(description="...")` — tells AI what to pass

### How does `mcp.run()` call tools internally?
1. `@mcp.tool()` stores each function in a `dict`: `{"search_documents": <function>, ...}`
2. `mcp.run()` starts an event loop reading JSON from stdin
3. AI sends `{"method": "tools/list"}` → server returns all tool names + descriptions
4. AI sends `{"method": "tools/call", "params": {"name": "search_documents", "arguments": {...}}}` → server looks up the function by name, calls it, returns the result via stdout
- Like a `Map<String, Function>` + a while loop reading requests

### Why stdio instead of HTTP?
- No port conflicts — multiple MCP servers can run simultaneously
- No networking setup needed
- AI launches your server as a child process — simple and secure
- For remote/production: switch to SSE over HTTP with `mcp.run(transport="sse")`

### What's the `global` keyword in `_get_pipeline()`?
- Python functions can READ module-level variables freely
- But to REASSIGN them (`_pipeline = RAGPipeline(...)`) you must declare `global`
- Without it, Python creates a new local variable instead of updating the module-level one
- Java has no equivalent — instance fields are always accessible with `this`

### Why return errors as text instead of raising exceptions?
- If the tool crashes (raises exception), the AI gets nothing
- By returning `f"Error: {e}"` as a string, the AI can tell the user what went wrong
- The tool still "succeeds" from the protocol's perspective — it returned a response

### Who calls `ingest_documents`?
- **Not** called during normal Q&A — Claude only calls `search_documents` for questions
- Called manually via `python -m scripts.ingest` before starting (setup step)
- Or Claude can call it if user says "please re-index my documents"
- Like a database migration — you run it during setup, not at runtime

### What happens without ingestion?
- ChromaDB is empty (0 chunks) — search returns no useful results
- The query runs fine, but there's nothing to match against
- Like querying an empty database table — works but returns nothing
- Ingested data persists in `data/chroma_db/` on disk — survives restarts

### How to configure Claude Desktop?
- Config file: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Add `mcpServers` section with: `command` (python path), `args` (module), `cwd` (working dir), `env` (PYTHONPATH)
- Must quit Claude fully (Cmd+Q) and reopen for config changes to take effect
- Check connection: Settings → Developer → look for server name with green/red status

### How to tell if Claude used my tool vs its own knowledge?
- **Used tool**: response mentions your document filenames (sample_python_basics.txt), shows "Used tool: search_documents" indicator
- **Own knowledge**: generic answer with no document references
- Claude decides: generic questions → answers from memory; document-specific questions → calls your tool
- To force tool use: ask "Search my documents for..." or "What does my RAG guide say about..."

### Why did I get a "read-only file system" error?
- Claude Desktop wasn't using `cwd` properly → relative paths (`./data/chroma_db`) failed
- Fix: use absolute paths in `.env` for `CHROMA_PERSIST_DIR` and `DOCUMENTS_DIR`
- Always use absolute paths when your server runs from an external launcher

---

## Days 11-12: ADK Agent (Aug 14, 2026)

### Why do we need ADK Agent when MCP + Claude already works?
- **MCP + Claude**: give YOUR tools to SOMEONE ELSE'S AI. Needs Claude Desktop.
- **ADK Agent**: build YOUR OWN AI product. Self-contained, deploy as web app/chatbot.
- ADK agent is the backend brain for frontends (Streamlit, React, Slack bots).
- Both exist in this project for learning — in production, pick one based on use case.

### Does the ADK Agent call MCP server.py?
- No. They're completely independent paths into the same RAG pipeline.
- ADK agent calls `search_knowledge_base()` directly (Python function).
- MCP server exposes `search_documents()` for Claude to call.
- Both call `pipeline.query()` underneath.

### Why are return types different (dict vs str)?
- ADK tools return `dict` — Gemini reads structured data, decides how to present it.
- MCP tools return `str` — Claude receives pre-formatted text.
- `dict` = returning JSON from a REST API. `str` = returning HTML from a servlet.

### What is Session and Runner?
- `InMemorySessionService` = stores conversations in RAM (like `HttpSession`).
- `Session` = one conversation's history. Same `session_id` = agent remembers previous turns.
- `Runner` = execution engine that wires agent + session + tools (like Tomcat/DispatcherServlet).
- `async for event in runner.run_async()` = streams events as agent works (thinking → tool call → answer).

### How does session memory work?
- `InMemorySessionService` CAN remember across turns, but original code created new session per call.
- Fixed: reuse `_session_service`, `_runner`, `_session_id` across calls (module-level globals).
- `asyncio.new_event_loop()` keeps same event loop in CLI for session reuse.
- Memory limit: bounded by LLM context window (~32K-2M tokens), not RAM.

### How many Gemini calls per agent question?
- Two: #1 Agent decides which tool to call. #2 RAG pipeline generates answer from chunks.
- Without agent (scripts/query.py): only one Gemini call (pipeline only).

### What is `python -m`?
- `-m` = run as module (package-aware). Adds current directory to Python path.
- Without `-m`: `from src.config import ...` fails because Python doesn't know about packages.
- Java equivalent: `java -cp . com.example.Main` vs `java Main.java`.

### Is Streamlit production-ready?
- Yes for internal tools, dashboards, MVPs (~100 concurrent users).
- Not for customer-facing products with millions of users — use React/Next.js for that.
- ~20 lines of Python for a web chat UI. No new language needed.

---

## Days 13-14: Bridge + Error Handling (Aug 14, 2026)

### What is ADK + MCP bridge?
- ADK agents can connect to external MCP servers to use their tools.
- Only needed for external tools (GitHub MCP, Slack MCP) — not for your own functions.
- Your project uses direct function calls — simpler and correct.

### What is exponential backoff?
- Retry with increasing wait times: 5s → 10s → 20s (`2^attempt × 5`).
- Handles transient API errors (503 rate limits, 429 too many requests).
- Java equivalent: `@Retryable(maxAttempts=3, backoff=@Backoff(delay=5000, multiplier=2))`.
- Added in `pipeline._generate_with_retry()`.

### Why return errors instead of raising exceptions?
- Agent tools: return `{"error": str(e)}` → Gemini reads it and tells the user.
- MCP tools: return `f"Error: {e}"` → Claude reads it and tells the user.
- If the tool crashes (raises), the AI gets nothing and can't help the user.

### What are Python log levels?
- `DEBUG < INFO < WARNING < ERROR < CRITICAL`
- Java equivalent: `TRACE < DEBUG < INFO < WARN < ERROR`
- `logger.info()` = normal operations. `logger.warning()` = something unexpected but recoverable.
- `logger.exception()` = logs error + full stack trace (like Java's `log.error("msg", e)`).

---

## Git & GitHub

### How to use different emails for personal vs work repos?
- `git config user.email` (without `--global`) = sets email for THIS repo only.
- `git config --global includeIf."gitdir:~/Learning Project/".path "~/Learning Project/.gitconfig"` = auto-apply personal email to all repos in that folder.
- Global config (THD) applies everywhere else.

### How to undo a commit?
- `git update-ref -d HEAD` = undo the very first commit (keeps files staged).
- `git reset --soft HEAD~1` = undo any other commit (keeps files staged).
- `git push --force` = overwrite remote after rewriting history.

---

## Week 3: Production (Aug 14, 2026)

### What is structured logging?
- JSON-formatted logs that machines can parse (Grafana, CloudWatch, etc.)
- Dev mode: `2026-08-14 [INFO] pipeline: Query: What is Python?` (human-readable)
- Prod mode: `{"timestamp":"...","level":"INFO","message":"Query: What is Python?"}` (JSON)
- Switch with `setup_logging(json_output=True/False)`.
- Java equivalent: Logback with `JsonEncoder` vs `PatternLayout`.

### What is OWASP?
- Open Web Application Security Project — publishes top 10 security risks.
- Our protections: input validation (OWASP #3), path traversal (OWASP #1), secrets in env vars (OWASP #2).

### What is `top_k` validation?
- `top_k` = how many search results to return from ChromaDB.
- Clamped to 1-20: `max(1, min(top_k, 20))` prevents zero results or excessive token usage.
- Java equivalent: `@Min(1) @Max(20)`.

### What is path traversal protection?
- Attack: symlink inside `documents/` pointing to `/etc/passwd`.
- Protection: `file_path.resolve().is_relative_to(documents_dir.resolve())`.
- Java equivalent: `path.normalize().startsWith(baseDir)`.
- In our project: low risk (we control the folder), but defensive coding for production reuse.

### What is Docker layer caching?
- Copy `requirements.txt` BEFORE code → deps layer cached if unchanged.
- Code changes don't re-trigger `pip install` (saves minutes on rebuild).
- Same concept as Maven dependency caching in Java Docker builds.

### Why non-root user in Docker?
- `RUN useradd appuser` + `USER appuser` — never run as root.
- If app is hacked, attacker only has limited permissions.
- `chown -R appuser /app/data` needed for ChromaDB write access.

### Why absolute paths failed in Docker?
- `.env` had `/Users/sugumar/.../documents` — doesn't exist inside container.
- Fix: override with `-e DOCUMENTS_DIR=./documents` at runtime.
- Container filesystem starts at `/app/` — relative paths work inside it.

### What is Streamlit?
- Python library for building web UIs with ~20 lines of code.
- `st.chat_input()` = input box, `st.chat_message()` = chat bubble, `st.session_state` = memory.
- Good for internal tools, demos, MVPs. Not for millions of users.
- Java equivalent: Vaadin or Thymeleaf (but much less code).

### What is FastAPI?
- Python's Spring Boot — framework for building REST APIs.
- `@app.post("/query")` = `@PostMapping("/query")`.
- Auto-generates Swagger docs at `/docs`.
- Uses Pydantic for validation (like Lombok + Bean Validation).
- `uvicorn api:app --reload` = run the server with auto-restart.

### What is Pydantic?
- Data validation library. Java equivalent: Lombok + `@Valid` annotations.
- `BaseModel` = DTO class. `Field(min_length=1, max_length=2000)` = `@NotBlank @Size(max=2000)`.
- Auto-converts JSON → Python objects and validates on the way in.

### What does `uvicorn api:app --reload` mean?
- `uvicorn` = ASGI server (like Tomcat). `api` = file name. `app` = FastAPI instance.
- `--reload` = auto-restart on code change (like Spring DevTools).

### MCP vs FastAPI vs Streamlit — when to use?
- **MCP Server**: API for AI (Claude, VS Code call it via stdio/JSON).
- **FastAPI**: API for apps (mobile apps, Slack bots call it via HTTP REST).
- **Streamlit**: UI for humans (browser chat interface).

### What is fine-tuning vs RAG?
- **RAG**: search docs at query time. Data stays external. Update anytime.
- **Fine-tuning**: train model on your data. Knowledge baked in permanently. Expensive ($50-500+).
- RAG = looking up answers in a database. Fine-tuning = compiling knowledge into the app.
- Use RAG for factual Q&A. Use fine-tuning for style/tone/domain expertise.

### What is Vertex AI?
- Google Cloud's managed AI platform. Same Gemini models as AI Studio.
- Needs GCP account + credit card. Code change: swap `api_key=` for `vertexai=True`.
- Only needed for production deployment at scale, SLA guarantees, or VPC isolation.

---

## Day 4: embeddings.py

### What does EmbeddingService do?
- Calls Google AI's `text-embedding-004` model via API
- Converts text → 768 numbers (a vector) that represent its meaning
- Similar texts get similar numbers, different texts get different numbers
- Example: "Python programming" and "coding in Python" → similar vectors

### What is `genai.embed_content()`?
- Google AI Studio SDK method — sends text to Google, gets numbers back
- Like calling a REST API but the SDK handles HTTP for you
- Free tier: 1,500 requests/day

### Why two methods: `embed_texts()` vs `embed_query()`?
- `embed_texts()` — for storing documents, uses `task_type="retrieval_document"`
- `embed_query()` — for search questions, uses `task_type="retrieval_query"`
- Google optimizes vectors differently for each (5-10% accuracy improvement)
- Both work without task types, but using them gives better search results

### What are all the task types in Google AI?
| Task type | Purpose | Used in our project? |
|-----------|---------|:----:|
| `retrieval_document` | Store docs for later search | Yes |
| `retrieval_query` | Search query to find docs | Yes |
| `semantic_similarity` | Compare two texts directly (no DB) | No |
| `classification` | Categorize text into labels | No |
| `clustering` | Discover groups in text data | No |

### Why different task types for storing vs searching?
- Documents STATE facts: "Functions are defined with def"
- Queries ASK questions: "How do I define a function?"
- Worded very differently but same topic
- Google tunes vectors so queries match documents better
- Like a librarian translating your question into book-label language

### What is `semantic_similarity` for?
- Comparing two texts directly — no database involved
- Use cases: duplicate detection, plagiarism check, FAQ matching
- Example: "My payment failed" vs "Payment not going through" → 92% similar
- Symmetric comparison (both texts treated the same way)

### What are classification and clustering for?
- **Classification:** assign text to predefined categories (spam/not spam, sentiment)
- **Clustering:** discover groups automatically in unlabeled data (topic discovery)
- Neither is used in RAG — they're separate AI applications

### What is batch processing in `embed_texts()`?
- Google allows max 100 texts per API call
- `_MAX_BATCH_SIZE = 100` — processes 100 at a time
- Like Java's `Lists.partition(texts, 100)` then process each batch
- `texts[i : i + 100]` = Python slice = Java's `subList(i, i+100)`

---

## Day 5: vector_store.py

### What is VectorStore?
- Like a Spring `@Repository` — stores and searches data
- Uses ChromaDB: local embedded vector database (like SQLite for vectors)
- `PersistentClient` saves to disk, `EphemeralClient` is in-memory (tests)
- `collection` = like a database table

### What is cosine distance?
- Measures angle between vectors (direction, not length)
- Score 0.0 = identical, 1.0 = completely different
- Lower score = better match
- `distance = 1 - similarity` (ChromaDB returns distance, not similarity)
- Cosine is best for text because it ignores text length, only cares about meaning

### What is `hnsw:space: cosine`?
- Two different things: `hnsw` = search algorithm, `cosine` = distance formula
- Configures HOW ChromaDB compares vectors (not related to Google's task_type)
- ChromaDB supports: `cosine` (text), `l2`/Euclidean (images), `ip`/inner product (recommendations)

### What is `upsert()`?
- Insert OR update (like SQL MERGE)
- Same ID = updates existing record. New ID = inserts new record
- No duplicates. Safe to run ingestion multiple times

### What is `collection_name`?
- Like a table name in SQL. Ours is `"rag_documents"`
- Can have multiple collections in same ChromaDB for different data

### How does `search()` work?
- Takes query_embedding (768 numbers from embed_query)
- ChromaDB compares against all stored embeddings
- Returns top_k closest matches sorted by score (lowest first)
- `where` parameter = optional metadata filter (like SQL WHERE)

### Why does ChromaDB return `[[...]]` (nested lists)?
- ChromaDB supports batch queries (multiple searches at once)
- We always send 1 query, so we access `[0]` for first (only) result set
- `results["documents"][0][i]` = first query's i-th match

### What is `SearchResult`?
- Dataclass with 3 fields: `content` (text), `metadata` (source info), `score` (distance)
- Defined at top of vector_store.py (lines 30-34)
- Clean wrapper around raw ChromaDB response

### What is `**query_params`?
- Dict unpacking into keyword arguments
- `func(**{"a": 1, "b": 2})` = `func(a=1, b=2)`
- No direct Java equivalent

### Why `clear()` drops and recreates?
- ChromaDB has no "delete all items" method
- So we drop the collection + create fresh one
- Like `DROP TABLE` + `CREATE TABLE` in SQL

### ChromaDB vs Vertex AI Search?
- Different APIs, same concepts: connect → store → search → return
- Only `VectorStore` class changes when switching. Rest of pipeline stays same.
- Vertex AI Search can auto-embed (you send text, it creates vectors internally)

### Interview tip:
- Know the RAG pattern: Load → Chunk → Embed → Store → Search → LLM answers
- Don't memorize API methods — know concepts (they're the same across all vector DBs)

---

## Day 6: pipeline.py

### What is RAGPipeline?
- Orchestrator that ties all services together (like Spring's main `@Service`)
- Constructor creates: DocumentLoader, TextSplitter, EmbeddingService, VectorStore, Gemini LLM
- Like `@Autowired` constructor injection in Spring

### What is `ingest_documents()`?
- Stores documents for later search
- Flow: load files → chunk text → embed chunks → store in ChromaDB
- Returns count of chunks ingested

### What is `query()`?
- Answers a user's question using RAG
- Flow: embed question → search ChromaDB → build context → ask Gemini
- Returns `QueryResult(answer, sources, query)`

### What is the difference between `chunks` and `texts`?
- `chunks` = list of TextChunk objects (content + metadata + chunk_index)
- `texts` = list of plain strings extracted from chunks via `[chunk.content for chunk in chunks]`
- `embed_texts()` needs plain strings, not TextChunk objects
- Both are used together: chunks for metadata, texts for embedding

### What is `_build_context()`?
- Converts SearchResult objects into a formatted text string for Gemini
- Adds source labels: `[Source 1: python.txt]`, `[Source 2: guide.pdf (page 3)]`
- Joins with `---` separators between chunks

### What is `_build_prompt()`?
- Wraps context + question with instructions for Gemini
- Key instruction: "Use ONLY the information from the context" — prevents hallucination
- "Always cite which source(s) you used" — ensures traceability

### What does RAG stand for?
- **R**etrieval: find relevant chunks from ChromaDB
- **A**ugmented: add those chunks to the prompt as context
- **G**eneration: Gemini generates an answer grounded in your documents

### What is `get_stats()`?
- Health check returning document count + model names
- Called by: `scripts/query.py`, MCP server, ADK agent
- Like Spring Actuator `/health` endpoint
