# logpose-ui — Document Intelligence Web UI

**System 2 of 3.** Standalone Flask web application — full browser UI for searching,
previewing, and chatting with your local documents.

> Completely independent from `logpose-mcp` and `poneglyph`. Separate codebase, separate ChromaDB.

## What it does

- Full search UI at `http://localhost:7892`
- Filter by file type, source folder, date
- Click any result → full document preview with matched chunk highlighted
- Chat with any document via local LLM (qwen3:0.6b)
- One-click "Open file" / "Copy path"
- Indexing dashboard with real-time progress

## Setup

```bash
cd logpose-ui
python -m venv .venv && source .venv/bin/activate
pip install -e ".[full]"        # full: parsers + OCR + reranker
# or:
pip install -e ".[minimal]"     # PDF only, fast install
```

```bash
cp .env.example .env
# Edit .env — set LOGPOSE_CORPUS_DIR to your documents folder
```

## Run

```bash
aether-logpose-ui
# Opens at http://127.0.0.1:7892
```

## Index your documents

Open the browser → click **Indexing** in the sidebar → click **Re-index now**.

Or set `LOGPOSE_AUTO_INDEX_ON_START=true` in `.env` to index automatically on startup.

## Directory layout

```
logpose-ui/
├── logpose/
│   ├── web_server.py    ← Flask entry point + REST API
│   ├── config.py        ← all settings (LOGPOSE_ env vars)
│   ├── embeddings.py    ← Ollama embed client
│   ├── searcher.py      ← hybrid search + reranker
│   ├── indexer.py       ← file indexer
│   ├── bm25_index.py    ← SQLite FTS5 BM25
│   ├── parsers.py       ← PDF/DOCX/MD/EPUB parsers
│   └── watcher.py       ← filesystem watcher
├── web/
│   └── Logpose.html     ← single-file React UI
├── tests/
├── chroma_db/           ← created on first index (gitignored)
├── .env                 ← your config (gitignored)
├── .env.example
└── pyproject.toml
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serves `web/Logpose.html` |
| GET | `/api/search?q=…` | Hybrid search |
| GET | `/api/file-content?path=…` | All chunks for a file |
| GET | `/api/stats` | Index statistics |
| GET | `/api/status` | Server status |
| POST | `/api/index` | Trigger re-index |
| GET | `/api/index-status` | Index job progress |
| POST | `/api/open-file` | Open file in OS |
| POST | `/api/ask-followup` | LLM Q&A on a document |
