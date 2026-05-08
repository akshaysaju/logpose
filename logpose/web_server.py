"""
logpose/web_server.py — System 2: Logpose standalone product server.

Serves web/Logpose.html at / and exposes REST API backed by the real
ChromaDB + BM25 search engine. This is the *product* path; the pure MCP
server (System 1) lives in logpose/server.py — both share the same core.

Port: 7892  (poneglyph face UI is 7891 — no conflict)
Entry point: aether-logpose-ui

Usage:
    LOGPOSE_CORPUS_DIR=~/your/docs LOGPOSE_CHROMA_DIR=./chroma_db aether-logpose-ui
    # or:
    python -m logpose.web_server
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from flask import Flask, jsonify, request, send_file  # noqa: E402
import chromadb  # noqa: E402

from logpose.config import settings  # noqa: E402
from logpose.bm25_index import BM25Index  # noqa: E402
from logpose.searcher import FileSearcher  # noqa: E402
from logpose.agent import RAGAgent as _RAGAgent  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("logpose.web_server")

# ── ChromaDB + Searcher ───────────────────────────────────────────────────────
settings.chroma_dir.mkdir(parents=True, exist_ok=True)
_client = chromadb.PersistentClient(path=str(settings.chroma_dir))
_collection = _client.get_or_create_collection(
    name=settings.collection_name,
    metadata={"hnsw:space": "cosine"},
)
_bm25 = BM25Index(settings.bm25_db_path)
_searcher = FileSearcher(settings, _collection, _bm25)
logger.info(
    "ChromaDB collection '%s' opened (%d items)",
    settings.collection_name,
    _collection.count(),
)

# Runtime corpus dirs added via /api/add-corpus or /api/upload (not in .env)
_extra_corpus_dirs: list[Path] = []
_uploads_dir: Path = settings.chroma_dir.parent / "uploads"
_uploads_dir.mkdir(parents=True, exist_ok=True)

# ── Single shared event loop for all async work ───────────────────────────────
# Flask uses a thread pool; each thread is a different OS thread.  asyncio
# primitives (Semaphore, httpx.AsyncClient) bind to the loop they're first
# used in.  Running them in different per-thread loops → "bound to a different
# event loop".  Fix: one background thread runs a single asyncio loop forever;
# all Flask handlers submit coroutines to it via run_coroutine_threadsafe().
_async_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
threading.Thread(target=_async_loop.run_forever, daemon=True, name="async-loop").start()


def _run_async(coro, timeout: float = 120.0):
    future = asyncio.run_coroutine_threadsafe(coro, _async_loop)
    return future.result(timeout=timeout)


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.logger.setLevel(logging.WARNING)

_WEB_DIR = _ROOT / "web"
_HTML_PATH = _WEB_DIR / "Logpose.html"


@app.after_request
def _cors(response):
    origin = request.headers.get("Origin", "")
    if "localhost" in origin or "127.0.0.1" in origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/")
def index():
    return send_file(_HTML_PATH)


# ── /api/stats ────────────────────────────────────────────────────────────────
@app.route("/api/stats")
def api_stats():
    try:
        total_chunks = _collection.count()
        base = {
            "corpus_dirs": [str(d) for d in settings.corpus_dirs + _extra_corpus_dirs],
            "embed_model": settings.embed_model,
            "reranker_enabled": settings.enable_reranker,
            "reranker_model": settings.reranker_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "ocr_enabled": settings.enable_ocr,
            "ocr_min_chars": settings.ocr_min_chars_per_page,
        }
        if total_chunks == 0:
            return jsonify({**base, "total_chunks": 0, "total_files": 0,
                            "file_types": {}, "sources": [], "last_indexed": None})

        raw = _collection.get(include=["metadatas"], limit=min(total_chunks, 50_000))
        metas = raw.get("metadatas") or []

        file_per_ext: dict[str, int] = {}
        sources: set[str] = set()
        file_paths: set[str] = set()
        last_indexed: str | None = None

        for meta in metas:
            if not isinstance(meta, dict):
                continue
            fp = meta.get("file_path", "")
            if fp:
                file_paths.add(fp)
            src = meta.get("source", "")
            if src:
                sources.add(src)
            ia = meta.get("indexed_at", "")
            if ia and (last_indexed is None or ia > last_indexed):
                last_indexed = ia
            if int(meta.get("chunk_index", 0)) == 0:
                ext = meta.get("extension", "unknown")
                file_per_ext[ext] = file_per_ext.get(ext, 0) + 1

        return jsonify({
            **base,
            "total_chunks": total_chunks,
            "total_files": len(file_paths),
            "file_types": file_per_ext,
            "sources": sorted(sources),
            "last_indexed": last_indexed,
        })
    except Exception as exc:
        logger.exception("api_stats failed")
        return jsonify({"error": str(exc)}), 500


# ── /api/search ───────────────────────────────────────────────────────────────
import re as _re

# Words that add no lexical value to a BM25 keyword search
_BM25_NOISE = frozenset({
    "docs", "document", "documents", "file", "files", "folder", "folders",
    "regarding", "about", "find", "search", "show", "tell", "give", "list",
    "related", "get", "on", "in", "the", "a", "an", "any", "some",
    "what", "which", "where", "who", "how", "why", "of", "for", "with",
})

# Compound phrases that BM25 splits badly → normalise to one token
_COMPOUNDS = [
    (r'\bopen\s+ai\b', 'OpenAI'),
    (r'\bmachine\s+learning\b', 'machine_learning'),
    (r'\bdeep\s+learning\b', 'deep_learning'),
    (r'\bnatural\s+language\s+processing\b', 'NLP'),
    (r'\blarge\s+language\s+model\b', 'LLM'),
    (r'\bneural\s+network\b', 'neural_network'),
    (r'\breinforcement\s+learning\b', 'reinforcement_learning'),
]


def _bm25_query(q: str) -> str:
    """Clean a natural-language query for BM25: normalise compounds + strip noise words.

    Semantic embeddings understand phrases like "docs regarding open ai" fine;
    BM25 tokenises naively and matches "open" against "open source", "open weight"
    etc.  Stripping noise and merging compound terms cuts false positives.
    """
    out = q
    for pattern, replacement in _COMPOUNDS:
        out = _re.sub(pattern, replacement, out, flags=_re.IGNORECASE)
    tokens = [t for t in out.split() if t.lower() not in _BM25_NOISE]
    return " ".join(tokens) if tokens else q.strip()


def _ext_to_kind(ext: str) -> str:
    return {
        ".pdf": "pdf", ".docx": "docx", ".doc": "docx",
        ".xlsx": "xlsx", ".xls": "xlsx",
        ".md": "md", ".markdown": "md",
        ".epub": "epub", ".html": "html", ".htm": "html",
        ".py": "code", ".js": "code", ".ts": "code",
        ".jsx": "code", ".tsx": "code", ".go": "code",
        ".rs": "code", ".java": "code", ".cpp": "code",
        ".c": "code", ".h": "code",
        ".png": "img", ".jpg": "img", ".jpeg": "img",
    }.get(ext.lower(), "doc")


def _fmt_date(iso: str) -> str:
    if not iso:
        return "—"
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y")
    except Exception:
        return iso[:10]


def _format_snippet(text: str, query: str) -> str:
    lower = text.lower()
    q_lower = query.lower()
    pos = lower.find(q_lower)
    if pos == -1:
        return text[:300].replace("\n", " ")
    start = max(0, pos - 80)
    end = min(len(text), pos + len(q_lower) + 220)
    before = ("…" if start > 0 else "") + text[start:pos].replace("\n", " ")
    match = text[pos:pos + len(q_lower)]
    after = (text[pos + len(q_lower):end].replace("\n", " ")
             + ("…" if end < len(text) else ""))
    return f"{before}@@{match}@@{after}"


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "q parameter required"}), 400

    n = min(int(request.args.get("n", 10)), 20)
    types_raw = request.args.get("types", "")
    source = request.args.get("source", "") or None

    exts = [t.strip() for t in types_raw.split(",") if t.strip()] if types_raw else []
    # Semantic search uses the full natural-language query (model handles intent well).
    # BM25 search gets a cleaned version: noise words stripped, compounds merged.
    q_bm25 = _bm25_query(q)
    logger.info("search q=%r  bm25_q=%r", q, q_bm25)

    t0 = time.monotonic()
    try:
        if exts:
            seen: dict = {}
            for ext in exts:
                rs = _run_async(_searcher.search(query=q, n_results=n,
                                                 filter_extension=ext, filter_source=source,
                                                 bm25_query=q_bm25))
                for r in rs:
                    k = f"{r.file_path}::{r.chunk_index}"
                    if k not in seen:
                        seen[k] = r
            results = sorted(seen.values(), key=lambda r: -r.score)[:n]
        else:
            results = _run_async(_searcher.search(query=q, n_results=n,
                                                  filter_extension=None, filter_source=source,
                                                  bm25_query=q_bm25))
    except Exception as exc:
        logger.exception("search failed q=%r", q)
        return jsonify({"error": str(exc)}), 500

    elapsed_ms = round((time.monotonic() - t0) * 1000)

    # Drop results significantly below the top score (RRF noise cutoff)
    # 0.65 threshold (up from 0.60) reduces noise; 0.015 absolute floor
    if len(results) > 1:
        top = results[0].score
        cutoff = max(top * 0.65, 0.015)
        results = [r for r in results if r.score >= cutoff]

    out = []
    for r in results:
        p = Path(r.file_path)
        ext = p.suffix.lower()
        out.append({
            "id": f"{r.file_path}::{r.chunk_index}",
            "kind": _ext_to_kind(ext),
            "file": p.name,
            "path": r.file_path,
            "cite": r.loc or (f"lines {r.line_start}–{r.line_end}" if r.line_start else "—"),
            "score": round(r.score, 3),
            "mod": _fmt_date(r.metadata.get("modified_at", "")),
            "snippet_raw": _format_snippet(r.chunk_text, q),
            "chunk_text": r.chunk_text,
        })

    return jsonify({"query": q, "results": out, "total": len(out), "duration_ms": elapsed_ms})


# ── /api/agent-search ─────────────────────────────────────────────────────────
@app.route("/api/agent-search", methods=["POST", "OPTIONS"])
def api_agent_search():
    """Agentic RAG: LLM agent iterates search/get_file/filter tools to answer queries."""
    if request.method == "OPTIONS":
        return "", 204
    try:
        body = request.get_json(force=True) or {}
        query = (body.get("query") or body.get("q") or "").strip()
        if not query:
            return jsonify({"error": "query is required"}), 400
        max_steps = int(body.get("max_steps") or settings.agent_max_steps)
        max_steps = max(1, min(max_steps, 10))  # clamp 1–10

        agent = _RAGAgent(_searcher, max_steps=max_steps)
        result = _run_async(agent.run(query), timeout=300.0)

        return jsonify({
            "answer": result.answer,
            "citations": result.citations,
            "reasoning_trace": result.reasoning_trace,
            "steps_used": result.steps_used,
            "model": result.model,
        })
    except Exception as exc:
        logger.exception("api_agent_search failed")
        return jsonify({"error": str(exc)}), 500


# ── /api/open-file ────────────────────────────────────────────────────────────
@app.route("/api/open-file", methods=["POST", "OPTIONS"])
def api_open_file():
    if request.method == "OPTIONS":
        return "", 204
    data = request.get_json(silent=True) or {}
    path = data.get("path", "").strip()
    if not path:
        return jsonify({"error": "path required"}), 400
    p = Path(path)
    if not p.exists():
        return jsonify({"error": f"File not found: {path}"}), 404
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        elif sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", str(p)])
        else:
            subprocess.Popen(["explorer", str(p)], shell=True)
        return jsonify({"ok": True})
    except Exception as exc:
        logger.exception("open-file failed path=%r", path)
        return jsonify({"error": str(exc)}), 500


# ── /api/file-content ────────────────────────────────────────────────────────
@app.route("/api/file-content")
def api_file_content():
    path = request.args.get("path", "").strip()
    if not path:
        return jsonify({"error": "path required"}), 400
    try:
        raw = _collection.get(
            where={"file_path": path},
            include=["documents", "metadatas"],
        )
        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []
        chunks = sorted(
            [
                {"text": doc, "chunk_index": int(m.get("chunk_index", 0))}
                for doc, m in zip(docs, metas)
                if doc
            ],
            key=lambda x: x["chunk_index"],
        )
        return jsonify({"chunks": chunks, "total": len(chunks)})
    except Exception as exc:
        logger.exception("file-content failed path=%r", path)
        return jsonify({"error": str(exc)}), 500


# ── /api/ask-followup ─────────────────────────────────────────────────────────
@app.route("/api/ask-followup", methods=["POST", "OPTIONS"])
def api_ask_followup():
    if request.method == "OPTIONS":
        return "", 204
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    chunk_text = data.get("chunk_text", "").strip()
    file_path = data.get("path", "").strip()

    if not question:
        return jsonify({"error": "question required"}), 400

    # Gather context: matched chunk + up to 3 more from same file
    context_chunks = [chunk_text] if chunk_text else []
    if file_path and len(context_chunks) < 4:
        try:
            raw = _collection.get(
                where={"file_path": file_path},
                include=["documents"],
                limit=20,
            )
            for doc in (raw.get("documents") or []):
                if doc and doc not in context_chunks:
                    context_chunks.append(doc)
                    if len(context_chunks) >= 4:
                        break
        except Exception:
            pass

    context = "\n\n---\n\n".join(context_chunks[:4])
    fname = Path(file_path).name if file_path else "the document"
    # /no_think disables Qwen3's chain-of-thought to keep responses fast
    prompt = (
        f"/no_think\n"
        f"Answer briefly using ONLY the excerpt below. 2-3 sentences max. "
        f"If the answer isn't there, say so.\n\n"
        f"FILE: {fname}\n"
        f"<excerpt>\n{context[:3000]}\n</excerpt>\n\n"
        f"Q: {question}"
    )

    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    chat_model = os.environ.get("LOGPOSE_CHAT_MODEL", "qwen3:0.6b")

    try:
        import httpx
        resp = httpx.post(
            f"{ollama_url}/api/chat",
            json={
                "model": chat_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"num_predict": 256, "temperature": 0.1},
            },
            timeout=45.0,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()
        # Strip any residual <think>...</think> blocks qwen3 may emit
        import re
        answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return jsonify({"answer": answer or raw, "model": chat_model})
    except Exception as exc:
        logger.exception("ask-followup failed")
        return jsonify({"error": str(exc)}), 500


# ── /api/health ───────────────────────────────────────────────────────────────
@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "agent_enabled": settings.agent_enabled,
                    "agent_model": settings.agent_model})


# ── /api/status ───────────────────────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    try:
        total = _collection.count()
        return jsonify({
            "total_chunks": total,
            "status": "idle",
            "message": f"{total:,} chunks indexed.",
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── /api/index + /api/index-status ───────────────────────────────────────────
_index_job: dict = {
    "running": False, "message": "idle",
    "files": 0, "files_done": 0,
    "indexed": 0, "skipped": 0, "failed": 0,
    "chunks": 0,
    "current_file": "",
    "elapsed": 0.0, "eta": 0.0,
    "recent": [],          # last 10 newly-indexed filenames
}


def _run_index_dirs(dirs: list):
    """Index a list of directories, pushing live progress into _index_job."""
    from logpose.indexer import FileIndexer
    indexer = FileIndexer(settings, _collection, _bm25)
    _index_job.update({
        "running": True, "message": "scanning…",
        "files": 0, "files_done": 0,
        "indexed": 0, "skipped": 0, "failed": 0,
        "chunks": 0, "current_file": "",
        "elapsed": 0.0, "eta": 0.0, "recent": [],
    })

    # Accumulate across multiple dirs
    acc = {"files": 0, "indexed": 0, "skipped": 0, "failed": 0, "chunks": 0}

    def _on_progress(p: dict):
        """Called by indexer after each file."""
        acc["files"] = p["total"]          # total for current dir
        acc["indexed"] += max(0, p["indexed"] - acc.get("_prev_indexed", 0))
        acc["skipped"] += max(0, p["skipped"] - acc.get("_prev_skipped", 0))
        acc["failed"]  += max(0, p["failed"]  - acc.get("_prev_failed",  0))
        acc["chunks"]  += max(0, p["chunks"]  - acc.get("_prev_chunks",  0))
        acc["_prev_indexed"] = p["indexed"]
        acc["_prev_skipped"] = p["skipped"]
        acc["_prev_failed"]  = p["failed"]
        acc["_prev_chunks"]  = p["chunks"]

        done = p["files_done"]
        total = p["total"]
        elapsed = p["elapsed"]
        rate = done / elapsed if elapsed > 0 and done > 0 else 0
        eta = (total - done) / rate if rate > 0 and total > done else 0

        recent = list(_index_job.get("recent") or [])
        if p.get("new_file"):
            recent = ([p["new_file"]] + recent)[:12]

        _index_job.update({
            "running": True,
            "message": f"indexing {p['current_file']}…" if p["current_file"] else "scanning…",
            "files": total,
            "files_done": done,
            "indexed": p["indexed"],
            "skipped": p["skipped"],
            "failed": p["failed"],
            "chunks": p["chunks"],
            "current_file": p["current_file"],
            "elapsed": round(elapsed, 1),
            "eta": round(eta, 1),
            "recent": recent,
        })

    try:
        for corp_dir in dirs:
            p = Path(corp_dir)
            # Reset per-dir accumulators
            for k in ("_prev_indexed", "_prev_skipped", "_prev_failed", "_prev_chunks"):
                acc[k] = 0
            result = _run_async(indexer.index_directory(p, progress_fn=_on_progress))

        total_indexed = _index_job.get("indexed", 0)
        total_files   = _index_job.get("files", 0)
        total_failed  = _index_job.get("failed", 0)
        total_chunks  = _index_job.get("chunks", 0)
        elapsed       = _index_job.get("elapsed", 0)
        _index_job.update({
            "running": False,
            "message": (f"Done — {total_indexed} files indexed, "
                        f"{total_chunks} chunks, {total_files} total in {elapsed}s"),
            "eta": 0.0, "current_file": "",
        })
        logger.info("Index job complete: %s", _index_job["message"])
    except Exception as exc:
        _index_job.update({"running": False, "message": f"Error: {exc}"})
        logger.exception("Index job failed")


def _run_index():
    _run_index_dirs(settings.corpus_dirs + _extra_corpus_dirs)


@app.route("/api/index", methods=["POST", "OPTIONS"])
def api_index():
    if request.method == "OPTIONS":
        return "", 204
    if _index_job["running"]:
        return jsonify({"ok": False, "running": True, "message": "Index job already running."}), 409
    t = threading.Thread(target=_run_index, daemon=True)
    t.start()
    return jsonify({"ok": True, "running": True, "message": "Index job started."})


@app.route("/api/index-status")
def api_index_status():
    return jsonify(_index_job)


# ── /api/add-corpus  (add a local folder path at runtime) ─────────────────────
@app.route("/api/add-corpus", methods=["POST", "OPTIONS"])
def api_add_corpus():
    if request.method == "OPTIONS":
        return "", 204
    data = request.get_json(silent=True) or {}
    path_str = (data.get("path") or "").strip()
    if not path_str:
        return jsonify({"ok": False, "error": "path required"}), 400
    p = Path(path_str).expanduser().resolve()
    if not p.exists():
        return jsonify({"ok": False, "error": f"Path not found: {p}"}), 404
    if not p.is_dir():
        # single file — add parent dir, index just the file
        target = p.parent
    else:
        target = p
    all_dirs = settings.corpus_dirs + _extra_corpus_dirs
    if target not in all_dirs:
        _extra_corpus_dirs.append(target)
    if _index_job["running"]:
        return jsonify({"ok": False, "running": True, "message": "Index job already running."}), 409
    t = threading.Thread(target=_run_index_dirs, args=([target],), daemon=True)
    t.start()
    return jsonify({"ok": True, "running": True, "path": str(target), "message": f"Indexing {target.name}…"})


# ── /api/upload  (browser file / folder upload) ───────────────────────────────
@app.route("/api/upload", methods=["POST", "OPTIONS"])
def api_upload():
    if request.method == "OPTIONS":
        return "", 204
    files = request.files.getlist("files")
    if not files:
        return jsonify({"ok": False, "error": "no files provided"}), 400
    saved = 0
    for f in files:
        if not f.filename:
            continue
        # webkitdirectory sends relative paths like "folder/sub/file.pdf"
        # preserve structure inside _uploads_dir
        safe_rel = Path(f.filename)
        dest = _uploads_dir / safe_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        f.save(dest)
        saved += 1
    if saved == 0:
        return jsonify({"ok": False, "error": "no files saved"}), 400
    if _uploads_dir not in settings.corpus_dirs and _uploads_dir not in _extra_corpus_dirs:
        _extra_corpus_dirs.append(_uploads_dir)
    if _index_job["running"]:
        return jsonify({"ok": False, "running": True, "message": "Index job already running."}), 409
    t = threading.Thread(target=_run_index_dirs, args=([_uploads_dir],), daemon=True)
    t.start()
    return jsonify({"ok": True, "running": True, "saved": saved, "message": f"Uploaded {saved} files, indexing…"})


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    port = int(os.environ.get("LOGPOSE_WEB_PORT", 7892))
    logger.info("Logpose UI  →  http://127.0.0.1:%d", port)
    logger.info("MCP server  →  aether-logpose  (separate process, System 1)")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
