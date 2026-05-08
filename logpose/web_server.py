"""
logpose/web_server.py — Logpose standalone product server (production-grade).

Serves web/Logpose.html and exposes a REST API backed by ChromaDB + BM25
hybrid search, type-aware chunking, and optional HyDE / query expansion.

Port: 7892
Entry point: aether-logpose-ui  (pyproject.toml [project.scripts])

Usage:
    LOGPOSE_CORPUS_DIR=~/your/docs aether-logpose-ui
    # or:
    python -m logpose.web_server
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Generator

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from flask import Flask, Response, jsonify, request, send_file, stream_with_context  # noqa: E402
import chromadb  # noqa: E402

from logpose.config import settings  # noqa: E402
from logpose.bm25_index import BM25Index  # noqa: E402
from logpose.searcher import FileSearcher  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("logpose.web_server")

# ── ChromaDB + Searcher ───────────────────────────────────────────────────────
settings.chroma_dir.mkdir(parents=True, exist_ok=True)
_chroma_client = chromadb.PersistentClient(path=str(settings.chroma_dir))
_collection = _chroma_client.get_or_create_collection(
    name=settings.collection_name,
    metadata={"hnsw:space": "cosine"},
)
_bm25 = BM25Index(settings.bm25_db_path)
_searcher = FileSearcher(settings, _collection, _bm25)
logger.info(
    "ChromaDB collection '%s' ready (%d items)",
    settings.collection_name,
    _collection.count(),
)

# ── Extra runtime corpus dirs (added via /api/add-corpus or /api/upload) ──────
_extra_corpus_dirs: list[Path] = []
_uploads_dir: Path = settings.chroma_dir.parent / "uploads"
_uploads_dir.mkdir(parents=True, exist_ok=True)

# ── Single shared asyncio event loop for all async work ──────────────────────
# Flask uses a thread pool; all handlers submit coroutines to one background
# loop so asyncio primitives (Semaphore, httpx.AsyncClient) stay on one loop.
_async_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
threading.Thread(target=_async_loop.run_forever, daemon=True, name="async-loop").start()


def _run_async(coro, timeout: float = 120.0):
    future = asyncio.run_coroutine_threadsafe(coro, _async_loop)
    return future.result(timeout=timeout)


# ── Index job state (thread-safe via lock) ────────────────────────────────────
_index_lock = threading.Lock()
_index_job: dict = {
    "running": False, "message": "idle",
    "files": 0, "files_done": 0,
    "indexed": 0, "skipped": 0, "failed": 0,
    "chunks": 0, "current_file": "",
    "elapsed": 0.0, "eta": 0.0,
    "recent": [],
}


def _update_job(**kwargs) -> None:
    with _index_lock:
        _index_job.update(kwargs)


def _read_job() -> dict:
    with _index_lock:
        return dict(_index_job)


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.logger.setLevel(logging.WARNING)

_WEB_DIR = _ROOT / "web"
_HTML_PATH = _WEB_DIR / "Logpose.html"

# Upload limit: 500 MB
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


@app.after_request
def _add_headers(response):
    # CORS: allow local dev origins only
    origin = request.headers.get("Origin", "")
    if "localhost" in origin or "127.0.0.1" in origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    # Security headers (belt-and-suspenders for a local app)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
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
            "chunk_size_code": settings.chunk_size_code,
            "chunk_size_data": settings.chunk_size_data,
            "chunk_overlap": settings.chunk_overlap,
            "ocr_enabled": settings.enable_ocr,
            "hyde_enabled": settings.hyde_enabled,
            "query_expansion_enabled": settings.query_expansion_enabled,
            "expand_model": settings.expand_model,
        }
        if total_chunks == 0:
            return jsonify({
                **base, "total_chunks": 0, "total_files": 0,
                "file_types": {}, "chunk_types": {}, "sources": [], "last_indexed": None,
            })

        raw = _collection.get(include=["metadatas"], limit=min(total_chunks, 50_000))
        metas = raw.get("metadatas") or []

        file_per_ext: dict[str, int] = {}
        chunk_type_counts: dict[str, int] = {}
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
            ct = meta.get("chunk_type", "text")
            chunk_type_counts[ct] = chunk_type_counts.get(ct, 0) + 1
            if int(meta.get("chunk_index", 0)) == 0:
                ext = meta.get("extension", "unknown")
                file_per_ext[ext] = file_per_ext.get(ext, 0) + 1

        return jsonify({
            **base,
            "total_chunks": total_chunks,
            "total_files": len(file_paths),
            "file_types": file_per_ext,
            "chunk_types": chunk_type_counts,
            "sources": sorted(sources),
            "last_indexed": last_indexed,
        })
    except Exception as exc:
        logger.exception("api_stats failed")
        return jsonify({"error": str(exc)}), 500


# ── /api/search ───────────────────────────────────────────────────────────────

# Words that add no value to BM25 keyword matching
_BM25_NOISE = frozenset({
    "docs", "document", "documents", "file", "files", "folder", "folders",
    "regarding", "about", "find", "search", "show", "tell", "give", "list",
    "related", "get", "on", "in", "the", "a", "an", "any", "some",
    "what", "which", "where", "who", "how", "why", "of", "for", "with",
})

_COMPOUNDS = [
    (r"\bopen\s+ai\b", "OpenAI"),
    (r"\bmachine\s+learning\b", "machine_learning"),
    (r"\bdeep\s+learning\b", "deep_learning"),
    (r"\bnatural\s+language\s+processing\b", "NLP"),
    (r"\blarge\s+language\s+model\b", "LLM"),
    (r"\bneural\s+network\b", "neural_network"),
    (r"\breinforcement\s+learning\b", "reinforcement_learning"),
]


def _bm25_query(q: str) -> str:
    out = q
    for pattern, replacement in _COMPOUNDS:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    tokens = [t for t in out.split() if t.lower() not in _BM25_NOISE]
    return " ".join(tokens) if tokens else q.strip()


def _ext_to_kind(ext: str) -> str:
    return {
        ".pdf": "pdf", ".docx": "docx", ".doc": "docx",
        ".xlsx": "xlsx", ".xls": "xlsx",
        ".md": "md", ".markdown": "md", ".rst": "md",
        ".epub": "epub", ".html": "html", ".htm": "html",
        ".py": "code", ".js": "code", ".ts": "code",
        ".jsx": "code", ".tsx": "code", ".go": "code",
        ".rs": "code", ".java": "code", ".cpp": "code",
        ".c": "code", ".h": "code", ".swift": "code",
        ".kt": "code", ".cs": "code", ".rb": "code",
        ".sql": "code", ".sh": "code",
        ".json": "data", ".csv": "data", ".tsv": "data", ".xlsx": "data",
        ".yaml": "data", ".yml": "data", ".toml": "data",
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
        # Try first query token
        tokens = [t for t in q_lower.split() if len(t) > 3]
        for token in tokens:
            p = lower.find(token)
            if p != -1:
                pos = p
                break
    if pos == -1:
        return text[:300].replace("\n", " ")
    start = max(0, pos - 80)
    end = min(len(text), pos + len(q_lower) + 220)
    before = ("…" if start > 0 else "") + text[start:pos].replace("\n", " ")
    match = text[pos: pos + max(len(q_lower), 1)]
    after = text[pos + max(len(q_lower), 1): end].replace("\n", " ") + ("…" if end < len(text) else "")
    return f"{before}@@{match}@@{after}"


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "q parameter required"}), 400

    n = min(int(request.args.get("n", 10)), 20)
    types_raw = request.args.get("types", "")
    source = request.args.get("source", "") or None
    hyde = request.args.get("hyde", "").lower() in ("1", "true", "yes")
    expand = request.args.get("expand", "").lower() in ("1", "true", "yes")

    exts = [t.strip() for t in types_raw.split(",") if t.strip()] if types_raw else []
    q_bm25 = _bm25_query(q)
    logger.info("search q=%r  bm25_q=%r  hyde=%s  expand=%s", q, q_bm25, hyde, expand)

    t0 = time.monotonic()
    try:
        search_kwargs = dict(
            query=q,
            n_results=n,
            filter_source=source,
            bm25_query=q_bm25,
            use_hyde=hyde or None,        # None = fall back to config
            use_query_expansion=expand or None,
        )
        if exts:
            seen: dict = {}
            for ext in exts:
                rs = _run_async(_searcher.search(**{**search_kwargs, "filter_extension": ext}))
                for r in rs:
                    k = f"{r.file_path}::{r.chunk_index}"
                    if k not in seen:
                        seen[k] = r
            results = sorted(seen.values(), key=lambda r: -r.score)[:n]
        else:
            results = _run_async(_searcher.search(**search_kwargs))
    except Exception as exc:
        logger.exception("search failed q=%r", q)
        return jsonify({"error": str(exc)}), 500

    elapsed_ms = round((time.monotonic() - t0) * 1000)

    # Score noise cutoff
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
            # Rich metadata for UI
            "chunk_type": r.chunk_type,
            "symbol_name": r.symbol_name,
            "heading_path": r.heading_path,
        })

    return jsonify({
        "query": q,
        "results": out,
        "total": len(out),
        "duration_ms": elapsed_ms,
        "hyde_used": hyde,
        "expansion_used": expand,
    })


# ── /api/open-file ────────────────────────────────────────────────────────────
@app.route("/api/open-file", methods=["POST", "OPTIONS"])
def api_open_file():
    if request.method == "OPTIONS":
        return "", 204
    data = request.get_json(silent=True) or {}
    path_str = data.get("path", "").strip()
    if not path_str:
        return jsonify({"error": "path required"}), 400
    p = Path(path_str)
    if not p.exists():
        return jsonify({"error": f"File not found: {path_str}"}), 404
    # Only allow opening files within known corpus dirs (path traversal guard)
    all_corpus = settings.corpus_dirs + _extra_corpus_dirs + [_uploads_dir]
    resolved = p.resolve()
    allowed = any(
        str(resolved).startswith(str(d)) for d in all_corpus
    )
    if not allowed:
        return jsonify({"error": "Path not within an indexed corpus directory"}), 403
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        elif sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", str(p)])
        else:
            subprocess.Popen(["explorer", str(p)], shell=True)
        return jsonify({"ok": True})
    except Exception as exc:
        logger.exception("open-file failed path=%r", path_str)
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
                {
                    "text": doc,
                    "chunk_index": int(m.get("chunk_index", 0)),
                    "chunk_type": m.get("chunk_type", ""),
                    "symbol_name": m.get("symbol_name", ""),
                    "heading_path": m.get("heading_path", ""),
                    "loc": m.get("loc", ""),
                }
                for doc, m in zip(docs, metas)
                if doc
            ],
            key=lambda x: x["chunk_index"],
        )
        return jsonify({"chunks": chunks, "total": len(chunks)})
    except Exception as exc:
        logger.exception("file-content failed path=%r", path)
        return jsonify({"error": str(exc)}), 500


# ── /api/ask-followup (streaming) ────────────────────────────────────────────
@app.route("/api/ask-followup", methods=["POST", "OPTIONS"])
def api_ask_followup():
    if request.method == "OPTIONS":
        return "", 204
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    chunk_text = data.get("chunk_text", "").strip()
    file_path = data.get("path", "").strip()
    context_before = data.get("context_before", "").strip()
    context_after = data.get("context_after", "").strip()

    if not question:
        return jsonify({"error": "question required"}), 400

    # ── Build context: matched chunk (with its context window) + semantic search
    # within the file for more relevant chunks ────────────────────────────────
    primary_context = "\n\n".join(
        p for p in [context_before, chunk_text, context_after] if p
    )
    context_chunks: list[str] = [primary_context] if primary_context else []

    if file_path and len(context_chunks) < 6:
        try:
            # Semantic search *within this file* for the question
            file_results = _run_async(
                _searcher.search(
                    query=question,
                    n_results=5,
                    deduplicate=False,
                    rerank=False,
                ),
                timeout=20.0,
            )
            for r in file_results:
                if r.file_path == file_path:
                    ctx = r.full_context or r.chunk_text
                    if ctx and ctx not in context_chunks:
                        context_chunks.append(ctx)
                        if len(context_chunks) >= 6:
                            break
        except Exception:
            pass

        # Fallback: fetch top chunks from ChromaDB directly for the file
        if len(context_chunks) < 3:
            try:
                raw = _collection.get(
                    where={"file_path": file_path},
                    include=["documents", "metadatas"],
                    limit=20,
                )
                for doc, meta in zip(raw.get("documents") or [], raw.get("metadatas") or []):
                    if doc and doc not in context_chunks:
                        context_chunks.append(doc)
                        if len(context_chunks) >= 6:
                            break
            except Exception:
                pass

    # Cap total context at ~6000 chars to fit within LLM context
    context_text = "\n\n---\n\n".join(context_chunks[:6])
    if len(context_text) > 6000:
        context_text = context_text[:6000]

    fname = Path(file_path).name if file_path else "the document"
    prompt = (
        "/no_think\n"
        "Answer the question using ONLY the excerpts below. "
        "Be concise — 2-4 sentences. "
        "If the answer is not in the excerpts, say so clearly.\n\n"
        f"FILE: {fname}\n\n"
        f"<excerpts>\n{context_text}\n</excerpts>\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    ollama_url = settings.ollama_base_url.rstrip("/")
    chat_model = settings.chat_model

    # Streaming response
    def _stream() -> Generator[str, None, None]:
        try:
            import httpx
            with httpx.stream(
                "POST",
                f"{ollama_url}/api/chat",
                json={
                    "model": chat_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                    "options": {"num_predict": 512, "temperature": 0.1},
                },
                timeout=60.0,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    import json as _json
                    try:
                        chunk = _json.loads(line)
                    except Exception:
                        continue
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
        except Exception as exc:
            logger.exception("ask-followup stream failed")
            yield f"\n\n[Error: {exc}]"

    # Check if client accepts streaming
    accept = request.headers.get("Accept", "")
    if "text/event-stream" in accept or request.args.get("stream") == "1":
        def _sse_stream():
            buf = ""
            for token in _stream():
                # Strip Qwen3 think tags on-the-fly
                buf += token
                cleaned = re.sub(r"<think>.*?</think>", "", buf, flags=re.DOTALL)
                yield f"data: {cleaned[len(buf) - len(token):]}\n\n"
            yield "data: [DONE]\n\n"
        return Response(
            stream_with_context(_sse_stream()),
            content_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    # Non-streaming: collect full response then return JSON
    try:
        import httpx
        resp = httpx.post(
            f"{ollama_url}/api/chat",
            json={
                "model": chat_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"num_predict": 512, "temperature": 0.1},
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        raw_answer = resp.json()["message"]["content"].strip()
        answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()
        return jsonify({"answer": answer or raw_answer, "model": chat_model})
    except Exception as exc:
        logger.exception("ask-followup failed")
        return jsonify({"error": str(exc)}), 500


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

def _run_index_dirs(dirs: list):
    """Index a list of directories, pushing live progress into _index_job."""
    from logpose.indexer import FileIndexer
    indexer = FileIndexer(settings, _collection, _bm25)
    _update_job(
        running=True, message="scanning…",
        files=0, files_done=0,
        indexed=0, skipped=0, failed=0,
        chunks=0, current_file="",
        elapsed=0.0, eta=0.0, recent=[],
    )

    def _on_progress(p: dict):
        done = p["files_done"]
        total = p["total"]
        elapsed = p["elapsed"]
        rate = done / elapsed if elapsed > 0 and done > 0 else 0
        eta = (total - done) / rate if rate > 0 and total > done else 0

        with _index_lock:
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
            _run_async(
                indexer.index_directory(Path(corp_dir), progress_fn=_on_progress),
                timeout=3600.0,
            )

        with _index_lock:
            ti = _index_job.get("indexed", 0)
            tf = _index_job.get("files", 0)
            tc = _index_job.get("chunks", 0)
            te = _index_job.get("elapsed", 0)
        _update_job(
            running=False,
            message=f"Done — {ti} files indexed, {tc:,} chunks, {tf} total in {te}s",
            eta=0.0, current_file="",
        )
        logger.info("Index job complete: %s indexed, %s chunks", ti, tc)
    except Exception as exc:
        _update_job(running=False, message=f"Error: {exc}")
        logger.exception("Index job failed")


def _run_index():
    _run_index_dirs(settings.corpus_dirs + _extra_corpus_dirs)


@app.route("/api/index", methods=["POST", "OPTIONS"])
def api_index():
    if request.method == "OPTIONS":
        return "", 204
    job = _read_job()
    if job["running"]:
        return jsonify({"ok": False, "running": True, "message": "Index job already running."}), 409
    threading.Thread(target=_run_index, daemon=True, name="index-job").start()
    return jsonify({"ok": True, "running": True, "message": "Index job started."})


@app.route("/api/index-status")
def api_index_status():
    return jsonify(_read_job())


# ── /api/add-corpus ───────────────────────────────────────────────────────────
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
    target = p.parent if p.is_file() else p
    all_dirs = settings.corpus_dirs + _extra_corpus_dirs
    if target not in all_dirs:
        _extra_corpus_dirs.append(target)
    job = _read_job()
    if job["running"]:
        return jsonify({"ok": False, "running": True, "message": "Index job already running."}), 409
    threading.Thread(
        target=_run_index_dirs, args=([target],), daemon=True, name="index-job"
    ).start()
    return jsonify({"ok": True, "running": True, "path": str(target), "message": f"Indexing {target.name}…"})


# ── /api/upload ───────────────────────────────────────────────────────────────
_ALLOWED_UPLOAD_EXTS = set(settings.extensions)


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
        # Validate extension
        ext = Path(f.filename).suffix.lower()
        if ext not in _ALLOWED_UPLOAD_EXTS:
            logger.warning("Upload rejected — unsupported extension: %s", f.filename)
            continue
        # Preserve relative folder structure (webkitdirectory)
        safe_rel = Path(f.filename)
        dest = _uploads_dir / safe_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        f.save(dest)
        saved += 1
    if saved == 0:
        return jsonify({"ok": False, "error": "no valid files saved"}), 400
    if _uploads_dir not in settings.corpus_dirs and _uploads_dir not in _extra_corpus_dirs:
        _extra_corpus_dirs.append(_uploads_dir)
    job = _read_job()
    if job["running"]:
        return jsonify({"ok": False, "running": True, "message": "Index job already running."}), 409
    threading.Thread(
        target=_run_index_dirs, args=([_uploads_dir],), daemon=True, name="index-job"
    ).start()
    return jsonify({"ok": True, "running": True, "saved": saved, "message": f"Uploaded {saved} files, indexing…"})


# ── /api/health ───────────────────────────────────────────────────────────────
@app.route("/api/health")
def api_health():
    """Health check endpoint for monitoring / process supervisors."""
    try:
        count = _collection.count()
        return jsonify({
            "status": "ok",
            "chunks": count,
            "embed_model": settings.embed_model,
            "version": "2.0.0",
        })
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    port = settings.web_port
    logger.info("=" * 60)
    logger.info("Logpose UI  →  http://127.0.0.1:%d", port)
    logger.info("Corpus      →  %s", [str(d) for d in settings.corpus_dirs])
    logger.info("Embed model →  %s", settings.embed_model)
    logger.info("Chat model  →  %s", settings.chat_model)
    logger.info("HyDE        →  %s", settings.hyde_enabled)
    logger.info("Expansion   →  %s", settings.query_expansion_enabled)
    logger.info("Reranker    →  %s", settings.enable_reranker)
    logger.info("=" * 60)

    if settings.auto_index_on_start:
        logger.info("Auto-indexing on start (LOGPOSE_AUTO_INDEX_ON_START=true)")
        threading.Thread(target=_run_index, daemon=True, name="auto-index").start()

    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
