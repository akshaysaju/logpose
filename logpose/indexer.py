"""
indexer.py — Parse → chunk → embed → persist.

Pipeline per file:
  1. Exclusion check   — skip hidden dirs, node_modules, large binaries, etc.
  2. Parse             — right FileParser for the extension → ParsedDocument
  3. Chunk             — chunk_document() dispatches type-aware chunker
  4. Embed             — batch Ollama /api/embed calls
  5. Upsert            — ChromaDB (vector) + SQLite FTS5 (BM25)

Idempotency: mtime-hash based — unchanged files are skipped.
Hashes preloaded in batched ChromaDB queries before the main loop.

New vs original:
  - Uses chunker.chunk_document() instead of inline paragraph chunker
  - Stores richer metadata: chunk_type, symbol_name, heading_path,
    context_before, context_after (for retrieval expansion)
  - Exclusion patterns: .git, node_modules, __pycache__, *.pyc, etc.
  - File size guard: skip files > max_file_size_mb
  - Per-type chunk sizes from config
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import yaml

from .bm25_index import BM25Index
from .chunker import Chunk, chunk_document
from .config import Settings
from .embeddings import get_embeddings_batch
from .parsers import ParsedDocument, registry as parser_registry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Exclusion helpers
# ─────────────────────────────────────────────────────────────────────────────

def _should_skip(path: Path, config: Settings) -> Optional[str]:
    """
    Return a skip-reason string if this path should be excluded, or None.
    Checked before any parsing or I/O on the file.
    """
    # 1. Excluded directory component
    for part in path.parts:
        if part in config.exclude_dirs_set:
            return f"excluded dir '{part}'"

    # 2. Hidden directory (any component starts with '.' except the top-level dir)
    for part in path.parts[:-1]:  # Don't skip the file itself for being hidden
        if part.startswith(".") and part not in (".", ".."):
            # Allow .claude, .config etc. at the top level corpus dir only
            # but block .git, .venv etc. anywhere deeper
            if part in {".git", ".svn", ".hg", ".venv", ".env", ".pytest_cache",
                        ".mypy_cache", ".ruff_cache", ".hypothesis", ".next",
                        ".nuxt", ".cargo", ".gradle"}:
                return f"hidden dir '{part}'"

    # 3. File name patterns
    name = path.name
    for pattern in config.exclude_patterns:
        if fnmatch.fnmatch(name, pattern):
            return f"excluded pattern '{pattern}'"

    # 4. File size guard
    try:
        if path.stat().st_size > config.max_file_bytes:
            return f"file too large ({path.stat().st_size // 1024 // 1024} MB)"
    except OSError:
        return "stat failed"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# YAML front-matter
# ─────────────────────────────────────────────────────────────────────────────

def _extract_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """Parse YAML front-matter from a Markdown string."""
    pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    match = pattern.match(text)
    if not match:
        return {}, text
    try:
        metadata = yaml.safe_load(match.group(1)) or {}
        if not isinstance(metadata, dict):
            metadata = {}
    except yaml.YAMLError as exc:
        logger.warning("YAML front-matter parse error: %s", exc)
        metadata = {}
    return metadata, text[match.end():]


def _mtime_hash(path: Path) -> str:
    mtime = path.stat().st_mtime
    return hashlib.md5(f"{path}:{mtime}".encode()).hexdigest()  # noqa: S324


# ─────────────────────────────────────────────────────────────────────────────
# Location metadata  (translate line range → friendly citation)
# ─────────────────────────────────────────────────────────────────────────────

def _location_metadata(chunk: Chunk, parsed: ParsedDocument) -> Dict[str, Any]:
    """Build location-related metadata for a chunk."""
    meta: Dict[str, Any] = {
        "line_start": chunk.line_start,
        "line_end": chunk.line_end,
    }

    if not parsed.line_labels:
        meta["loc"] = (
            f"line {chunk.line_start}"
            if chunk.line_start == chunk.line_end
            else f"lines {chunk.line_start}–{chunk.line_end}"
        )
        meta["label_kind"] = "line"
        return meta

    labels = parsed.line_labels
    s = max(0, min(chunk.line_start - 1, len(labels) - 1))
    e = max(0, min(chunk.line_end - 1, len(labels) - 1))
    span_labels = labels[s: e + 1]
    seen: List[str] = []
    for lbl in span_labels:
        if lbl and lbl not in seen:
            seen.append(lbl)

    meta["label_kind"] = parsed.label_kind
    if seen:
        meta["label_start"] = seen[0]
        meta["label_end"] = seen[-1]

    kind = parsed.label_kind
    if kind == "page":
        meta["page_start"] = seen[0] if seen else ""
        meta["page_end"] = seen[-1] if seen else ""
        loc = (
            f"page {seen[0]}" if len(seen) == 1
            else (f"pages {seen[0]}–{seen[-1]}" if seen
                  else f"lines {chunk.line_start}–{chunk.line_end}")
        )
    elif kind == "row":
        loc = (
            f"row {seen[0]}" if len(seen) == 1
            else (f"rows {seen[0]}–{seen[-1]}" if seen
                  else f"lines {chunk.line_start}–{chunk.line_end}")
        )
    elif kind == "paragraph":
        loc = (
            f"paragraph {seen[0]}" if len(seen) == 1
            else (f"paragraphs {seen[0]}–{seen[-1]}" if seen
                  else f"lines {chunk.line_start}–{chunk.line_end}")
        )
    elif kind == "chapter":
        loc = (
            f"chapter {seen[0]}" if len(seen) == 1
            else (f"chapters {seen[0]}–{seen[-1]}" if seen
                  else f"lines {chunk.line_start}–{chunk.line_end}")
        )
    else:
        loc = f"lines {chunk.line_start}–{chunk.line_end}"

    meta["loc"] = loc
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# FileIndexer
# ─────────────────────────────────────────────────────────────────────────────

class FileIndexer:
    """Indexes files from local directories into ChromaDB + BM25."""

    def __init__(
        self,
        config: Settings,
        collection: chromadb.Collection,
        bm25_index: Optional[BM25Index] = None,
    ) -> None:
        self.config = config
        self.collection = collection
        self.bm25_index = bm25_index
        self._hash_cache: Dict[str, str] = {}

    # ── Public async API ──────────────────────────────────────────────────────

    async def preload_hashes(self, paths: List[Path]) -> None:
        """Pre-load mtime hashes from ChromaDB in batched queries."""
        if not paths:
            return
        str_paths = [str(p.resolve()) for p in paths]
        _BATCH = 500
        batches = [str_paths[i: i + _BATCH] for i in range(0, len(str_paths), _BATCH)]
        for batch_num, batch in enumerate(batches):
            try:
                results = await asyncio.to_thread(
                    self.collection.get,
                    where={
                        "$and": [
                            {"file_path": {"$in": batch}},
                            {"chunk_index": {"$eq": 0}},
                        ]
                    },
                    include=["metadatas"],
                )
            except Exception as exc:
                logger.warning("Hash preload failed (batch %d/%d): %s", batch_num + 1, len(batches), exc)
                continue
            for meta in results.get("metadatas") or []:
                if isinstance(meta, dict):
                    fp = meta.get("file_path", "")
                    h = meta.get("mtime_hash", "")
                    if fp and h:
                        self._hash_cache[fp] = h

        logger.debug("Preloaded %d mtime hashes in %d batch(es)", len(self._hash_cache), len(batches))

    async def index_file(self, path: Path) -> int:
        """Index one file. Returns number of new chunks (0 = skipped/unchanged)."""
        path = path.resolve()
        ext = path.suffix.lower()
        current_hash = _mtime_hash(path)

        # Check mtime cache
        stored = self._hash_cache.get(str(path))
        if stored is not None:
            if stored == current_hash:
                logger.debug("Skipping unchanged: %s", path)
                return 0
        else:
            unchanged = await asyncio.to_thread(self._is_unchanged, path, current_hash)
            if unchanged:
                logger.debug("Skipping unchanged: %s", path)
                return 0

        logger.info("Indexing: %s", path.name)

        # Parse
        parser = parser_registry.get(ext)
        try:
            parsed: ParsedDocument = await asyncio.to_thread(parser.parse, path)
        except ImportError as exc:
            logger.warning("Skipping %s — missing dep: %s", path.name, exc)
            return 0
        except Exception as exc:
            logger.error("Parse error %s: %s", path.name, exc)
            return 0

        # Strip Markdown front-matter
        frontmatter: Dict[str, Any] = {}
        if ext in (".md", ".markdown"):
            frontmatter, body = _extract_frontmatter(parsed.text)
            if body != parsed.text:
                offset = len(parsed.text.splitlines()) - len(body.splitlines())
                parsed = ParsedDocument(
                    text=body,
                    line_labels=parsed.line_labels[offset:] if parsed.line_labels else [],
                    label_kind=parsed.label_kind,
                )

        # Chunk — dispatch to type-aware chunker
        chunks = chunk_document(
            parsed=parsed,
            extension=ext,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            code_chunk_size=self.config.chunk_size_code,
            data_chunk_size=self.config.chunk_size_data,
        )
        if not chunks:
            logger.warning("No usable text in %s — skipping", path)
            return 0

        total_chunks = len(chunks)
        batch_size = self.config.embed_batch_size

        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for batch_start in range(0, total_chunks, batch_size):
            batch = chunks[batch_start: batch_start + batch_size]
            batch_texts = [c.text for c in batch]
            try:
                batch_embeddings = await get_embeddings_batch(batch_texts)
            except Exception as exc:
                logger.error("Embedding failed for %s batch %d: %s", path.name, batch_start, exc)
                continue

            for i, (chunk, embedding) in enumerate(zip(batch, batch_embeddings)):
                chunk_index = batch_start + i
                meta = self._build_metadata(
                    path=path,
                    chunk=chunk,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    parsed=parsed,
                    frontmatter=frontmatter,
                    mtime_hash=current_hash,
                )
                ids.append(f"{path}::chunk_{chunk_index}")
                embeddings.append(embedding)
                documents.append(chunk.text)
                metadatas.append(meta)

        if not ids:
            return 0

        # Replace old chunks for this file
        await asyncio.to_thread(self.collection.delete, where={"file_path": str(path)})
        await asyncio.to_thread(
            self.collection.upsert,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        if self.bm25_index is not None:
            bm25_chunks = list(zip(ids, documents))
            await asyncio.to_thread(
                self.bm25_index.upsert_file,
                str(path),
                path.name,
                bm25_chunks,
            )

        self._hash_cache[str(path)] = current_hash
        logger.info("Indexed %d chunks from %s", total_chunks, path.name)
        return total_chunks

    async def index_directory(self, directory: Path, progress_fn=None) -> Dict[str, Any]:
        """Recursively index all supported files in directory."""
        directory = directory.resolve()
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        supported = set(self.config.extensions)

        # Collect files, applying exclusion filters
        files: List[Path] = []
        skipped_dirs: set[str] = set()
        for p in directory.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in supported:
                continue
            reason = _should_skip(p, self.config)
            if reason:
                skipped_dirs.add(reason)
                continue
            files.append(p)

        if skipped_dirs:
            logger.info("Exclusions applied (%d unique reasons)", len(skipped_dirs))

        total = len(files)
        indexed = skipped = failed = chunks_total = 0

        if progress_fn:
            progress_fn({
                "current_file": "", "files_done": 0, "total": total,
                "indexed": 0, "skipped": 0, "failed": 0,
                "chunks": 0, "elapsed": 0.0, "new_file": None,
            })

        start_time = time.monotonic()
        await self.preload_hashes(files)

        for file_path in files:
            new_chunks = 0
            try:
                new_chunks = await self.index_file(file_path)
                if new_chunks == 0:
                    skipped += 1
                else:
                    indexed += 1
                    chunks_total += new_chunks
            except Exception as exc:
                logger.error("Failed to index %s: %s", file_path, exc, exc_info=True)
                failed += 1

            if progress_fn:
                done = indexed + skipped + failed
                elapsed = time.monotonic() - start_time
                rate = done / elapsed if elapsed > 0 and done > 0 else 0
                eta = (total - done) / rate if rate > 0 and total > done else 0
                progress_fn({
                    "current_file": file_path.name,
                    "files_done": done,
                    "total": total,
                    "indexed": indexed,
                    "skipped": skipped,
                    "failed": failed,
                    "chunks": chunks_total,
                    "elapsed": round(elapsed, 1),
                    "eta": round(eta, 1),
                    "new_file": file_path.name if new_chunks > 0 else None,
                })

        duration = time.monotonic() - start_time
        return {
            "files": total,
            "indexed": indexed,
            "skipped": skipped,
            "failed": failed,
            "chunks": chunks_total,
            "duration_seconds": round(duration, 1),
        }

    async def delete_file(self, path: Path) -> None:
        path = path.resolve()
        await asyncio.to_thread(self.collection.delete, where={"file_path": str(path)})
        if self.bm25_index is not None:
            await asyncio.to_thread(self.bm25_index.delete_file, str(path))
        self._hash_cache.pop(str(path), None)
        logger.info("Deleted chunks for %s", path)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_metadata(
        self,
        path: Path,
        chunk: Chunk,
        chunk_index: int,
        total_chunks: int,
        parsed: ParsedDocument,
        frontmatter: Dict[str, Any],
        mtime_hash: str,
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "file_path": str(path),
            "file_name": path.name,
            "extension": path.suffix.lower(),
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "modified_at": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "indexed_at": datetime.now(tz=timezone.utc).isoformat(),
            "source": path.parent.name,
            "mtime_hash": mtime_hash,
            # Rich chunk metadata
            "chunk_type": chunk.chunk_type,
            "symbol_name": chunk.symbol_name or "",
            "heading_path": chunk.heading_path or "",
            # Context window for retrieval expansion (not indexed, used at query time)
            "context_before": chunk.context_before or "",
            "context_after": chunk.context_after or "",
        }

        # Location metadata (page / line / row / paragraph citations)
        meta.update(_location_metadata(chunk, parsed))

        # Markdown front-matter fields
        _primitive = (str, int, float, bool)
        for key, value in frontmatter.items():
            safe_key = str(key)
            if isinstance(value, _primitive):
                meta[safe_key] = value
            elif value is None:
                meta[safe_key] = ""
            else:
                meta[safe_key] = str(value)

        # Extra meta from chunker (language, class name, etc.)
        for key, value in chunk.extra_meta.items():
            if isinstance(value, _primitive):
                meta[f"chunk_{key}"] = value

        return meta

    def _is_unchanged(self, path: Path, current_hash: str) -> bool:
        try:
            results = self.collection.get(
                where={
                    "$and": [
                        {"file_path": {"$eq": str(path)}},
                        {"chunk_index": {"$eq": 0}},
                    ]
                },
                limit=1,
                include=["metadatas"],
            )
            if results["metadatas"]:
                stored_hash = results["metadatas"][0].get("mtime_hash", "")
                return stored_hash == current_hash
        except Exception as exc:
            logger.warning("ChromaDB change-check failed for %s: %s", path.name, exc)
        return False
