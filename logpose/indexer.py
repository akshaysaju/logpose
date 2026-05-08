"""
indexer.py — Parse, chunk (with line tracking), embed, and persist.

Pipeline per file:
  1. Parse with the right :class:`FileParser` → :class:`ParsedDocument`
  2. Chunk into :class:`Chunk` objects that carry **line ranges**
  3. Translate line ranges into a friendly ``loc`` string (lines / pages /
     paragraphs / sheet+rows) using the parsed document's ``line_labels``
  4. Embed each batch of chunks via Ollama
  5. Upsert into ChromaDB (vector) + SQLite FTS5 (BM25)

Idempotency: mtime-based — unchanged files are skipped. Hashes are
preloaded in one ChromaDB query per indexing run.
"""

from __future__ import annotations

import ast
import asyncio
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
from .config import Settings
from .embeddings import get_embeddings_batch
from .parsers import ParsedDocument, registry as parser_registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunk type — carries line range alongside text
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A chunk of source text plus the 1-indexed line range it spans."""

    text: str
    line_start: int
    line_end: int


# ---------------------------------------------------------------------------
# Chunkers — every chunker returns List[Chunk] with line ranges
# ---------------------------------------------------------------------------


def _split_paragraphs_with_lines(text: str) -> List[Tuple[str, int, int]]:
    """Split ``text`` on blank lines, preserving 1-indexed line ranges.

    Returns list of (paragraph_text, line_start, line_end).
    """
    paragraphs: List[Tuple[str, int, int]] = []
    lines = text.splitlines()
    current: List[str] = []
    current_start: Optional[int] = None
    for i, line in enumerate(lines, start=1):
        if line.strip() == "":
            if current:
                paragraphs.append(("\n".join(current), current_start, i - 1))
                current = []
                current_start = None
            continue
        if current_start is None:
            current_start = i
        current.append(line)
    if current:
        paragraphs.append(("\n".join(current), current_start, len(lines)))
    return paragraphs


def chunk_text_lined(text: str, chunk_size: int, overlap: int) -> List[Chunk]:
    """Paragraph-preserving chunker that tracks 1-indexed line ranges."""
    paragraphs = _split_paragraphs_with_lines(text)
    if not paragraphs:
        return []

    chunks: List[Chunk] = []
    buf: List[Tuple[str, int, int]] = []
    buf_chars = 0

    def flush() -> None:
        nonlocal buf, buf_chars
        if not buf:
            return
        joined = "\n\n".join(p[0] for p in buf)
        if len(joined) >= 50:
            chunks.append(Chunk(text=joined, line_start=buf[0][1], line_end=buf[-1][2]))
        buf = []
        buf_chars = 0

    for ptext, l_start, l_end in paragraphs:
        if len(ptext) > chunk_size:
            flush()
            # Hard-slice oversized paragraph but keep its full line range on every slice.
            step = max(chunk_size - overlap, 1)
            i = 0
            while i < len(ptext):
                slice_text = ptext[i : i + chunk_size].strip()
                if len(slice_text) >= 50:
                    chunks.append(Chunk(text=slice_text, line_start=l_start, line_end=l_end))
                i += step
            continue

        if buf_chars + len(ptext) > chunk_size and buf:
            flush()
            # Carry the last paragraph forward as overlap context.
            buf.append((ptext, l_start, l_end))
            buf_chars = len(ptext)
        else:
            buf.append((ptext, l_start, l_end))
            buf_chars += len(ptext) + 2

    flush()
    return chunks


def chunk_python_lined(source: str, chunk_size: int, overlap: int) -> List[Chunk]:
    """AST-aware chunker for Python — one chunk per top-level function/class."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        logger.debug("AST parse failed — falling back to paragraph chunking")
        return chunk_text_lined(source, chunk_size, overlap)

    lines = source.splitlines()

    defs: List[Tuple[int, int]] = []  # (1-indexed start, 1-indexed end inclusive)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs.append((node.lineno, node.end_lineno or node.lineno))

    covered: set = set()
    for s, e in defs:
        for i in range(s, e + 1):
            covered.add(i)

    # Module-level lines: pick the longest consecutive run for the line range.
    mod_lines: List[Tuple[int, str]] = [(i, ln) for i, ln in enumerate(lines, start=1) if i not in covered]
    chunks: List[Chunk] = []

    if mod_lines:
        mod_text = "\n".join(ln for _, ln in mod_lines).strip()
        if len(mod_text) >= 50:
            chunks.append(
                Chunk(
                    text=mod_text,
                    line_start=mod_lines[0][0],
                    line_end=mod_lines[-1][0],
                )
            )

    for s, e in sorted(defs):
        block = "\n".join(lines[s - 1 : e]).strip()
        if not block or len(block) < 50:
            continue
        if len(block) > chunk_size:
            # Sub-chunk the oversized block; sub-chunks share the def's line range.
            for sub in chunk_text_lined(block, chunk_size, overlap):
                chunks.append(Chunk(text=sub.text, line_start=s, line_end=e))
        else:
            chunks.append(Chunk(text=block, line_start=s, line_end=e))

    return chunks if chunks else chunk_text_lined(source, chunk_size, overlap)


_JS_SPLIT_RE = re.compile(
    r"^(?:export\s+(?:default\s+)?)?"
    r"(?:async\s+)?"
    r"(?:function\s+\w+|class\s+\w+|(?:const|let|var)\s+\w+\s*=)",
    re.MULTILINE,
)


def chunk_js_ts_lined(source: str, chunk_size: int, overlap: int) -> List[Chunk]:
    """Regex-aware chunker for JS/TS that tracks line ranges."""
    boundaries = [m.start() for m in _JS_SPLIT_RE.finditer(source)]
    if not boundaries:
        return chunk_text_lined(source, chunk_size, overlap)

    def line_of_offset(off: int) -> int:
        return source.count("\n", 0, off) + 1

    chunks: List[Chunk] = []
    preamble = source[: boundaries[0]].strip()
    if len(preamble) >= 50:
        chunks.append(Chunk(text=preamble, line_start=1, line_end=line_of_offset(boundaries[0]) - 1))

    for idx, start in enumerate(boundaries):
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(source)
        block = source[start:end].strip()
        if not block or len(block) < 50:
            continue
        l_start = line_of_offset(start)
        l_end = line_of_offset(end - 1)
        if len(block) > chunk_size:
            for sub in chunk_text_lined(block, chunk_size, overlap):
                chunks.append(Chunk(text=sub.text, line_start=l_start, line_end=l_end))
        else:
            chunks.append(Chunk(text=block, line_start=l_start, line_end=l_end))

    return chunks if chunks else chunk_text_lined(source, chunk_size, overlap)


def chunk_pdf_lined(text: str, chunk_size: int, overlap: int) -> List[Chunk]:
    """Line-aware PDF chunker. Each line becomes the unit; chunks accumulate
    until ``chunk_size`` is reached; the last few lines carry forward as overlap.
    Tracks the 1-indexed line range that maps back to ``ParsedDocument.line_labels``.
    """
    lines = text.splitlines()
    if not lines:
        return []

    chunks: List[Chunk] = []
    buf: List[Tuple[int, str]] = []  # (1-indexed line number, content)
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        body = "\n".join(ln for _, ln in buf if ln.strip())
        if len(body) >= 50:
            chunks.append(Chunk(text=body, line_start=buf[0][0], line_end=buf[-1][0]))
        buf = []
        buf_len = 0

    for i, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        line_len = len(line) + 1
        if buf_len + line_len > chunk_size and buf:
            # Flush, then carry tail forward as overlap.
            tail: List[Tuple[int, str]] = []
            tail_len = 0
            for prev in reversed(buf):
                if tail_len + len(prev[1]) + 1 > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev[1]) + 1
            flush()
            buf = tail
            buf_len = tail_len
        buf.append((i, line))
        buf_len += line_len

    flush()
    return chunks


def chunk_for_extension(
    parsed: ParsedDocument,
    extension: str,
    chunk_size: int,
    overlap: int,
) -> List[Chunk]:
    """Pick the right chunker based on extension and parser output."""
    ext = extension.lower()
    text = parsed.text
    if ext == ".py":
        return chunk_python_lined(text, chunk_size, overlap)
    if ext in (".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"):
        return chunk_js_ts_lined(text, chunk_size, overlap)
    # PDFs / DOCX / XLSX / CSV / EPUB all benefit from line-mode chunking
    # because their parsers emit one logical row per line.
    if parsed.label_kind in ("page", "row", "paragraph", "chapter") or ext == ".pdf":
        return chunk_pdf_lined(text, chunk_size, overlap)
    return chunk_text_lined(text, chunk_size, overlap)


# ---------------------------------------------------------------------------
# Location helpers — translate line ranges into citations
# ---------------------------------------------------------------------------


def _location_metadata(chunk: Chunk, parsed: ParsedDocument) -> Dict[str, Any]:
    """Build location-related metadata fields for a chunk.

    Always populates ``line_start`` / ``line_end``. When the parser provided
    page/sheet/etc. labels, also derives ``label_start`` / ``label_end`` and
    a friendly ``loc`` citation string.
    """
    meta: Dict[str, Any] = {
        "line_start": chunk.line_start,
        "line_end": chunk.line_end,
    }

    if not parsed.line_labels:
        meta["loc"] = (
            f"line {chunk.line_start}"
            if chunk.line_start == chunk.line_end
            else f"lines {chunk.line_start}-{chunk.line_end}"
        )
        meta["label_kind"] = "line"
        return meta

    # Parsers index labels parallel to text.splitlines() — i.e. labels[i] is for line i+1.
    labels = parsed.line_labels
    s = max(0, min(chunk.line_start - 1, len(labels) - 1))
    e = max(0, min(chunk.line_end - 1, len(labels) - 1))
    span_labels = labels[s : e + 1]

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
        if not seen:
            loc = f"lines {chunk.line_start}-{chunk.line_end}"
        elif len(seen) == 1:
            loc = f"page {seen[0]}"
        else:
            loc = f"pages {seen[0]}-{seen[-1]}"
    elif kind == "row":
        loc = (
            f"row {seen[0]}" if len(seen) == 1
            else (f"rows {seen[0]}-{seen[-1]}" if seen else f"lines {chunk.line_start}-{chunk.line_end}")
        )
    elif kind == "paragraph":
        loc = (
            f"paragraph {seen[0]}" if len(seen) == 1
            else (f"paragraphs {seen[0]}-{seen[-1]}" if seen else f"lines {chunk.line_start}-{chunk.line_end}")
        )
    elif kind == "chapter":
        loc = (
            f"chapter {seen[0]}" if len(seen) == 1
            else (f"chapters {seen[0]}-{seen[-1]}" if seen else f"lines {chunk.line_start}-{chunk.line_end}")
        )
    else:
        loc = f"lines {chunk.line_start}-{chunk.line_end}"

    meta["loc"] = loc
    return meta


# ---------------------------------------------------------------------------
# YAML front-matter
# ---------------------------------------------------------------------------


def extract_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
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
    body = text[match.end():]
    return metadata, body


def _mtime_hash(path: Path) -> str:
    mtime = path.stat().st_mtime
    return hashlib.md5(f"{path}:{mtime}".encode()).hexdigest()  # noqa: S324


# ---------------------------------------------------------------------------
# FileIndexer
# ---------------------------------------------------------------------------


class FileIndexer:
    """Indexes files from a local directory into ChromaDB + BM25."""

    def __init__(
        self,
        config: Settings,
        collection: chromadb.Collection,
        bm25_index: Optional[BM25Index] = None,
        graph=None,  # Optional[DocumentGraph]
    ) -> None:
        self.config = config
        self.collection = collection
        self.bm25_index = bm25_index
        self.graph = graph
        self._hash_cache: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def preload_hashes(self, paths: List[Path]) -> None:
        if not paths:
            return
        str_paths = [str(p.resolve()) for p in paths]
        _BATCH = 500
        batches = [str_paths[i : i + _BATCH] for i in range(0, len(str_paths), _BATCH)]
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
                logger.warning(
                    "Batch hash preload failed (batch %d/%d): %s",
                    batch_num + 1, len(batches), exc,
                )
                continue
            for meta in (results.get("metadatas") or []):
                if isinstance(meta, dict):
                    fp = meta.get("file_path", "")
                    h = meta.get("mtime_hash", "")
                    if fp and h:
                        self._hash_cache[fp] = h

        logger.debug("Preloaded %d mtime hashes in %d batch(es)", len(self._hash_cache), len(batches))

    async def index_file(self, path: Path) -> int:
        path = path.resolve()
        ext = path.suffix.lower()
        current_hash = _mtime_hash(path)

        stored_hash = self._hash_cache.get(str(path))
        if stored_hash is None:
            unchanged = await asyncio.to_thread(self._is_unchanged, path, current_hash)
        else:
            unchanged = stored_hash == current_hash
        if unchanged:
            logger.debug("Skipping unchanged file: %s", path)
            return 0

        logger.info("Indexing: %s", path.name)

        parser = parser_registry.get(ext)
        try:
            parsed: ParsedDocument = await asyncio.to_thread(parser.parse, path)
        except ImportError as exc:
            logger.warning("Skipping %s — missing dependency: %s", path.name, exc)
            return 0

        # Markdown front-matter (operate on parsed.text)
        frontmatter: Dict[str, Any] = {}
        if ext in (".md", ".markdown"):
            frontmatter, body = extract_frontmatter(parsed.text)
            if body != parsed.text:
                # Strip equivalent number of leading lines from the labels array.
                stripped_lines = parsed.text.splitlines()[: len(parsed.text.splitlines()) - len(body.splitlines())]
                offset = len(stripped_lines)
                parsed = ParsedDocument(
                    text=body,
                    line_labels=parsed.line_labels[offset:] if parsed.line_labels else [],
                    label_kind=parsed.label_kind,
                )

        chunks = chunk_for_extension(parsed, ext, self.config.chunk_size, self.config.chunk_overlap)
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
            batch = chunks[batch_start : batch_start + batch_size]
            batch_texts = [c.text for c in batch]
            batch_embeddings = await get_embeddings_batch(batch_texts)

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

        # Replace any old chunks (file may now have fewer chunks than before).
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
        directory = directory.resolve()
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        supported = set(self.config.extensions)
        files = [
            p for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in supported
        ]

        total = len(files)
        indexed = 0
        skipped = 0
        failed = 0
        chunks_total = 0

        # Notify caller of total before we start so UI can show N/M immediately
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
                progress_fn({
                    "current_file": file_path.name,
                    "files_done": indexed + skipped + failed,
                    "total": total,
                    "indexed": indexed,
                    "skipped": skipped,
                    "failed": failed,
                    "chunks": chunks_total,
                    "elapsed": time.monotonic() - start_time,
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
        logger.info("Deleted all chunks for %s", path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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
        }
        meta.update(_location_metadata(chunk, parsed))

        _primitive = (str, int, float, bool)
        for key, value in frontmatter.items():
            safe_key = str(key)
            if isinstance(value, _primitive):
                meta[safe_key] = value
            elif value is None:
                meta[safe_key] = ""
            else:
                meta[safe_key] = str(value)
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
