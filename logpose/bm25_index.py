"""
bm25_index.py — SQLite FTS5 full-text index with native Okapi BM25 ranking.

SQLite's FTS5 extension implements Okapi BM25 natively via the built-in
bm25() function.  No external libraries are needed — sqlite3 ships with the
Python standard library.

Formula (what FTS5 computes internally):
  BM25(d, q) = Σ_t IDF(t) × TF(t,d) × (k1+1) / (TF(t,d) + k1×(1-b+b×|d|/avgdl))
  where k1=1.2, b=0.75 (SQLite defaults)

Key design choices:
  - One FTS5 virtual table: chunks_fts(chunk_id, file_path, file_name, chunk_text)
  - chunk_text is the only FTS-indexed column; others are UNINDEXED (stored,
    not tokenised) so we can retrieve them without a join.
  - Deletes by file use a rowid sub-query — FTS5 allows a full-scan WHERE
    on UNINDEXED columns inside a SELECT: DELETE ... WHERE rowid IN (SELECT ...)
  - WAL mode + NORMAL synchronous for safe concurrent reads.
  - Every public method is synchronous; callers wrap in asyncio.to_thread.
  - Query terms are double-quoted ("term") so FTS5 special characters
    (*, ^, -, etc.) cannot cause syntax errors.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Words to skip when building the FTS5 query from a natural-language string.
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "not",
    "this", "that", "these", "those", "it", "its", "i", "you", "he",
    "she", "we", "they", "what", "which", "who", "how", "when", "where",
    "if", "then", "than", "so", "up", "out", "about", "into", "also",
})


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BM25Result:
    """A single result from the FTS5 BM25 search."""

    chunk_id: str
    file_path: str
    file_name: str
    chunk_text: str
    bm25_score: float   # Negative — FTS5 convention; more negative = better match


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------


class BM25Index:
    """SQLite FTS5 full-text index with native Okapi BM25 ranking.

    Thread safety:
        Each public method opens and closes its own connection, so multiple
        asyncio.to_thread workers can call in parallel safely.  SQLite WAL
        mode permits concurrent reads + one writer without blocking.

    Schema::

        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id  UNINDEXED,
            file_path UNINDEXED,
            file_name UNINDEXED,
            chunk_text,
            tokenize = 'unicode61'
        )
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with recommended pragmas."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        """Create the FTS5 table if it doesn't already exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id  UNINDEXED,
                    file_path UNINDEXED,
                    file_name UNINDEXED,
                    chunk_text,
                    tokenize = 'unicode61'
                )
            """)
        logger.debug("BM25 FTS5 index ready at %s", self.db_path)

    # ------------------------------------------------------------------
    # Write operations  (call via asyncio.to_thread in async code)
    # ------------------------------------------------------------------

    def upsert_file(
        self,
        file_path: str,
        file_name: str,
        chunks: List[Tuple[str, str]],
    ) -> None:
        """Replace all FTS5 entries for *file_path* with *chunks*.

        Args:
            file_path: Absolute path of the file being indexed.
            file_name: Basename of the file.
            chunks:    List of ``(chunk_id, chunk_text)`` tuples.
                       chunk_id must match the ChromaDB id (``path::chunk_N``).
        """
        with self._connect() as conn:
            # Delete any previously indexed chunks for this file
            conn.execute(
                "DELETE FROM chunks_fts WHERE rowid IN "
                "(SELECT rowid FROM chunks_fts WHERE file_path = ?)",
                (file_path,),
            )
            if chunks:
                conn.executemany(
                    "INSERT INTO chunks_fts(chunk_id, file_path, file_name, chunk_text) "
                    "VALUES (?, ?, ?, ?)",
                    [(cid, file_path, file_name, text) for cid, text in chunks],
                )
        logger.debug("BM25: upserted %d chunks for %s", len(chunks), file_name)

    def delete_file(self, file_path: str) -> None:
        """Remove all FTS5 entries for *file_path*."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM chunks_fts WHERE rowid IN "
                "(SELECT rowid FROM chunks_fts WHERE file_path = ?)",
                (file_path,),
            )
        logger.debug("BM25: deleted chunks for %s", file_path)

    # ------------------------------------------------------------------
    # Read operations  (call via asyncio.to_thread in async code)
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 50) -> List[BM25Result]:
        """Return up to *limit* chunks ranked by Okapi BM25 score.

        Args:
            query: Natural-language or keyword query.  Special FTS5 characters
                   are neutralised by quoting each term.
            limit: Maximum number of results to return.

        Returns:
            List of :class:`BM25Result` sorted by BM25 score ascending
            (most negative = best match — FTS5 convention).  Empty list if
            no matches or the index is empty.
        """
        fts_query = self._build_fts_query(query)
        if not fts_query:
            return []

        with self._connect() as conn:
            try:
                rows = conn.execute(
                    """
                    SELECT chunk_id, file_path, file_name, chunk_text,
                           bm25(chunks_fts) AS bm25_score
                    FROM chunks_fts
                    WHERE chunks_fts MATCH ?
                    ORDER BY bm25_score          -- most negative = best
                    LIMIT ?
                    """,
                    (fts_query, limit),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                # Rare: malformed FTS5 query after sanitisation
                logger.warning("FTS5 query failed for %r: %s", query, exc)
                return []

        return [
            BM25Result(
                chunk_id=row["chunk_id"],
                file_path=row["file_path"],
                file_name=row["file_name"],
                chunk_text=row["chunk_text"],
                bm25_score=float(row["bm25_score"]),
            )
            for row in rows
        ]

    def count(self) -> int:
        """Return the total number of chunks in the index."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()
            return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_fts_query(self, query: str) -> str:
        """Convert a natural-language query into a safe FTS5 OR expression.

        Each token is wrapped in double quotes so FTS5 treats it as a literal
        phrase rather than an operator.  Stop words are filtered unless the
        query consists entirely of stop words (in which case we keep them).

        Examples::

            "search for authentication tokens" →  "authentication" OR "tokens"
            "def"                              →  "def"
            "how do I do X"                    →  "how" OR "how" (... deduped)
        """
        # Tokenise: word chars only (handles snake_case via underscore split
        # by unicode61 tokenizer, but we still want camelCase as one unit here)
        tokens = re.findall(r"\b\w{2,}\b", query)

        # Filter stop words; if nothing survives, use all tokens
        meaningful = [t for t in tokens if t.lower() not in _STOP_WORDS]
        terms = meaningful if meaningful else tokens

        if not terms:
            return ""

        # Deduplicate while preserving order (dict preserves insertion order)
        unique = list(dict.fromkeys(terms))

        # Quote each term: special chars inside quotes are literals in FTS5
        return " OR ".join(f'"{t.replace(chr(34), "")}"' for t in unique)
