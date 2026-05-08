"""
graph.py — Document relationship graph backed by SQLite.

Tracks relationships between indexed files:
  - 'import'        Python/JS/Go/etc. import statements
  - 'citation'      Markdown [text](./local-link) references
  - 'shared_entity' Files sharing 2+ entity tokens (from enricher metadata)

Schema:
  document_relationships(src, dst, rel_type, weight)

At index time: extract_relationships(path, text, entities) → insert edges.
At query time: get_related(file_path) → list of related file paths.
At /api/graph: get_graph_data() → {nodes, edges} for D3 force graph.
"""

from __future__ import annotations

import ast
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS document_relationships (
    src      TEXT NOT NULL,
    dst      TEXT NOT NULL,
    rel_type TEXT NOT NULL,
    weight   REAL DEFAULT 1.0,
    PRIMARY KEY (src, dst, rel_type)
);
CREATE INDEX IF NOT EXISTS idx_dr_src ON document_relationships(src);
CREATE INDEX IF NOT EXISTS idx_dr_dst ON document_relationships(dst);
"""

_UPSERT_SQL = """
INSERT INTO document_relationships (src, dst, rel_type, weight)
VALUES (?, ?, ?, ?)
ON CONFLICT(src, dst, rel_type) DO UPDATE SET weight = weight + excluded.weight
"""


@dataclass
class Relationship:
    src: str
    dst: str
    rel_type: str  # 'import' | 'citation' | 'shared_entity'
    weight: float = 1.0


class DocumentGraph:
    """Thread-safe SQLite-backed document relationship graph."""

    def __init__(self, db_path: Union[str, Path]) -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(_CREATE_SQL)
            finally:
                conn.close()

    def add_relationships(self, rels: List[Relationship]) -> None:
        if not rels:
            return
        with self._lock:
            conn = self._connect()
            try:
                conn.executemany(
                    _UPSERT_SQL,
                    [(r.src, r.dst, r.rel_type, r.weight) for r in rels],
                )
                conn.commit()
            except Exception as exc:
                logger.warning("Graph insert failed: %s", exc)
            finally:
                conn.close()

    def upsert_file(self, file_path: str, rels: List[Relationship]) -> None:
        """Atomically replace all edges for file_path with rels in one transaction."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "DELETE FROM document_relationships WHERE src=? OR dst=?",
                    (file_path, file_path),
                )
                if rels:
                    conn.executemany(
                        _UPSERT_SQL,
                        [(r.src, r.dst, r.rel_type, r.weight) for r in rels],
                    )
                conn.commit()
            except Exception as exc:
                logger.warning("Graph upsert failed for %s: %s", file_path, exc)
            finally:
                conn.close()

    def clear_file(self, file_path: str) -> None:
        """Remove all edges where src OR dst is file_path."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "DELETE FROM document_relationships WHERE src=? OR dst=?",
                    (file_path, file_path),
                )
                conn.commit()
            finally:
                conn.close()

    def get_related(self, file_path: str, max_hops: int = 1) -> List[str]:
        """Return file paths related to file_path within max_hops."""
        visited: Set[str] = {file_path}
        frontier: Set[str] = {file_path}
        for _ in range(max_hops):
            if not frontier:
                break
            new_frontier: Set[str] = set()
            with self._lock:
                conn = self._connect()
                try:
                    placeholders = ",".join("?" * len(frontier))
                    rows = conn.execute(
                        f"SELECT src, dst FROM document_relationships "
                        f"WHERE src IN ({placeholders}) OR dst IN ({placeholders})",
                        list(frontier) + list(frontier),
                    ).fetchall()
                finally:
                    conn.close()
            for row in rows:
                for fp in (row["src"], row["dst"]):
                    if fp not in visited:
                        visited.add(fp)
                        new_frontier.add(fp)
            frontier = new_frontier
        return [fp for fp in visited if fp != file_path]

    def get_graph_data(self) -> Dict:
        """Return {nodes, edges} for D3 force-directed graph visualization."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT src, dst, rel_type, weight FROM document_relationships"
                ).fetchall()
            finally:
                conn.close()

        node_set: Dict[str, Dict] = {}
        edges = []
        for row in rows:
            src, dst = row["src"], row["dst"]
            for fp in (src, dst):
                if fp not in node_set:
                    node_set[fp] = {"id": fp, "name": Path(fp).name, "path": fp}
            edges.append({
                "source": src,
                "target": dst,
                "type": row["rel_type"],
                "weight": row["weight"],
            })

        return {"nodes": list(node_set.values()), "edges": edges}


# ── Relationship extraction helpers ───────────────────────────────────────────

_JS_IMPORT_RE = re.compile(
    r"""(?:import|require)\s*(?:.*?from\s*)?['"](\./[^'"]+)['"]""",
    re.MULTILINE,
)
_MD_LINK_RE = re.compile(r"\[.*?\]\((\./[^)]+)\)", re.MULTILINE)
_GO_IMPORT_RE = re.compile(r'"(\./[^"]+)"', re.MULTILINE)


def extract_relationships(
    file_path: str,
    text: str,
    extension: str,
    entities: str = "",
    all_file_paths: Optional[List[str]] = None,
) -> List[Relationship]:
    """
    Extract relationships from a file's text.

    Args:
        file_path: Absolute path of the file being indexed.
        text: Full text content of the file.
        extension: File extension e.g. '.py', '.md', '.js'
        entities: Comma-separated entity string from enricher (optional).
        all_file_paths: List of all indexed file paths for entity matching.

    Returns: List[Relationship] — may be empty.
    """
    rels: List[Relationship] = []
    base_dir = str(Path(file_path).parent)
    ext = extension.lower()

    if ext == ".py":
        rels.extend(_extract_python_imports(file_path, text, base_dir))

    elif ext in (".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"):
        for m in _JS_IMPORT_RE.finditer(text):
            resolved = _resolve_rel(base_dir, m.group(1))
            if resolved and resolved != file_path:
                rels.append(Relationship(file_path, resolved, "import"))

    elif ext == ".go":
        for m in _GO_IMPORT_RE.finditer(text):
            resolved = _resolve_rel(base_dir, m.group(1))
            if resolved and resolved != file_path:
                rels.append(Relationship(file_path, resolved, "import"))

    elif ext in (".md", ".markdown", ".rst"):
        for m in _MD_LINK_RE.finditer(text):
            resolved = _resolve_rel(base_dir, m.group(1))
            if resolved and resolved != file_path:
                rels.append(Relationship(file_path, resolved, "citation"))

    return rels


def _extract_python_imports(file_path: str, text: str, base_dir: str) -> List[Relationship]:
    """AST-based Python import extraction — relative imports only."""
    rels = []
    try:
        tree = ast.parse(text, filename=file_path)
    except SyntaxError:
        return rels
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
            parts = [base_dir]
            for _ in range(node.level - 1):
                parts.append("..")
            if node.module:
                parts.append(node.module.replace(".", "/"))
            candidate = str(Path(*parts).resolve()) + ".py"
            if candidate != file_path:
                rels.append(Relationship(file_path, candidate, "import", weight=1.5))
    return rels


def _resolve_rel(base_dir: str, rel_path: str) -> Optional[str]:
    """Resolve a relative path reference to an absolute path string, or None if unresolvable."""
    try:
        p = (Path(base_dir) / rel_path).resolve()
        # Try as-is, then with .py extension
        if p.exists():
            return str(p)
        py = p.with_suffix(".py")
        if py.exists():
            return str(py)
    except Exception:
        pass
    return None


def make_shared_entity_relationships(
    file_entity_map: Dict[str, Set[str]],
    min_shared: int = 2,
) -> List[Relationship]:
    """
    Build shared-entity edges between all pairs of files sharing >= min_shared entity tokens.

    Args:
        file_entity_map: {file_path → set of entity tokens}
        min_shared: Minimum number of shared tokens to create an edge.
    """
    rels = []
    file_list = list(file_entity_map.items())
    for i, (fp_a, ents_a) in enumerate(file_list):
        for fp_b, ents_b in file_list[i + 1:]:
            shared = len(ents_a & ents_b)
            if shared >= min_shared:
                rels.append(Relationship(fp_a, fp_b, "shared_entity", weight=float(shared)))
    return rels
