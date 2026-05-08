"""
chunker.py — Semantic, type-aware chunking for all supported file formats.

Strategy per file type:

  Markdown/RST     → Heading-hierarchy-aware. Each section chunk carries its full
                     heading breadcrumb as a prefix ("# Guide > ## Install\\n\\n...").
  Python           → AST-based: module preamble, each class header, each method
                     (prefixed with class name), each standalone function.
  JS/TS/JSX/TSX    → Declaration-boundary regex (function/class/const/arrow).
  Go               → func/type/var/const declaration boundaries.
  Java/Kotlin      → class/interface/method boundaries.
  Rust             → fn/impl/struct/enum/trait boundaries.
  C/C++/Swift      → Function/class regex boundaries.
  SQL              → Statement-level (split at ';').
  Shell            → Function-definition boundaries.
  JSON             → Schema-first (top-level key inventory) + per-key value chunks.
  CSV/TSV          → Header-prefixed row batches (header on every chunk).
  XLSX             → Same, per sheet.
  PDF/DOCX/EPUB    → Paragraph-grouped with adjacent-paragraph context window.
  Plain text/Log   → Paragraph-grouped.

Every Chunk carries:
  text             — the content indexed and embedded
  line_start/end   — 1-indexed source line range
  chunk_type       — categorical label for UI and filtering
  symbol_name      — function/class/heading name (empty for prose)
  heading_path     — full breadcrumb for markdown ("# A > ## B")
  context_before   — preceding block text (stored in metadata, NOT indexed)
  context_after    — following block text (stored in metadata, NOT indexed)
  extra_meta       — language, sheet name, etc.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .parsers import ParsedDocument

# Minimum characters for a chunk to be worth indexing.
_MIN_CHUNK = 40

# Maximum chars stored for context_before / context_after.
_CTX_WINDOW = 600


# ─────────────────────────────────────────────────────────────────────────────
# Chunk dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A chunk of source text with location and rich context metadata."""

    text: str
    line_start: int
    line_end: int

    # Categorical type — drives UI badges and retrieval display
    chunk_type: str = "text"

    # Symbol this chunk belongs to (function / class / heading name)
    symbol_name: str = ""

    # Full heading breadcrumb for markdown: "# Guide > ## Installation"
    heading_path: str = ""

    # Surrounding context for retrieval expansion (stored in metadata, not indexed)
    context_before: str = ""
    context_after: str = ""

    # Extra key-value pairs (language, sheet_name, etc.)
    extra_meta: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _line_of(source: str, offset: int) -> int:
    """1-indexed line number for a character offset."""
    return source.count("\n", 0, offset) + 1


def _split_paragraphs(text: str) -> List[Tuple[str, int, int]]:
    """Split on blank lines → [(paragraph_text, line_start, line_end)] (1-indexed)."""
    paras: List[Tuple[str, int, int]] = []
    lines = text.splitlines()
    buf: List[str] = []
    start: Optional[int] = None
    for i, line in enumerate(lines, 1):
        if line.strip():
            if start is None:
                start = i
            buf.append(line)
        else:
            if buf and start is not None:
                paras.append(("\n".join(buf), start, i - 1))
            buf, start = [], None
    if buf and start is not None:
        paras.append(("\n".join(buf), start, len(lines)))
    return paras


def _attach_context(chunks: List[Chunk], window: int = _CTX_WINDOW) -> None:
    """Attach adjacent-chunk text as context_before / context_after in-place."""
    for i, chunk in enumerate(chunks):
        if i > 0:
            chunk.context_before = chunks[i - 1].text[-window:]
        if i < len(chunks) - 1:
            chunk.context_after = chunks[i + 1].text[:window]


def _merge_paras(
    paras: List[Tuple[str, int, int]],
    chunk_size: int,
    overlap: int,
    chunk_type: str = "text",
) -> List[Chunk]:
    """
    Greedily merge consecutive paragraphs into chunks up to chunk_size chars.
    Oversized single paragraphs are hard-sliced with overlap.
    Adjacent context is attached after all chunks are created.
    """
    if not paras:
        return []

    chunks: List[Chunk] = []
    buf: List[Tuple[str, int, int]] = []
    buf_len = 0

    def _flush() -> None:
        if not buf:
            return
        body = "\n\n".join(p[0] for p in buf)
        if len(body) >= _MIN_CHUNK:
            chunks.append(Chunk(
                text=body,
                line_start=buf[0][1],
                line_end=buf[-1][2],
                chunk_type=chunk_type,
            ))

    for ptext, l_start, l_end in paras:
        if len(ptext) > chunk_size:
            _flush()
            buf, buf_len = [], 0
            step = max(chunk_size - overlap, 100)
            j = 0
            while j < len(ptext):
                sl = ptext[j: j + chunk_size].strip()
                if len(sl) >= _MIN_CHUNK:
                    chunks.append(Chunk(
                        text=sl,
                        line_start=l_start,
                        line_end=l_end,
                        chunk_type=chunk_type,
                    ))
                j += step
            continue

        if buf_len + len(ptext) > chunk_size and buf:
            _flush()
            # Carry last para as overlap context
            tail = buf[-1:]
            buf = tail + [(ptext, l_start, l_end)]
            buf_len = sum(len(p[0]) for p in buf)
        else:
            buf.append((ptext, l_start, l_end))
            buf_len += len(ptext) + 2

    _flush()
    _attach_context(chunks)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Markdown — heading-hierarchy-aware
# ─────────────────────────────────────────────────────────────────────────────

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)


def _chunk_markdown(text: str, chunk_size: int, overlap: int) -> List[Chunk]:
    """Split at heading boundaries; prepend full heading breadcrumb to each chunk."""
    lines = text.splitlines()
    n = len(lines)

    headings: List[Tuple[int, int, str]] = []
    for i, line in enumerate(lines):
        m = _MD_HEADING_RE.match(line)
        if m:
            headings.append((i, len(m.group(1)), m.group(2).strip()))

    if not headings:
        return _merge_paras(_split_paragraphs(text), chunk_size, overlap, "markdown_prose")

    breadcrumb: List[Tuple[int, str]] = []
    chunks: List[Chunk] = []

    for idx, (line_i, level, htext) in enumerate(headings):
        while breadcrumb and breadcrumb[-1][0] >= level:
            breadcrumb.pop()
        breadcrumb.append((level, htext))
        heading_path = " > ".join(h for _, h in breadcrumb)

        body_start = line_i + 1
        body_end = headings[idx + 1][0] if idx + 1 < len(headings) else n
        body = "\n".join(lines[body_start:body_end]).strip()
        prefix = f"## {heading_path}\n\n"

        if not body:
            if len(heading_path) >= _MIN_CHUNK:
                chunks.append(Chunk(
                    text=heading_path,
                    line_start=line_i + 1,
                    line_end=line_i + 1,
                    chunk_type="markdown_heading",
                    symbol_name=htext,
                    heading_path=heading_path,
                ))
            continue

        full = prefix + body
        if len(full) <= chunk_size:
            chunks.append(Chunk(
                text=full,
                line_start=line_i + 1,
                line_end=body_end,
                chunk_type="markdown_section",
                symbol_name=htext,
                heading_path=heading_path,
            ))
        else:
            paras = _split_paragraphs(body)
            sub = _merge_paras(paras, chunk_size - len(prefix), overlap, "markdown_section")
            for sc in sub:
                sc.text = prefix + sc.text
                sc.symbol_name = htext
                sc.heading_path = heading_path
                sc.line_start = line_i + sc.line_start
                sc.line_end = line_i + sc.line_end
                chunks.append(sc)

    _attach_context(chunks)
    return chunks or _merge_paras(_split_paragraphs(text), chunk_size, overlap, "markdown_prose")


# ─────────────────────────────────────────────────────────────────────────────
# Python — AST-based with class + method grouping
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_python(source: str, chunk_size: int, overlap: int) -> List[Chunk]:
    """
    AST-based chunker:
    - Module preamble (imports + module docstring) as one chunk
    - Each class: header chunk + one chunk per method (prefixed with class name)
    - Each top-level function: one chunk
    Oversized blocks are sub-chunked by paragraph.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _merge_paras(_split_paragraphs(source), chunk_size, overlap, "code")

    lines = source.splitlines()
    chunks: List[Chunk] = []

    top_defs = [
        node for node in ast.iter_child_nodes(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]

    # Module preamble
    preamble_end = top_defs[0].lineno - 1 if top_defs else len(lines)
    preamble = "\n".join(lines[:preamble_end]).strip()
    if len(preamble) >= _MIN_CHUNK:
        chunks.append(Chunk(
            text=preamble,
            line_start=1,
            line_end=preamble_end,
            chunk_type="code_module",
            symbol_name="<module>",
            extra_meta={"language": "python"},
        ))

    for node in top_defs:
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            class_start = node.lineno
            class_end = node.end_lineno or node.lineno

            method_nodes = [
                n for n in ast.iter_child_nodes(node)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            header_end = method_nodes[0].lineno - 1 if method_nodes else class_end
            class_header = "\n".join(lines[class_start - 1: header_end]).strip()

            if len(class_header) >= _MIN_CHUNK:
                chunks.append(Chunk(
                    text=class_header,
                    line_start=class_start,
                    line_end=header_end,
                    chunk_type="code_class",
                    symbol_name=class_name,
                    extra_meta={"language": "python"},
                ))

            for method in method_nodes:
                m_start = method.lineno
                m_end = method.end_lineno or method.lineno
                method_text = "\n".join(lines[m_start - 1: m_end]).strip()
                if len(method_text) < _MIN_CHUNK:
                    continue
                symbol = f"{class_name}.{method.name}"
                # Prefix with class context so embeddings capture the relationship
                prefixed = f"# class {class_name}\n{method_text}"

                if len(prefixed) <= chunk_size:
                    chunks.append(Chunk(
                        text=prefixed,
                        line_start=m_start,
                        line_end=m_end,
                        chunk_type="code_method",
                        symbol_name=symbol,
                        extra_meta={"language": "python", "class": class_name},
                    ))
                else:
                    for sc in _merge_paras(
                        _split_paragraphs(method_text), chunk_size, overlap, "code_method"
                    ):
                        sc.text = f"# class {class_name}\n{sc.text}"
                        sc.symbol_name = symbol
                        sc.line_start = m_start + sc.line_start - 1
                        sc.line_end = m_start + sc.line_end - 1
                        sc.extra_meta = {"language": "python", "class": class_name}
                        chunks.append(sc)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            f_start = node.lineno
            f_end = node.end_lineno or node.lineno
            func_text = "\n".join(lines[f_start - 1: f_end]).strip()
            if len(func_text) < _MIN_CHUNK:
                continue

            if len(func_text) <= chunk_size:
                chunks.append(Chunk(
                    text=func_text,
                    line_start=f_start,
                    line_end=f_end,
                    chunk_type="code_function",
                    symbol_name=node.name,
                    extra_meta={"language": "python"},
                ))
            else:
                for sc in _merge_paras(
                    _split_paragraphs(func_text), chunk_size, overlap, "code_function"
                ):
                    sc.symbol_name = node.name
                    sc.line_start = f_start + sc.line_start - 1
                    sc.line_end = f_start + sc.line_end - 1
                    sc.extra_meta = {"language": "python"}
                    chunks.append(sc)

    _attach_context(chunks)
    return chunks or _merge_paras(_split_paragraphs(source), chunk_size, overlap, "code")


# ─────────────────────────────────────────────────────────────────────────────
# Generic regex-based code chunker
# ─────────────────────────────────────────────────────────────────────────────

# Language → (declaration regex, chunk_type prefix, language label)
_LANG_PATTERNS: Dict[str, Tuple[str, str, str]] = {
    # JS/TS/JSX/TSX
    ".js": (
        r"^(?:export\s+(?:default\s+)?)?(?:async\s+)?(?:function\s+\w+|class\s+\w+|"
        r"(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?(?:function|\(|[\w]+\s*=>))",
        "code_function", "javascript",
    ),
    ".ts": (
        r"^(?:export\s+(?:default\s+)?)?(?:async\s+)?(?:function\s+\w+|class\s+\w+|interface\s+\w+|"
        r"type\s+\w+\s*=|enum\s+\w+|(?:const|let|var)\s+\w+\s*(?::\s*\S+\s*)?=)",
        "code_function", "typescript",
    ),
    # Go
    ".go": (
        r"^(?:func\s+\(?[^)]*\)?\s*\w+|type\s+\w+|var\s+\w+|const\s+\w+|package\s+\w+)",
        "code_function", "go",
    ),
    # Java
    ".java": (
        r"^(?:\s*(?:public|private|protected|static|abstract|final|synchronized|native|"
        r"strictfp|\@\w+)\s+)*(?:class|interface|enum|record|void|[\w<>\[\]]+)\s+\w+\s*"
        r"(?:\(|<|extends|implements|\{)",
        "code_function", "java",
    ),
    # Kotlin
    ".kt": (
        r"^(?:(?:public|private|protected|internal|open|abstract|final|override|"
        r"companion|data|sealed|inline|suspend)\s+)*(?:fun|class|interface|object|"
        r"enum\s+class|data\s+class|sealed\s+class)\s+\w+",
        "code_function", "kotlin",
    ),
    # Rust
    ".rs": (
        r"^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:fn|impl|struct|enum|trait|mod|"
        r"const|static|type)\s+\w+",
        "code_function", "rust",
    ),
    # C
    ".c": (
        r"^[\w\*\s]+\s+\w+\s*\([^;]*\)\s*\{",
        "code_function", "c",
    ),
    ".h": (
        r"^(?:typedef\s+)?(?:struct|enum|union|class)\s+\w+|^[\w\*\s]+\s+\w+\s*\([^;]*\)\s*[;\{]",
        "code_function", "c_header",
    ),
    ".cpp": (
        r"^(?:[\w\*\s:]+::)?[\w\*\s]+\s+\w+\s*\([^;]*\)\s*(?:const\s*)?\{",
        "code_function", "cpp",
    ),
    ".hpp": (
        r"^(?:template\s*<[^>]*>\s*)?(?:class|struct|[\w\*\s:]+)\s+\w+\s*(?:\(|:|\{)",
        "code_function", "cpp_header",
    ),
    # Swift
    ".swift": (
        r"^(?:(?:public|private|internal|fileprivate|open|final|override|static|class|"
        r"mutating|nonmutating|lazy|weak|unowned|required|convenience)\s+)*"
        r"(?:func|class|struct|enum|protocol|extension|init|var|let)\s+\w+",
        "code_function", "swift",
    ),
    # Ruby
    ".rb": (
        r"^(?:class|module|def)\s+\w+",
        "code_function", "ruby",
    ),
    # C#
    ".cs": (
        r"^(?:\s*(?:public|private|protected|internal|static|abstract|virtual|override|"
        r"sealed|async|partial|extern|new)\s+)*(?:class|interface|struct|enum|delegate|"
        r"record|void|[\w<>\[\]?]+)\s+\w+\s*(?:\(|<|:|\{)",
        "code_function", "csharp",
    ),
    # Shell
    ".sh": (
        r"^(?:function\s+\w+\s*\(\)|^\w+\s*\(\)\s*\{)",
        "code_function", "shell",
    ),
}
# Map extensions sharing the same pattern
_LANG_PATTERNS[".jsx"] = _LANG_PATTERNS[".js"]
_LANG_PATTERNS[".tsx"] = _LANG_PATTERNS[".ts"]
_LANG_PATTERNS[".mjs"] = _LANG_PATTERNS[".js"]
_LANG_PATTERNS[".cjs"] = _LANG_PATTERNS[".js"]


def _chunk_by_regex(
    source: str,
    pattern: str,
    chunk_type: str,
    language: str,
    chunk_size: int,
    overlap: int,
) -> List[Chunk]:
    """Generic regex-boundary chunker used for most non-Python code."""
    boundaries = [m.start() for m in re.finditer(pattern, source, re.MULTILINE)]
    if not boundaries:
        return _merge_paras(_split_paragraphs(source), chunk_size, overlap, chunk_type)

    chunks: List[Chunk] = []

    # Preamble before first declaration
    preamble = source[: boundaries[0]].strip()
    if len(preamble) >= _MIN_CHUNK:
        chunks.append(Chunk(
            text=preamble,
            line_start=1,
            line_end=_line_of(source, boundaries[0]) - 1,
            chunk_type=f"{chunk_type}_preamble",
            extra_meta={"language": language},
        ))

    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(source)
        block = source[start:end].strip()
        if len(block) < _MIN_CHUNK:
            continue
        l_start = _line_of(source, start)
        l_end = _line_of(source, end - 1)

        # Try to extract symbol name from first line
        first_line = block.splitlines()[0] if block else ""
        sym_match = re.search(r"\b(\w+)\s*[\(\{<]", first_line)
        symbol_name = sym_match.group(1) if sym_match else ""

        if len(block) <= chunk_size:
            chunks.append(Chunk(
                text=block,
                line_start=l_start,
                line_end=l_end,
                chunk_type=chunk_type,
                symbol_name=symbol_name,
                extra_meta={"language": language},
            ))
        else:
            for sc in _merge_paras(_split_paragraphs(block), chunk_size, overlap, chunk_type):
                sc.symbol_name = symbol_name
                sc.line_start = l_start + sc.line_start - 1
                sc.line_end = l_start + sc.line_end - 1
                sc.extra_meta = {"language": language}
                chunks.append(sc)

    _attach_context(chunks)
    return chunks or _merge_paras(_split_paragraphs(source), chunk_size, overlap, chunk_type)


# ─────────────────────────────────────────────────────────────────────────────
# SQL — statement-level splitting
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_sql(text: str, chunk_size: int, overlap: int) -> List[Chunk]:
    """Split SQL at statement boundaries (';'). DDL context attached as prefix."""
    raw_stmts = [s.strip() for s in text.split(";") if len(s.strip()) >= _MIN_CHUNK]
    if not raw_stmts:
        return _merge_paras(_split_paragraphs(text), chunk_size, overlap, "code_sql")

    # Accumulate DDL (CREATE/ALTER/DROP) as context prefix for DML statements
    ddl_context: List[str] = []
    chunks: List[Chunk] = []
    line = 1

    for stmt in raw_stmts:
        stmt_lines = stmt.count("\n") + 1
        is_ddl = bool(re.match(r"^\s*(?:CREATE|ALTER|DROP|COMMENT)\s", stmt, re.IGNORECASE))
        stmt_type = re.match(r"^\s*(\w+)", stmt, re.IGNORECASE)
        sym = stmt_type.group(1).upper() if stmt_type else "SQL"

        if is_ddl:
            ddl_context = ddl_context[-2:] + [stmt]  # keep last 2 DDL stmts as context

        if len(stmt) <= chunk_size:
            chunks.append(Chunk(
                text=stmt,
                line_start=line,
                line_end=line + stmt_lines - 1,
                chunk_type="code_sql",
                symbol_name=sym,
                context_before="\n\n".join(ddl_context[:-1]) if is_ddl else "\n\n".join(ddl_context),
            ))
        else:
            for sc in _merge_paras(_split_paragraphs(stmt), chunk_size, overlap, "code_sql"):
                sc.symbol_name = sym
                sc.line_start = line + sc.line_start - 1
                sc.line_end = line + sc.line_end - 1
                chunks.append(sc)

        line += stmt_lines + 1

    _attach_context(chunks)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# JSON — schema-first + per-key value chunks
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_json(text: str, chunk_size: int, overlap: int) -> List[Chunk]:
    """
    Schema-first chunking:
    1. Top-level key inventory with types → schema chunk
    2. Each top-level key's value → its own chunk (or sub-chunked if large)
    3. Arrays → schema + sampled items chunk
    Falls back to paragraph chunking for invalid JSON.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return _merge_paras(_split_paragraphs(text), chunk_size, overlap, "text")

    chunks: List[Chunk] = []

    def _type_label(v: Any) -> str:
        if isinstance(v, dict):
            return f"object({len(v)} keys)"
        if isinstance(v, list):
            return f"array({len(v)} items)"
        return type(v).__name__

    if isinstance(data, dict):
        # Schema overview chunk
        schema = {k: _type_label(v) for k, v in data.items()}
        schema_text = "JSON object keys:\n" + "\n".join(f"  {k}: {t}" for k, t in schema.items())
        if len(schema_text) >= _MIN_CHUNK:
            chunks.append(Chunk(
                text=schema_text,
                line_start=1,
                line_end=1,
                chunk_type="data_schema",
                symbol_name="<schema>",
            ))

        # Per-key value chunks
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                val_text = f'"{key}": {json.dumps(value, indent=2, ensure_ascii=False)}'
            else:
                val_text = f'"{key}": {json.dumps(value, ensure_ascii=False)}'

            prefix = f"Key: {key}\n"
            full = prefix + val_text

            if len(full) <= chunk_size:
                if len(full) >= _MIN_CHUNK:
                    chunks.append(Chunk(
                        text=full,
                        line_start=1,
                        line_end=1,
                        chunk_type="data_value",
                        symbol_name=key,
                    ))
            else:
                # Sub-chunk the value's text representation
                val_str = json.dumps(value, indent=2, ensure_ascii=False)
                for sc in _merge_paras(_split_paragraphs(val_str), chunk_size - len(prefix), overlap, "data_value"):
                    sc.text = prefix + sc.text
                    sc.symbol_name = key
                    chunks.append(sc)

    elif isinstance(data, list):
        item_count = len(data)
        sample = data[:5]
        sample_text = (
            f"JSON array — {item_count} items\n"
            f"First {min(5, item_count)} items:\n"
            + json.dumps(sample, indent=2, ensure_ascii=False)
        )
        if len(sample_text) >= _MIN_CHUNK:
            chunks.append(Chunk(
                text=sample_text,
                line_start=1,
                line_end=1,
                chunk_type="data_schema",
                symbol_name="<array>",
            ))
        # Remaining items in batches
        batch = 10
        for i in range(5, item_count, batch):
            sub = data[i: i + batch]
            sub_text = f"Items {i}–{i + len(sub) - 1}:\n" + json.dumps(sub, indent=2, ensure_ascii=False)
            if len(sub_text) >= _MIN_CHUNK:
                chunks.append(Chunk(
                    text=sub_text[:chunk_size],
                    line_start=1,
                    line_end=1,
                    chunk_type="data_value",
                    symbol_name=f"items_{i}",
                ))

    _attach_context(chunks)
    return chunks or _merge_paras(_split_paragraphs(text), chunk_size, overlap, "text")


# ─────────────────────────────────────────────────────────────────────────────
# CSV / TSV — header-prefixed row batches
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_csv_rows(text: str, chunk_size: int, rows_per_chunk: int = 30) -> List[Chunk]:
    """
    Always include the header row at the top of every chunk.
    Groups data rows into batches of rows_per_chunk.
    """
    lines = text.splitlines()
    if not lines:
        return []

    header = lines[0]
    data_lines = [l for l in lines[1:] if l.strip()]
    if not data_lines:
        return [Chunk(text=header, line_start=1, line_end=1, chunk_type="data_header")]

    chunks: List[Chunk] = []
    for i in range(0, len(data_lines), rows_per_chunk):
        batch = data_lines[i: i + rows_per_chunk]
        chunk_text = header + "\n" + "\n".join(batch)
        if len(chunk_text) < _MIN_CHUNK:
            continue
        # Trim oversized chunks
        if len(chunk_text) > chunk_size:
            chunk_text = chunk_text[:chunk_size]
        chunks.append(Chunk(
            text=chunk_text,
            line_start=i + 2,        # 1-indexed, row 1 is header
            line_end=i + 2 + len(batch) - 1,
            chunk_type="data_rows",
            symbol_name=f"rows {i + 2}–{i + 2 + len(batch) - 1}",
        ))

    _attach_context(chunks)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# XLSX — per-sheet header-prefixed row batches (operated on parsed text)
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_xlsx(text: str, chunk_size: int) -> List[Chunk]:
    """
    The XLSX parser emits lines like 'Sheet1!1', 'Sheet1!2' etc.
    We chunk per-sheet using _chunk_csv_rows on each sheet's text block.
    """
    # Re-use paragraph chunker; each sheet block is already separated by blank lines
    paras = _split_paragraphs(text)
    chunks: List[Chunk] = []
    for ptext, l_start, l_end in paras:
        sub = _chunk_csv_rows(ptext, chunk_size)
        for sc in sub:
            sc.line_start = l_start + sc.line_start - 1
            sc.line_end = l_start + sc.line_end - 1
            sc.chunk_type = "data_rows"
        chunks.extend(sub)
    _attach_context(chunks)
    return chunks or _merge_paras(paras, chunk_size, 0, "data_rows")


# ─────────────────────────────────────────────────────────────────────────────
# Main dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def chunk_document(
    parsed: ParsedDocument,
    extension: str,
    chunk_size: int = 800,
    overlap: int = 150,
    code_chunk_size: int = 1500,
    data_chunk_size: int = 500,
) -> List[Chunk]:
    """
    Dispatch to the correct chunker for the given file extension.

    Args:
        parsed:          Output of the file parser.
        extension:       File extension (e.g. '.py', '.md', '.pdf').
        chunk_size:      Target char count for prose / generic chunks.
        overlap:         Chars shared between consecutive prose chunks.
        code_chunk_size: Target char count for code chunks (functions can be long).
        data_chunk_size: Target char count for structured data (JSON, CSV, etc.)

    Returns:
        List of Chunk objects ready for embedding and storage.
    """
    ext = extension.lower()
    text = parsed.text

    # ── Markdown / RST ────────────────────────────────────────────────────────
    if ext in (".md", ".markdown", ".rst"):
        return _chunk_markdown(text, chunk_size, overlap)

    # ── Python ────────────────────────────────────────────────────────────────
    if ext == ".py":
        return _chunk_python(text, code_chunk_size, overlap)

    # ── JS/TS and other typed languages with regex patterns ───────────────────
    if ext in _LANG_PATTERNS:
        pattern, ctype, lang = _LANG_PATTERNS[ext]
        return _chunk_by_regex(text, pattern, ctype, lang, code_chunk_size, overlap)

    # ── SQL ───────────────────────────────────────────────────────────────────
    if ext == ".sql":
        return _chunk_sql(text, code_chunk_size, overlap)

    # ── JSON ──────────────────────────────────────────────────────────────────
    if ext == ".json":
        return _chunk_json(text, data_chunk_size, overlap)

    # ── CSV / TSV ─────────────────────────────────────────────────────────────
    if ext in (".csv", ".tsv"):
        return _chunk_csv_rows(text, data_chunk_size)

    # ── XLSX (parsed text has sheet+row labels) ───────────────────────────────
    if ext == ".xlsx":
        return _chunk_xlsx(text, data_chunk_size)

    # ── PDF / DOCX / EPUB / RTF — prose with page/paragraph labels ───────────
    if parsed.label_kind in ("page", "paragraph", "chapter", "row") or ext == ".pdf":
        paras = _split_paragraphs(text)
        chunks = _merge_paras(paras, chunk_size, overlap, "prose")
        for c in chunks:
            c.chunk_type = "prose"
        return chunks

    # ── Generic plain text / log / config / YAML / TOML ──────────────────────
    return _merge_paras(_split_paragraphs(text), chunk_size, overlap, "text")
