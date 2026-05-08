"""
config.py — Pydantic-settings based configuration for the Logpose web UI.
All values can be overridden via environment variables (LOGPOSE_ prefix) or a .env file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Central configuration for the Logpose document search server."""

    model_config = SettingsConfigDict(
        env_prefix="LOGPOSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Corpus & Storage ──────────────────────────────────────────────────────
    corpus_dir: Path = Path("./corpus")
    "Primary root directory of files to index."

    extra_corpus_dirs: str = ""
    "Comma-separated list of additional directories to index alongside corpus_dir."

    chroma_dir: Path = Path("./chroma_db")
    "Directory where ChromaDB persists its data."

    collection_name: str = "local_files"
    "ChromaDB collection name."

    # ── Ollama / Embedding ────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    "Base URL of the running Ollama instance."

    embed_model: str = "nomic-embed-text-v2-moe"
    """Ollama embedding model.
    nomic-embed-text-v2-moe  — 768d, 8192 ctx window (excellent for long code blocks)
    mxbai-embed-large        — 1024d, 512 ctx (higher quality for short prose)
    nomic-embed-text         — 768d, 8192 ctx (lighter alternative)
    """

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 800
    "Target character count for prose / generic chunks."

    chunk_overlap: int = 150
    "Chars shared between consecutive prose chunks."

    chunk_size_code: int = 1500
    "Target char count for code chunks (functions can be long)."

    chunk_size_data: int = 600
    "Target char count for structured data chunks (JSON, CSV, XLSX)."

    # ── File type support ─────────────────────────────────────────────────────
    supported_extensions: str = (
        # Text & code
        ".md,.markdown,.rst,.txt,.log,.py,.js,.ts,.jsx,.tsx,.mjs,.cjs,"
        ".java,.go,.rs,.rb,.cs,.cpp,.c,.h,.hpp,.swift,.kt,.sh,.sql,"
        # Structured data
        ".json,.yaml,.yml,.toml,.ini,.csv,.tsv,"
        # Office docs
        ".pdf,.docx,.xlsx,"
        # Web / books
        ".html,.htm,.epub,.rtf,"
        # Images (OCR)
        ".png,.jpg,.jpeg,.tiff,.tif,.bmp,.webp"
    )
    "Comma-separated file extensions that the indexer will process."

    # ── Exclusions ────────────────────────────────────────────────────────────
    exclude_dirs: str = (
        ".git,.svn,.hg,"
        "node_modules,bower_components,"
        "__pycache__,.pytest_cache,.mypy_cache,.ruff_cache,.hypothesis,"
        ".venv,venv,env,.env.d,"
        "dist,build,out,target,.next,.nuxt,.svelte-kit,"
        ".cargo,.gradle,.m2,"
        ".idea,.vscode,"
        "Pods,DerivedData,"
        "coverage,.nyc_output,"
        "__MACOSX,.Spotlight-V100,.Trashes"
    )
    "Comma-separated directory names to skip during indexing."

    exclude_file_patterns: str = (
        "*.pyc,*.pyo,*.pyd,*.so,*.dll,*.dylib,*.exe,*.a,*.o,"
        "*.egg-info,*.dist-info,"
        "package-lock.json,yarn.lock,pnpm-lock.yaml,Gemfile.lock,"
        "*.min.js,*.min.css,*.map,"
        "Thumbs.db,.DS_Store,.localized,"
        "*.tmp,*.swp,*.swo,*~,"
        "*.log.gz,*.log.1,*.log.2"
    )
    "Comma-separated glob patterns for files to skip during indexing."

    max_file_size_mb: float = 50.0
    "Skip files larger than this (MB) to avoid OOM during parsing."

    # ── Search ────────────────────────────────────────────────────────────────
    max_results: int = 10
    "Default top-k results returned by search."

    # ── Reranker ─────────────────────────────────────────────────────────────
    enable_reranker: bool = True
    "Run a cross-encoder over the fused top results for higher precision."

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    """HuggingFace cross-encoder model id.
    ms-marco-MiniLM-L-6-v2  — ~90 MB, fast, English-only (default)
    BAAI/bge-reranker-v2-m3 — ~568 MB, multilingual, higher quality
    """

    rerank_top_k: int = 30
    "Number of fused results passed to the cross-encoder."

    # ── HyDE & Query Expansion ────────────────────────────────────────────────
    hyde_enabled: bool = False
    """HyDE (Hypothetical Document Embeddings).
    Generates a hypothetical answer to the query, embeds that instead of the
    raw query. Adds ~1–3 s latency but dramatically improves recall on vague
    or vocabulary-mismatch queries.
    """

    query_expansion_enabled: bool = False
    """Generate 2 alternative phrasings of the query and RRF-merge all results.
    Adds ~1–2 s latency. Most useful when queries are short or ambiguous.
    """

    expand_model: str = "qwen3.5:0.8b"
    "Ollama model used for HyDE passage generation and query expansion."

    # ── OCR ───────────────────────────────────────────────────────────────────
    enable_ocr: bool = True
    "Run Tesseract on scanned PDF pages and image files."

    ocr_min_chars_per_page: int = 50
    "If pdfplumber returns fewer chars on a page, treat it as scanned and OCR it."

    ocr_dpi: int = 200
    "DPI used when rendering PDF pages for OCR."

    # ── Embedding performance ─────────────────────────────────────────────────
    embed_batch_size: int = 16
    "Number of chunks to embed per Ollama API call."

    # ── Misc ──────────────────────────────────────────────────────────────────
    auto_index_on_start: bool = False
    "If True, automatically index all corpus_dirs when the server starts."

    web_port: int = 7892
    "HTTP port for the web UI server."

    chat_model: str = "qwen3.5:4b"
    "Ollama model used for the Ask follow-up feature."

    # ── Derived — populated by _post_init, treat as read-only ─────────────────
    _extensions_list: List[str] = []
    _corpus_dirs_list: List[Path] = []
    _exclude_dirs_set: set = set()
    _exclude_patterns_list: List[str] = []

    @model_validator(mode="after")
    def _post_init(self) -> "Settings":
        self.corpus_dir = Path(str(self.corpus_dir)).expanduser().resolve()
        self.chroma_dir = Path(str(self.chroma_dir)).expanduser().resolve()

        self._extensions_list = [
            ext.strip().lower()
            for ext in self.supported_extensions.split(",")
            if ext.strip()
        ]

        dirs: List[Path] = [self.corpus_dir]
        for raw in self.extra_corpus_dirs.split(","):
            raw = raw.strip()
            if raw:
                dirs.append(Path(raw).expanduser().resolve())

        seen: set[Path] = set()
        self._corpus_dirs_list = []
        for d in dirs:
            if d not in seen:
                seen.add(d)
                self._corpus_dirs_list.append(d)

        self._exclude_dirs_set = {
            d.strip() for d in self.exclude_dirs.split(",") if d.strip()
        }
        self._exclude_patterns_list = [
            p.strip() for p in self.exclude_file_patterns.split(",") if p.strip()
        ]

        return self

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def extensions(self) -> List[str]:
        return self._extensions_list

    @property
    def corpus_dirs(self) -> List[Path]:
        return self._corpus_dirs_list

    @property
    def exclude_dirs_set(self) -> set:
        return self._exclude_dirs_set

    @property
    def exclude_patterns(self) -> List[str]:
        return self._exclude_patterns_list

    @property
    def bm25_db_path(self) -> Path:
        return self.chroma_dir / "bm25.db"

    @property
    def ollama_embed_url(self) -> str:
        return f"{self.ollama_base_url.rstrip('/')}/api/embed"

    @property
    def max_file_bytes(self) -> int:
        return int(self.max_file_size_mb * 1024 * 1024)


# Module-level singleton — import this everywhere.
settings = Settings()
