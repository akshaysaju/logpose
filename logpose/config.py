"""
config.py — Pydantic-settings based configuration for the Logpose MCP server.
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
    """Central configuration for the Logpose document search MCP server."""

    model_config = SettingsConfigDict(
        env_prefix="LOGPOSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Corpus & Storage --------------------------------------------------
    corpus_dir: Path = Path("./corpus")
    "Primary root directory of files to index."

    extra_corpus_dirs: str = ""
    "Comma-separated list of additional directories to index alongside corpus_dir."

    chroma_dir: Path = Path("./chroma_db")
    "Directory where ChromaDB persists its data."

    collection_name: str = "local_files"
    "ChromaDB collection name."

    # --- Ollama / Embedding ------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    "Base URL of the running Ollama instance."

    embed_model: str = "nomic-embed-text-v2-moe"
    """Ollama embedding model. nomic-embed-text-v2-moe (768d, 8192 ctx) by default.
    Alternatives: bge-m3 (1024d, multilingual), mxbai-embed-large (1024d, 512 ctx)."""

    # --- Search defaults ---------------------------------------------------
    max_results: int = 10
    "Default top-k results returned by search."

    # --- Reranker ----------------------------------------------------------
    enable_reranker: bool = True
    "Run a cross-encoder over the fused top results for higher precision."

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    """HuggingFace model id for the cross-encoder.
    Defaults to ~90MB MiniLM (fast, English-only).
    For multilingual, switch to 'BAAI/bge-reranker-v2-m3' (~568MB)."""

    rerank_top_k: int = 30
    "Number of fused results passed to the cross-encoder. Beyond this, RRF order is kept."

    # --- Chunking ----------------------------------------------------------
    chunk_size: int = 800
    "Target character count per chunk."

    chunk_overlap: int = 150
    "Number of characters shared between consecutive chunks."

    # --- Supported file types ----------------------------------------------
    supported_extensions: str = (
        # Text & code
        ".md,.markdown,.txt,.log,.py,.js,.ts,.jsx,.tsx,.mjs,.cjs,"
        ".java,.go,.rs,.rb,.cs,.cpp,.c,.h,.hpp,.swift,.kt,.sh,.sql,"
        # Structured
        ".json,.yaml,.yml,.toml,.ini,.csv,.tsv,"
        # Office
        ".pdf,.docx,.xlsx,"
        # Web / books
        ".html,.htm,.epub,.rtf,"
        # Images (OCR)
        ".png,.jpg,.jpeg,.tiff,.tif,.bmp,.webp"
    )
    "Comma-separated file extensions that the indexer will process."

    # --- OCR ---------------------------------------------------------------
    enable_ocr: bool = True
    "Run Tesseract on scanned PDF pages and image files."

    ocr_min_chars_per_page: int = 50
    "If pdfplumber returns fewer chars on a page, treat it as scanned and OCR it."

    ocr_dpi: int = 200
    "DPI used when rendering PDF pages for OCR. 200 is a good speed/quality compromise."

    # --- Misc --------------------------------------------------------------
    embed_batch_size: int = 10
    "Number of chunks to embed per Ollama API call batch."

    auto_index_on_start: bool = False
    "If True, automatically index all corpus_dirs when the server starts."

    # Derived lists populated by _post_init — treat as read-only.
    _extensions_list: List[str] = []
    _corpus_dirs_list: List[Path] = []

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

        return self

    @property
    def extensions(self) -> List[str]:
        return self._extensions_list

    @property
    def corpus_dirs(self) -> List[Path]:
        return self._corpus_dirs_list

    @property
    def bm25_db_path(self) -> Path:
        return self.chroma_dir / "bm25.db"

    @property
    def ollama_embed_url(self) -> str:
        return f"{self.ollama_base_url.rstrip('/')}/api/embed"


# Module-level singleton — import this everywhere else.
settings = Settings()
