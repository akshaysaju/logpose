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

    # --- Metadata Enrichment -----------------------------------------------
    enrich_enabled: bool = False
    """LLM extracts entities, topics, and keywords from each chunk at index time.
    Enable: LOGPOSE_ENRICH_ENABLED=true
    Adds latency during indexing; enriched metadata improves search filtering and UI.
    """

    enrich_model: str = "qwen3.5:0.8b"
    "Ollama model for metadata enrichment. Fast model recommended (0.8b or similar)."

    enrich_batch_size: int = 5
    "Number of chunks to enrich per LLM call. Higher = fewer calls but larger prompt."

    # --- Document Relationship Graph ---------------------------------------
    graph_enabled: bool = False
    """Build a document relationship graph at index time.
    Tracks imports, citations, and shared-entity links between files.
    Enable: LOGPOSE_GRAPH_ENABLED=true
    At query time, related files' chunks are appended to results.
    """

    # --- Self-RAG Relevance Gating -----------------------------------------
    self_rag_enabled: bool = False
    """Self-RAG: score retrieved chunks for relevance before sending to LLM.
    Filters low-relevance chunks to reduce noise in the context window.
    Enable: LOGPOSE_SELF_RAG_ENABLED=true
    """

    self_rag_threshold: float = 0.6
    "Chunks scoring below this relevance score (0.0–1.0) are filtered out."

    self_rag_min_pass: int = 3
    "Always keep at least this many chunks even if all score below threshold."

    self_rag_model: str = "qwen3.5:0.8b"
    "Ollama model used for Self-RAG relevance scoring."

    # --- Agentic RAG -------------------------------------------------------
    agent_enabled: bool = False
    """Agentic RAG: LLM agent with tools (search, get_file, filter) iterates up to
    max_steps before producing a final answer. Best for multi-hop questions.
    Enable: LOGPOSE_AGENT_ENABLED=true
    """

    agent_model: str = "qwen3.5:4b"
    "Ollama model for the RAG agent loop. Needs sufficient reasoning ability."

    agent_max_steps: int = 5
    "Maximum tool-call iterations the agent may take before forced answer."

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

    @property
    def graph_db_path(self) -> str:
        """Absolute path to the SQLite document-graph database."""
        return str(self.chroma_dir / "graph.db")


# Module-level singleton — import this everywhere else.
settings = Settings()
