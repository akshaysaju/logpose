"""
searcher.py — Hybrid search with optional HyDE, query expansion, and context expansion.

Four-stage retrieval pipeline:

  1. Recall       — semantic (Ollama → ChromaDB cosine) + BM25 (SQLite FTS5)
                    + optional HyDE passage embedding
                    + optional query-expansion alternatives
                    All run concurrently via asyncio.gather.

  2. Fusion       — Reciprocal Rank Fusion (k=60) merges all ranked lists.

  3. Rerank       — Optional cross-encoder (sentence-transformers) re-scores
                    the top-N with full attention over (query, chunk) pairs.

  4. Expansion    — For the final top results, if context_before / context_after
                    metadata is available, the retrieval context is widened to
                    include adjacent blocks (function neighbours, surrounding
                    paragraphs) for the LLM context window.

HyDE and query expansion are enabled via config:
  LOGPOSE_HYDE_ENABLED=true
  LOGPOSE_QUERY_EXPANSION_ENABLED=true
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

import chromadb

from .bm25_index import BM25Index
from .config import Settings
from .embeddings import get_embedding

logger = logging.getLogger(__name__)

_RRF_K = 60


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """A single ranked search hit."""

    file_path: str
    file_name: str
    chunk_text: str
    chunk_index: int
    score: float
    line_start: int = 0
    line_end: int = 0
    loc: str = ""
    # Rich chunk metadata
    chunk_type: str = ""
    symbol_name: str = ""
    heading_path: str = ""
    # Expanded context (adjacent blocks — not in the embedding, used for LLM)
    context_before: str = ""
    context_after: str = ""
    # Full metadata dict
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_context(self) -> str:
        """Chunk text expanded with adjacent context for LLM prompts."""
        parts = [p for p in [self.context_before, self.chunk_text, self.context_after] if p]
        return "\n\n".join(parts)


def _chunk_key(r: SearchResult) -> str:
    return f"{r.file_path}::chunk_{r.chunk_index}"


def _parse_chunk_index(chunk_id: str) -> int:
    try:
        return int(chunk_id.rsplit("::chunk_", 1)[-1])
    except (ValueError, IndexError):
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Cross-encoder reranker (lazy singleton)
# ─────────────────────────────────────────────────────────────────────────────

class _Reranker:
    """Lazy-loaded cross-encoder. Returns identity scores if unavailable."""

    _instance: Optional["_Reranker"] = None

    @classmethod
    def get(cls, model_name: str) -> "_Reranker":
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._tried = False

    def _load(self) -> None:
        if self._tried:
            return
        self._tried = True
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
            logger.info("Reranker ready: %s", self.model_name)
        except Exception as exc:
            logger.warning(
                "Reranker disabled — could not load %r: %s. "
                "Install with: pip install sentence-transformers",
                self.model_name, exc,
            )

    def score(self, query: str, passages: List[str]) -> Optional[List[float]]:
        self._load()
        if self._model is None or not passages:
            return None
        try:
            scores = self._model.predict([(query, p) for p in passages])
            return [float(s) for s in scores]
        except Exception as exc:
            logger.warning("Reranker scoring failed: %s", exc)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# FileSearcher
# ─────────────────────────────────────────────────────────────────────────────

class FileSearcher:
    """Hybrid (semantic + BM25 + HyDE + expansion + RRF + cross-encoder) search."""

    def __init__(
        self,
        config: Settings,
        collection: chromadb.Collection,
        bm25_index: Optional[BM25Index] = None,
    ) -> None:
        self.config = config
        self.collection = collection
        self.bm25_index = bm25_index

    # ── Public API ────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        n_results: int = 10,
        filter_extension: Optional[str] = None,
        filter_source: Optional[str] = None,
        min_score: float = 0.0,
        deduplicate: bool = True,
        rerank: Optional[bool] = None,
        bm25_query: Optional[str] = None,
        use_hyde: Optional[bool] = None,
        use_query_expansion: Optional[bool] = None,
    ) -> List[SearchResult]:
        """
        Search the corpus with hybrid retrieval + optional HyDE/expansion/reranking.

        Args:
            query:                Natural-language query.
            n_results:            Top-K to return.
            filter_extension:     Restrict to one extension, e.g. ".pdf".
            filter_source:        Restrict to a parent-directory name.
            min_score:            Discard results below this fused score.
            deduplicate:          Keep only the best-scoring chunk per file.
            rerank:               Override config enable_reranker.
            bm25_query:           Cleaned query for BM25 (defaults to query).
            use_hyde:             Override config hyde_enabled.
            use_query_expansion:  Override config query_expansion_enabled.
        """
        count = await asyncio.to_thread(self.collection.count)
        if count == 0:
            return []

        where = self._build_where(filter_extension, filter_source)
        fetch_n = min(n_results * 5, max(n_results, 50))
        fetch_n = min(fetch_n, count)

        bm25_q = bm25_query if bm25_query is not None else query
        do_hyde = self.config.hyde_enabled if use_hyde is None else use_hyde
        do_expand = self.config.query_expansion_enabled if use_query_expansion is None else use_query_expansion

        # Launch all recall tasks concurrently
        tasks = [
            asyncio.create_task(self._semantic_search(query, fetch_n, where)),
            asyncio.create_task(self._bm25_search(bm25_q, fetch_n)),
        ]
        if do_hyde:
            tasks.append(asyncio.create_task(self._hyde_search(query, fetch_n, where)))
        if do_expand:
            tasks.append(asyncio.create_task(self._expansion_search(query, fetch_n, where)))

        raw_lists = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten valid result lists
        all_lists: List[List[SearchResult]] = []
        for res in raw_lists:
            if isinstance(res, Exception):
                logger.warning("Search task failed: %s", res)
            else:
                all_lists.append(res)

        if not all_lists:
            return []

        # RRF merge across all result lists
        merged = self._rrf_merge_many(all_lists)

        # Post-merge filters (covers BM25-only hits with empty metadata)
        if filter_extension or filter_source:
            merged = self._apply_post_filters(merged, filter_extension, filter_source)
        if min_score > 0.0:
            merged = [r for r in merged if r.score >= min_score]

        # Rerank before deduplication so cross-encoder can promote under-ranked chunks
        use_reranker = self.config.enable_reranker if rerank is None else rerank
        if use_reranker and merged:
            merged = await self._rerank(query, merged)

        if deduplicate:
            merged = self._deduplicate(merged)

        return merged[:n_results]

    async def search_by_filename(self, pattern: str) -> List[SearchResult]:
        total = await asyncio.to_thread(self.collection.count)
        if total == 0:
            return []
        all_items = await asyncio.to_thread(
            self.collection.get,
            where={"chunk_index": {"$eq": 0}},
            include=["documents", "metadatas"],
        )
        results: List[SearchResult] = []
        pattern_lower = pattern.lower()
        docs = all_items.get("documents") or []
        metas = all_items.get("metadatas") or []
        for doc, meta in zip(docs, metas):
            if not isinstance(meta, dict):
                continue
            fname: str = meta.get("file_name", "")
            if pattern_lower in fname.lower():
                results.append(self._result_from_meta(doc or "", meta, score=1.0))
        return self._deduplicate(results)

    # ── Retrieval backends ────────────────────────────────────────────────────

    async def _semantic_search(
        self, query: str, fetch_n: int, where: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        query_embedding = await get_embedding(query)
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": fetch_n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        raw = await asyncio.to_thread(self.collection.query, **kwargs)
        return self._parse_chroma_results(raw)

    async def _bm25_search(self, query: str, fetch_n: int) -> List[SearchResult]:
        if self.bm25_index is None:
            return []
        bm25_results = await asyncio.to_thread(self.bm25_index.search, query, fetch_n)
        return [
            SearchResult(
                file_path=r.file_path,
                file_name=r.file_name,
                chunk_text=r.chunk_text,
                chunk_index=_parse_chunk_index(r.chunk_id),
                score=0.0,
                metadata={},
            )
            for r in bm25_results
        ]

    async def _hyde_search(
        self, query: str, fetch_n: int, where: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Embed a hypothetical answer passage instead of the raw query."""
        try:
            from .hyde import generate_hyde_passage
            passage = await generate_hyde_passage(query)
            if not passage:
                return []
            return await self._semantic_search(passage, fetch_n, where)
        except Exception as exc:
            logger.debug("HyDE search failed: %s", exc)
            return []

    async def _expansion_search(
        self, query: str, fetch_n: int, where: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Search with 2 query rewrites, return merged results."""
        try:
            from .hyde import expand_query
            alternatives = await expand_query(query)
            if not alternatives:
                return []
            sub_tasks = [
                self._semantic_search(alt, max(fetch_n // 2, 10), where)
                for alt in alternatives
            ]
            sub_results = await asyncio.gather(*sub_tasks, return_exceptions=True)
            merged: List[SearchResult] = []
            for res in sub_results:
                if not isinstance(res, Exception):
                    merged.extend(res)
            return merged
        except Exception as exc:
            logger.debug("Query expansion search failed: %s", exc)
            return []

    async def _rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        top_k = max(self.config.rerank_top_k, len(results) // 2)
        head = results[:top_k]
        tail = results[top_k:]
        reranker = _Reranker.get(self.config.reranker_model)
        scores = await asyncio.to_thread(reranker.score, query, [r.chunk_text for r in head])
        if scores is None:
            return results
        rescored = [replace(r, score=round(float(s), 6)) for r, s in zip(head, scores)]
        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored + tail

    # ── RRF ──────────────────────────────────────────────────────────────────

    def _rrf_merge(
        self,
        semantic: List[SearchResult],
        bm25: List[SearchResult],
    ) -> List[SearchResult]:
        return self._rrf_merge_many([semantic, bm25])

    def _rrf_merge_many(self, lists: List[List[SearchResult]]) -> List[SearchResult]:
        """RRF merge across N ranked lists."""
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, SearchResult] = {}
        for ranked_list in lists:
            for rank, r in enumerate(ranked_list):
                k = _chunk_key(r)
                rrf_scores[k] = rrf_scores.get(k, 0.0) + 1.0 / (_RRF_K + rank + 1)
                # Prefer the result that has full metadata (ChromaDB result)
                if k not in result_map or not result_map[k].metadata:
                    result_map[k] = r
        merged = [
            replace(result_map[k], score=round(score, 6))
            for k, score in rrf_scores.items()
        ]
        return sorted(merged, key=lambda x: x.score, reverse=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_where(
        self,
        extension: Optional[str],
        source: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        conditions: List[Dict[str, Any]] = []
        if extension:
            ext = extension if extension.startswith(".") else f".{extension}"
            conditions.append({"extension": {"$eq": ext}})
        if source:
            conditions.append({"source": {"$eq": source}})
        if not conditions:
            return None
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    def _result_from_meta(self, doc: str, meta: Dict[str, Any], score: float) -> SearchResult:
        return SearchResult(
            file_path=meta.get("file_path", ""),
            file_name=meta.get("file_name", ""),
            chunk_text=doc,
            chunk_index=int(meta.get("chunk_index", 0)),
            score=score,
            line_start=int(meta.get("line_start", 0) or 0),
            line_end=int(meta.get("line_end", 0) or 0),
            loc=str(meta.get("loc", "") or ""),
            chunk_type=str(meta.get("chunk_type", "") or ""),
            symbol_name=str(meta.get("symbol_name", "") or ""),
            heading_path=str(meta.get("heading_path", "") or ""),
            context_before=str(meta.get("context_before", "") or ""),
            context_after=str(meta.get("context_after", "") or ""),
            metadata=meta,
        )

    def _parse_chroma_results(self, raw: Dict[str, Any]) -> List[SearchResult]:
        results: List[SearchResult] = []
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            if not isinstance(meta, dict):
                continue
            results.append(
                self._result_from_meta(doc or "", meta, score=round(1.0 - float(dist), 4))
            )
        return results

    def _apply_post_filters(
        self,
        results: List[SearchResult],
        extension: Optional[str],
        source: Optional[str],
    ) -> List[SearchResult]:
        from pathlib import Path
        ext = None
        if extension:
            ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        out: List[SearchResult] = []
        for r in results:
            p = Path(r.file_path)
            if ext and p.suffix.lower() != ext:
                continue
            if source and (r.metadata.get("source") or p.parent.name) != source:
                continue
            out.append(r)
        return out

    def _deduplicate(self, results: List[SearchResult]) -> List[SearchResult]:
        best: Dict[str, SearchResult] = {}
        for r in results:
            key = r.file_path
            if key not in best or r.score > best[key].score:
                best[key] = r
        return sorted(best.values(), key=lambda x: x.score, reverse=True)
