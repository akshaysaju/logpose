"""
searcher.py — Hybrid search with cross-encoder reranking.

Three-stage retrieval pipeline:

  1. **Recall**       — semantic (Ollama → ChromaDB cosine) and BM25
                        (SQLite FTS5) run *concurrently* via asyncio.gather.
  2. **Fusion**       — Reciprocal Rank Fusion merges the two ranked lists
                        with k=60: ``score(d) = Σ 1/(k + rank_i(d) + 1)``.
  3. **Rerank**       — Optional cross-encoder (sentence-transformers
                        ``ms-marco-MiniLM`` by default) re-scores the top-N
                        with full attention over (query, chunk) pairs.

The reranker materially improves precision on long natural-language queries.
It runs in a thread pool, falls back gracefully if the package or weights
are unavailable, and only re-orders the top ``rerank_top_k`` rather than
every fused result.
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

# Pre-compiled pattern to strip Qwen3 chain-of-thought blocks before parsing scores.
import re as _re
_THINK_RE = _re.compile(r"<think>.*?</think>", _re.DOTALL)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


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
    loc: str = ""  # Friendly citation: "page 3", "lines 42-78", "Sheet1!12-30"
    metadata: Dict[str, Any] = field(default_factory=dict)


def _chunk_key(r: SearchResult) -> str:
    return f"{r.file_path}::chunk_{r.chunk_index}"


def _parse_chunk_index(chunk_id: str) -> int:
    try:
        return int(chunk_id.rsplit("::chunk_", 1)[-1])
    except (ValueError, IndexError):
        return 0


# ---------------------------------------------------------------------------
# Cross-encoder reranker (lazy-loaded singleton)
# ---------------------------------------------------------------------------


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
            self._model = None

    def score(self, query: str, passages: List[str]) -> Optional[List[float]]:
        """Return one score per passage, or ``None`` if reranking is unavailable."""
        self._load()
        if self._model is None or not passages:
            return None
        try:
            # CrossEncoder.predict returns higher-better raw logit scores.
            scores = self._model.predict([(query, p) for p in passages])
            return [float(s) for s in scores]
        except Exception as exc:
            logger.warning("Reranker scoring failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# FileSearcher
# ---------------------------------------------------------------------------


class FileSearcher:
    """Hybrid (semantic + BM25 + RRF + optional cross-encoder) search."""

    def __init__(
        self,
        config: Settings,
        collection: chromadb.Collection,
        bm25_index: Optional[BM25Index] = None,
    ) -> None:
        self.config = config
        self.collection = collection
        self.bm25_index = bm25_index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        use_self_rag: Optional[bool] = None,
    ) -> List[SearchResult]:
        """Search the corpus with hybrid retrieval and optional reranking.

        Args:
            query:           Natural-language query.
            n_results:       Top-K to return.
            filter_extension: Restrict to a single extension (e.g. ``".pdf"``).
            filter_source:   Restrict to a parent-directory name.
            min_score:       Discard results below this fused score.
            deduplicate:     Keep only the best-scoring chunk per file.
            rerank:          Override config ``enable_reranker``. Set False
                             to skip cross-encoder pass even when configured.
        """
        count = await asyncio.to_thread(self.collection.count)
        if count == 0:
            logger.info("Collection is empty — no results.")
            return []

        where = self._build_where(filter_extension, filter_source)
        fetch_n = min(n_results * 5, max(n_results, 50))
        fetch_n = min(fetch_n, count)

        semantic_task = asyncio.create_task(self._semantic_search(query, fetch_n, where))
        bm25_q = bm25_query if bm25_query is not None else query
        bm25_task = asyncio.create_task(self._bm25_search(bm25_q, fetch_n))
        semantic_results, bm25_results = await asyncio.gather(semantic_task, bm25_task)

        merged = self._rrf_merge(semantic_results, bm25_results)

        # Apply post-merge filters so BM25-only hits (which have empty
        # metadata) are also subject to ``file_type`` / ``source``.
        if filter_extension or filter_source:
            merged = self._apply_post_filters(merged, filter_extension, filter_source)
        if min_score > 0.0:
            merged = [r for r in merged if r.score >= min_score]

        # Reranking happens BEFORE deduplication so the cross-encoder can
        # promote a strong chunk that the RRF stage initially under-ranked.
        use_reranker = self.config.enable_reranker if rerank is None else rerank
        if use_reranker and merged:
            merged = await self._rerank(query, merged)

        # Self-RAG relevance gating (after rerank, before dedup)
        do_self_rag = self.config.self_rag_enabled if use_self_rag is None else use_self_rag
        if do_self_rag and merged:
            merged = await self._self_rag_filter(query, merged)

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

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    async def _semantic_search(
        self, query: str, fetch_n: int, where: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        query_embedding = await get_embedding(query)
        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": fetch_n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where
        raw = await asyncio.to_thread(self.collection.query, **query_kwargs)
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

    async def _rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank the top ``rerank_top_k`` with a cross-encoder."""
        top_k = max(self.config.rerank_top_k, len(results) // 2)
        head = results[:top_k]
        tail = results[top_k:]

        reranker = _Reranker.get(self.config.reranker_model)
        scores = await asyncio.to_thread(reranker.score, query, [r.chunk_text for r in head])
        if scores is None:
            return results  # Fallback: keep RRF order.

        # Replace fused score with reranker score on the head.
        rescored = [replace(r, score=round(float(s), 6)) for r, s in zip(head, scores)]
        rescored.sort(key=lambda x: x.score, reverse=True)
        # Tail keeps fused scores (much smaller numerically) — stable suffix.
        return rescored + tail

    async def _self_rag_filter(
        self,
        query: str,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """Self-RAG: score top results for relevance to query, filter low-scoring chunks.

        Only scores the top 10 results (latency control). Always passes at least
        self_rag_min_pass chunks regardless of score to prevent empty results.
        """
        import httpx

        top_k = min(10, len(results))
        head = results[:top_k]
        tail = results[top_k:]

        url = f"{self.config.ollama_base_url.rstrip('/')}/api/chat"

        async def _score_chunk(client: httpx.AsyncClient, chunk_text: str) -> float:
            prompt = (
                "/no_think\n"
                "Rate how relevant this passage is to answering the query.\n"
                "Reply with ONLY a single decimal number between 0.0 and 1.0.\n"
                "0.0 = completely irrelevant, 1.0 = directly answers the query.\n\n"
                f"Query: {query}\n\n"
                f"Passage: {chunk_text[:500]}\n\n"
                "Score:"
            )
            try:
                resp = await client.post(url, json={
                    "model": self.config.self_rag_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"num_predict": 10, "temperature": 0.0},
                })
                resp.raise_for_status()
                raw = resp.json()["message"]["content"].strip()
                cleaned = _THINK_RE.sub("", raw).strip()
                m = _re.search(r"(\d+\.?\d*)", cleaned)
                if m:
                    return max(0.0, min(1.0, float(m.group(1))))
            except Exception as exc:
                logger.debug("Self-RAG scoring failed: %s", exc)
            return 1.0  # Fail-open: keep chunk if scoring fails

        # One shared client for all concurrent chunk-scoring requests.
        async with httpx.AsyncClient(timeout=15.0) as client:
            scores = await asyncio.gather(
                *[_score_chunk(client, r.chunk_text) for r in head]
            )

        threshold = self.config.self_rag_threshold
        min_pass = self.config.self_rag_min_pass

        # Sort by score descending so min_pass keeps the highest-scoring chunks.
        scored = sorted(zip(scores, head), key=lambda x: x[0], reverse=True)

        passed_keys: set[str] = set()
        for i, (score, result) in enumerate(scored):
            if score >= threshold or i < min_pass:
                passed_keys.add(_chunk_key(result))

        # Preserve original ranking order; update score to self-rag score.
        score_map = {_chunk_key(r): s for s, r in zip(scores, head)}
        ordered = [
            replace(r, score=round(score_map[_chunk_key(r)], 4))
            for r in head
            if _chunk_key(r) in passed_keys
        ]

        logger.debug(
            "Self-RAG: %d/%d chunks passed (threshold=%.2f)",
            len(ordered), top_k, threshold,
        )
        return ordered + tail

    # ------------------------------------------------------------------
    # RRF
    # ------------------------------------------------------------------

    def _rrf_merge(
        self,
        semantic: List[SearchResult],
        bm25: List[SearchResult],
    ) -> List[SearchResult]:
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, SearchResult] = {}
        for rank, r in enumerate(semantic):
            k = _chunk_key(r)
            rrf_scores[k] = rrf_scores.get(k, 0.0) + 1.0 / (_RRF_K + rank + 1)
            result_map[k] = r
        for rank, r in enumerate(bm25):
            k = _chunk_key(r)
            rrf_scores[k] = rrf_scores.get(k, 0.0) + 1.0 / (_RRF_K + rank + 1)
            if k not in result_map:
                result_map[k] = r
        merged = [
            replace(result_map[k], score=round(score, 6))
            for k, score in rrf_scores.items()
        ]
        return sorted(merged, key=lambda x: x.score, reverse=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _result_from_meta(
        self, doc: str, meta: Dict[str, Any], score: float
    ) -> SearchResult:
        return SearchResult(
            file_path=meta.get("file_path", ""),
            file_name=meta.get("file_name", ""),
            chunk_text=doc,
            chunk_index=int(meta.get("chunk_index", 0)),
            score=score,
            line_start=int(meta.get("line_start", 0) or 0),
            line_end=int(meta.get("line_end", 0) or 0),
            loc=str(meta.get("loc", "") or ""),
            metadata=meta,
        )

    def _parse_chroma_results(self, raw: Dict[str, Any]) -> List[SearchResult]:
        results: List[SearchResult] = []
        documents_list = raw.get("documents") or [[]]
        metadatas_list = raw.get("metadatas") or [[]]
        distances_list = raw.get("distances") or [[]]
        documents = documents_list[0] if documents_list else []
        metadatas = metadatas_list[0] if metadatas_list else []
        distances = distances_list[0] if distances_list else []
        for doc, meta, dist in zip(documents, metadatas, distances):
            if not isinstance(meta, dict):
                continue
            results.append(
                self._result_from_meta(
                    doc or "", meta, score=round(1.0 - float(dist), 4)
                )
            )
        return results

    def _apply_post_filters(
        self,
        results: List[SearchResult],
        extension: Optional[str],
        source: Optional[str],
    ) -> List[SearchResult]:
        """Drop results that don't match the requested filters.

        ChromaDB applies the filter on the semantic side, but BM25 does not —
        and BM25-only hits arrive with empty metadata. We re-derive extension
        from ``file_path`` and parent dir name from the path itself so the
        filter is uniform across both retrieval backends.
        """
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
