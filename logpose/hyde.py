"""
hyde.py — HyDE (Hypothetical Document Embeddings) and query expansion.

HyDE (Gao et al., 2022 — https://arxiv.org/abs/2212.10496):
  1. LLM generates a hypothetical passage that WOULD answer the query.
  2. That passage is embedded instead of (or alongside) the raw query.
  3. Hypothetical embeddings are systematically closer to real documents
     that answer the question than the query's own embedding, especially
     for vague / indirect / vocabulary-mismatch queries.

Query expansion:
  1. LLM rewrites the query in 2 alternative phrasings.
  2. Each phrasing is searched independently.
  3. All result lists are RRF-merged in the searcher.

Both techniques use qwen3.5:0.8b — 1 GB, fast on Apple Silicon.
Every call has a short timeout (15 s) and falls back silently to None / []
so the rest of the search pipeline is never blocked.

Enable via config:
  LOGPOSE_HYDE_ENABLED=true
  LOGPOSE_QUERY_EXPANSION_ENABLED=true
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import List, Optional

import httpx

from .config import settings

logger = logging.getLogger(__name__)

_TIMEOUT = 15.0
_MAX_TOKENS_HYDE = 180
_MAX_TOKENS_EXPAND = 120


async def _ollama_generate(prompt: str, max_tokens: int) -> Optional[str]:
    """Call Ollama /api/chat with the fast generation model. Returns None on any failure."""
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(url, json={
                "model": settings.expand_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.3,
                    "top_p": 0.9,
                },
            })
            resp.raise_for_status()
            raw = resp.json()["message"]["content"].strip()
            # Strip Qwen3-family <think>…</think> blocks
            cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            return cleaned or None
    except asyncio.TimeoutError:
        logger.debug("HyDE/expand LLM timed out")
        return None
    except Exception as exc:
        logger.debug("HyDE/expand LLM call failed: %s", exc)
        return None


async def generate_hyde_passage(query: str) -> Optional[str]:
    """
    Generate a hypothetical document passage that answers the query.

    The returned text shares vocabulary with real answer-containing documents,
    so its embedding is closer to relevant chunks than the query itself.

    Returns None if the model is unreachable or times out.
    """
    prompt = (
        "/no_think\n"
        "Write a concise technical passage (2–4 sentences) that DIRECTLY answers "
        "the question below. Write it as an excerpt from a real document — not an "
        "explanation, not a meta-answer. Just the facts.\n\n"
        f"Question: {query}\n\n"
        "Passage:"
    )
    passage = await _ollama_generate(prompt, _MAX_TOKENS_HYDE)
    if passage:
        logger.debug("HyDE passage (%d chars): %s…", len(passage), passage[:60])
    return passage


async def expand_query(query: str) -> List[str]:
    """
    Generate 2 alternative phrasings of the query.

    Returns a list of up to 2 alternative query strings (never includes the original).
    Returns [] if the model is unreachable or times out.
    """
    prompt = (
        "/no_think\n"
        "Rewrite the following search query in EXACTLY 2 alternative ways using "
        "different vocabulary but the same information need. "
        "Output ONLY the 2 rewrites — one per line — no numbers, no labels, no explanation.\n\n"
        f"Query: {query}"
    )
    result = await _ollama_generate(prompt, _MAX_TOKENS_EXPAND)
    if not result:
        return []

    alternatives = [
        line.strip().lstrip("1234567890.-) ").strip()
        for line in result.splitlines()
        if len(line.strip().lstrip("1234567890.-) ").strip()) > 5
    ]
    # Deduplicate and cap
    seen: set[str] = {query.lower()}
    out: List[str] = []
    for alt in alternatives:
        if alt.lower() not in seen:
            seen.add(alt.lower())
            out.append(alt)
        if len(out) >= 2:
            break

    logger.debug("Query expansion: %r → %r", query, out)
    return out
