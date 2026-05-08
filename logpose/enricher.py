"""
enricher.py — LLM-based metadata extraction for indexed chunks.

At index time, sends batches of chunks to a fast local LLM (qwen3.5:0.8b)
to extract:
  - entities: people, organizations, files, functions, classes, concepts
  - topics: 2-3 broad themes
  - keywords: 5-7 important terms

Results are stored in ChromaDB metadata as comma-separated strings:
  entities, topics, keywords

This enables:
  - Richer result display (badge pills in UI)
  - Metadata-based filtering (future)
  - Graph shared-entity detection (graph.py)

All failures are silent — enrichment never blocks indexing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import httpx

from .config import settings

logger = logging.getLogger(__name__)

_TIMEOUT = 20.0  # slightly longer than embedding calls — batch prompts are bigger
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_LINE_RE = re.compile(
    r"\[(\d+)\]\s+entities:\s*(.*?)\s*\|\s*topics:\s*(.*?)\s*\|\s*keywords:\s*(.*)",
    re.IGNORECASE,
)

# Lazily-initialised persistent client — avoids per-call TCP/SSL overhead.
_http_client: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=_TIMEOUT)
    return _http_client


@dataclass
class EnrichmentResult:
    entities: str = ""
    topics: str = ""
    keywords: str = ""


async def enrich_chunks(texts: List[str], model: Optional[str] = None) -> List[EnrichmentResult]:
    """Extract entities, topics, keywords from a list of chunk texts.

    Returns a list of EnrichmentResult — same length as texts.
    Empty results on any failure (never raises).
    """
    if not texts:
        return []

    _model = model or settings.enrich_model
    empty = [EnrichmentResult() for _ in texts]

    # Build numbered passages in the prompt
    passages = "\n\n".join(
        f"[{i + 1}] {text[:600]}"  # cap at 600 chars per passage to keep prompt manageable
        for i, text in enumerate(texts)
    )

    prompt = (
        "/no_think\n"
        f"For each of the {len(texts)} passages below, extract metadata.\n"
        "Reply ONLY in this exact format, one line per passage:\n"
        "[N] entities: X, Y, Z | topics: A, B | keywords: p, q, r, s, t\n\n"
        "Rules:\n"
        "- entities: names of files, functions, classes, people, orgs, or key concepts (max 5)\n"
        "- topics: 2-3 broad thematic categories\n"
        "- keywords: 5-7 important domain-specific terms\n"
        "- Use comma-separated values within each field\n"
        "- Do NOT add explanations or extra text\n\n"
        f"Passages:\n{passages}\n\n"
        "Metadata:"
    )

    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    try:
        resp = await _client().post(
            url,
            json={
                "model": _model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "num_predict": 80 * len(texts),
                    "temperature": 0.1,
                    "top_p": 0.9,
                },
            },
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()
        cleaned = _THINK_RE.sub("", raw).strip()
    except Exception as exc:
        logger.debug("Enrichment LLM call failed: %s", exc)
        return empty

    # Parse response lines
    results = list(empty)
    for line in cleaned.splitlines():
        m = _LINE_RE.match(line.strip())
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(texts):
            results[idx] = EnrichmentResult(
                entities=_clean_csv(m.group(2)),
                topics=_clean_csv(m.group(3)),
                keywords=_clean_csv(m.group(4)),
            )
    return results


def _clean_csv(value: str) -> str:
    """Normalize comma-separated values: strip whitespace, deduplicate, lowercase."""
    parts = [p.strip().lower() for p in value.split(",") if p.strip()]
    seen: set = set()
    out: List[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return ", ".join(out)
