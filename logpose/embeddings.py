"""
embeddings.py — Shared async embedding client with batch support, retry, and rate-limiting.

Single source of truth for all Ollama /api/embed calls.  Both the indexer and
searcher import from here instead of duplicating HTTP logic.

Key features:
  - Batch embedding: one Ollama call for multiple texts (huge speedup during indexing)
  - Exponential-backoff retry on transient failures (timeout / 5xx)
  - Semaphore caps concurrent Ollama requests across the whole process
  - Persistent AsyncClient reused across calls (no per-call SSL handshake overhead)
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

import httpx

from .config import settings

logger = logging.getLogger(__name__)

# Max concurrent Ollama requests (indexer batches + live search queries).
_MAX_CONCURRENT = 4
_MAX_RETRIES = 3

# Lazily initialised singletons — created inside the running event loop.
_http_client: httpx.AsyncClient | None = None
_semaphore: asyncio.Semaphore | None = None


def _client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=60.0)
    return _http_client


def _sem() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
    return _semaphore


async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts in a **single** Ollama API call.

    Ollama's /api/embed endpoint accepts an array of inputs, so we send the
    whole batch at once instead of one request per chunk.

    Args:
        texts: Strings to embed.  Must be non-empty.

    Returns:
        List of embedding vectors, one per input text, in the same order.

    Raises:
        RuntimeError: If Ollama is unreachable after all retries.
        httpx.HTTPStatusError: On persistent non-2xx responses.
    """
    if not texts:
        return []

    url = settings.ollama_embed_url
    payload = {"model": settings.embed_model, "input": texts}

    async with _sem():
        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(_MAX_RETRIES):
            try:
                response = await _client().post(url, json=payload)
                response.raise_for_status()
                return response.json()["embeddings"]
            except httpx.ConnectError as exc:
                raise RuntimeError(
                    f"Ollama is not reachable at {settings.ollama_base_url}. "
                    "Start it with: ollama serve"
                ) from exc
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if attempt == _MAX_RETRIES - 1:
                    raise
                wait = 2**attempt
                logger.warning(
                    "Embedding attempt %d/%d failed (%s) — retrying in %ds",
                    attempt + 1,
                    _MAX_RETRIES,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)

        raise last_exc


async def get_embedding(text: str) -> List[float]:
    """Embed a single text string. Convenience wrapper around get_embeddings_batch."""
    results = await get_embeddings_batch([text])
    return results[0]


async def warmup() -> None:
    """Pre-load the embedding model into Ollama's GPU/CPU memory.

    Sends a minimal embed request so the first real query doesn't pay the
    cold-start penalty (can be 1-2 seconds for large models).  Logs the
    round-trip time so users can confirm the model is ready.
    """
    import time

    logger.info("Warming up embedding model '%s'…", settings.embed_model)
    t0 = time.monotonic()
    try:
        await get_embeddings_batch(["warmup"])
        elapsed_ms = round((time.monotonic() - t0) * 1000)
        logger.info("Embedding model ready in %d ms", elapsed_ms)
    except Exception as exc:
        logger.warning("Warmup failed (will retry on first real request): %s", exc)
