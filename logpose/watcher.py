"""
watcher.py — File-system watcher for hot-reindex on file changes.

Uses the ``watchfiles`` library (Rust-backed, low CPU) to watch corpus
directories for added, modified, or deleted files and updates the index
in real-time.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional, Set

import watchfiles
from pydantic import AnyUrl

from .config import Settings

logger = logging.getLogger(__name__)


async def file_watcher_task(
    settings: Settings,
    indexer: Any,
    extensions: set[str],
    subscribed_uris: Set[str],
    session: Any = None,
) -> None:
    """Watch corpus dirs for changes; re-index on any add/modify/delete event.

    Parameters
    ----------
    settings:        Server-wide config.
    indexer:         FileIndexer instance.
    extensions:      Set of supported file extensions.
    subscribed_uris: Mutable set of resource URIs the client has subscribed to.
    session:         Active MCP session (for resource-updated notifications).
    """
    watch_dirs = [d for d in settings.corpus_dirs if d.exists()]
    if not watch_dirs:
        logger.warning("File watcher skipped: no corpus directories to watch.")
        return

    logger.info("Starting file watcher on: %s", ", ".join(str(d) for d in watch_dirs))
    try:
        async for changes in watchfiles.awatch(*watch_dirs):
            for change_type, path_str in changes:
                path = Path(path_str)
                ext = path.suffix.lower()

                if ext not in extensions:
                    continue

                file_uri = AnyUrl(f"file://{path.resolve()}")

                if change_type in (watchfiles.Change.added, watchfiles.Change.modified):
                    logger.info("Watcher: indexing changed file %s", path.name)
                    try:
                        await indexer.index_file(path)
                    except Exception as exc:
                        logger.error("Watcher failed to index %s: %s", path.name, exc)

                elif change_type == watchfiles.Change.deleted:
                    logger.info("Watcher: removing deleted file %s", path.name)
                    try:
                        await indexer.delete_file(path)
                    except Exception as exc:
                        logger.error("Watcher failed to delete %s: %s", path.name, exc)

                # Notify the client if it subscribed to this specific resource
                if session and str(file_uri) in subscribed_uris:
                    try:
                        await session.send_resource_updated(file_uri)
                    except Exception as exc:
                        logger.debug("send_resource_updated failed: %s", exc)

    except asyncio.CancelledError:
        logger.info("File watcher stopped.")
    except Exception as exc:
        logger.error("File watcher crashed: %s", exc)
