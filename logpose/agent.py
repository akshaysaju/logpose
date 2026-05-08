"""
agent.py — Agentic RAG: LLM-driven tool-calling loop for multi-hop retrieval.

The agent has access to four tools:
  search("query")     — hybrid semantic+BM25 search, returns top chunks
  get_file("path")    — retrieve all chunks from a specific file
  filter("keyword")   — narrow results to chunks containing keyword
  answer("text")      — final answer (terminates the loop)

Tool dispatch uses text-pattern matching (not JSON) for robustness with
small local models. The model outputs:
  TOOL: search("what does the chunker do")
  TOOL: get_file("logpose/chunker.py")
  TOOL: answer("The chunker splits files into typed chunks...")

The agent iterates up to config.agent_max_steps times. If no TOOL: pattern
is found, the response is treated as the final answer directly.

Research basis: ReAct (Yao et al., 2022) + tool-calling pattern adapted
for small local LLMs that struggle with structured JSON output.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from .config import settings
from .searcher import FileSearcher, SearchResult

logger = logging.getLogger(__name__)

_TOOL_RE = re.compile(r'TOOL:\s*(\w+)\("([^"]*)"\)', re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TIMEOUT = 60.0  # Agent needs more time than HyDE


@dataclass
class AgentResult:
    """Final result from a RAG agent run."""
    answer: str
    citations: List[str] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    steps_used: int = 0
    model: str = ""


class RAGAgent:
    """
    Tool-calling RAG agent. Uses FileSearcher as the retrieval backend.

    Usage:
        agent = RAGAgent(searcher)
        result = await agent.run("How does the Python chunker handle class methods?")
    """

    _SYSTEM_PROMPT = """You are a precise document search assistant with access to a local knowledge base.

You have these tools:
  TOOL: search("your query here")     — search for relevant documents
  TOOL: get_file("path/to/file.ext")  — get content of a specific file
  TOOL: filter("keyword")             — filter current results by keyword
  TOOL: answer("your final answer")   — provide your answer and stop

Rules:
- Use search first to find relevant information
- Use get_file when you need full context of a specific file
- Use filter to narrow down results when you have too many
- Always end with TOOL: answer("...") when you have enough information
- Base your answer ONLY on retrieved documents, not prior knowledge
- If you cannot find the answer, say so clearly in your answer
- Be concise and factual

After each tool call, you will receive the results. Analyze them and decide next steps.
Use at most {max_steps} tool calls total."""

    def __init__(self, searcher: FileSearcher, max_steps: Optional[int] = None) -> None:
        self.searcher = searcher
        self.max_steps = max_steps or settings.agent_max_steps
        self._context_chunks: List[SearchResult] = []

    async def run(self, query: str) -> AgentResult:
        """Run the agent loop. Returns AgentResult with answer + citations."""
        model = settings.agent_model
        trace: List[str] = []
        citations: List[str] = []
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": self._SYSTEM_PROMPT.format(max_steps=self.max_steps),
            },
            {"role": "user", "content": f"Question: {query}"},
        ]

        final_answer = ""
        steps_used = 0

        async with httpx.AsyncClient(timeout=_TIMEOUT) as http:
            for step in range(self.max_steps):
                steps_used = step + 1
                response = await self._llm_call(messages, model, http)
                if not response:
                    trace.append(f"[step {step+1}] LLM call failed — stopping")
                    break

                trace.append(f"[step {step+1}] {response[:200]}{'...' if len(response) > 200 else ''}")

                match = _TOOL_RE.search(response)
                if not match:
                    final_answer = _THINK_RE.sub("", response).strip()
                    break

                tool_name = match.group(1).lower()
                tool_arg = match.group(2).strip()

                tool_result = await self._dispatch(tool_name, tool_arg, citations)

                if tool_name == "answer":
                    final_answer = tool_arg or _THINK_RE.sub("", response).strip()
                    break

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Tool result:\n{tool_result}"})

        if not final_answer:
            final_answer = "I was unable to find a definitive answer in the available documents."

        return AgentResult(
            answer=final_answer,
            citations=list(dict.fromkeys(citations)),  # deduplicate, preserve order
            reasoning_trace=trace,
            steps_used=steps_used,
            model=model,
        )

    async def _dispatch(self, tool_name: str, arg: str, citations: List[str]) -> str:
        """Execute a tool and return its text result."""
        if tool_name == "search":
            return await self._tool_search(arg, citations)
        elif tool_name == "get_file":
            return await self._tool_get_file(arg, citations)
        elif tool_name == "filter":
            return self._tool_filter(arg)
        elif tool_name == "answer":
            return ""  # handled by caller
        else:
            return f"Unknown tool '{tool_name}'. Available: search, get_file, filter, answer."

    async def _tool_search(self, query: str, citations: List[str]) -> str:
        if not query:
            return "Error: search requires a query string."
        try:
            results = await self.searcher.search(query, n_results=5, deduplicate=True)
            self._context_chunks = results
            if not results:
                return "No results found."
            lines = []
            for i, r in enumerate(results, 1):
                citations.append(r.file_path)
                lines.append(
                    f"[{i}] {r.file_name} (score={r.score:.3f})\n"
                    f"    {r.chunk_text[:600]}"
                )
            return "\n\n".join(lines)
        except Exception as exc:
            logger.debug("Agent search failed: %s", exc)
            return f"Search error: {exc}"

    async def _tool_get_file(self, file_path: str, citations: List[str]) -> str:
        if not file_path:
            return "Error: get_file requires a file path."
        try:
            results = await self.searcher.search_by_filename(file_path)
            if not results:
                # Try by semantic search with file prefix
                results = await self.searcher.search(
                    f"file:{file_path}", n_results=10, deduplicate=False
                )
            if not results:
                return f"File not found in index: {file_path}"
            citations.append(results[0].file_path)
            # Return all chunks concatenated (up to 2000 chars)
            text = "\n\n---\n\n".join(r.chunk_text for r in results[:8])
            return f"{file_path}:\n\n{text[:2000]}"
        except Exception as exc:
            logger.debug("Agent get_file failed: %s", exc)
            return f"get_file error: {exc}"

    def _tool_filter(self, keyword: str) -> str:
        if not keyword:
            return "Error: filter requires a keyword."
        kw = keyword.lower()
        filtered = [
            r for r in self._context_chunks
            if kw in r.chunk_text.lower() or kw in r.file_name.lower()
        ]
        if not filtered:
            return f"No results contain '{keyword}'."
        lines = [
            f"[{i}] {r.file_name}\n    {r.chunk_text[:400]}"
            for i, r in enumerate(filtered, 1)
        ]
        return f"Filtered to {len(filtered)} results:\n\n" + "\n\n".join(lines)

    async def _llm_call(
        self, messages: List[Dict[str, str]], model: str, http: httpx.AsyncClient
    ) -> Optional[str]:
        url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
        try:
            resp = await http.post(url, json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": 400,
                    "temperature": 0.2,
                    "top_p": 0.9,
                },
            })
            resp.raise_for_status()
            raw = resp.json()["message"]["content"].strip()
            return _THINK_RE.sub("", raw).strip() or None
        except asyncio.TimeoutError:
            logger.warning("Agent LLM call timed out")
            return None
        except Exception as exc:
            logger.warning("Agent LLM call failed: %s", exc)
            return None
