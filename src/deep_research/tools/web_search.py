"""Cached Tavily web search, wrapped as an agent tool.

The course dumped `str(await client.search(query))` into the tool output, which
hands the model a wall of raw JSON. Here the results are distilled to a short
answer plus a titled source list, which cuts tokens and makes the agent's
citations legible in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..cache import SearchCache
from ..config import Settings

# Longest source snippet handed to the model. Tavily already returns excerpts,
# not full pages, so this only trims pathologically long ones.
SNIPPET_CHARS = 600


class TavilyClient(Protocol):
    """The slice of the Tavily client this module uses."""

    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(slots=True)
class WebSearcher:
    """Builds the `web_search` tool, backed by Tavily and an on-disk cache."""

    settings: Settings
    client: Any | None = None
    cache: SearchCache | None = None
    # `queries` counts tool invocations (cache included); `calls` counts only the
    # upstream searches that were actually billed.
    queries: int = field(default=0, init=False)
    calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.cache is None:
            self.cache = SearchCache(
                self.settings.cache_file,
                ttl_seconds=self.settings.cache_ttl_hours * 3600,
            )
        if self.client is None and self.settings.search_provider == "tavily":
            from tavily import AsyncTavilyClient

            self.client = AsyncTavilyClient(api_key=self.settings.tavily_api_key)

    async def search(self, query: str) -> str:
        """Return a condensed web search result for `query`."""
        assert self.cache is not None
        key = self.cache.make_key("tavily", query, str(self.settings.search_max_results), "v2")
        self.queries += 1
        return await self.cache.wrap(key, lambda: self._search_uncached(query))

    async def _search_uncached(self, query: str) -> str:
        if self.client is None:
            return "Web search is disabled for this run. Answer from your own knowledge."

        self.calls += 1
        payload = await self.client.search(query, max_results=self.settings.search_max_results)
        return self.format_results(payload)

    @staticmethod
    def format_results(payload: dict[str, Any]) -> str:
        """Turn a Tavily payload into compact, citable text."""
        if not isinstance(payload, dict):
            return str(payload)

        lines: list[str] = []
        answer = payload.get("answer")
        if answer:
            lines.append(f"ANSWER: {answer}")

        results = payload.get("results") or []
        if results:
            lines.append("SOURCES:")
            for item in results:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "Untitled").strip()
                url = str(item.get("url") or "").strip()
                snippet = " ".join(str(item.get("content") or "").split())
                if len(snippet) > SNIPPET_CHARS:
                    # Say so, so the model does not read a hard cut as the whole source.
                    snippet = snippet[:SNIPPET_CHARS].rstrip() + " [...]"
                lines.append(f"- [{title}]({url}): {snippet}")

        return "\n".join(lines) or "No results found."


def build_search_tools(searcher: WebSearcher) -> list[Any]:
    """Return the tool list for the research agent.

    The docstring is what the model sees when deciding whether to call the tool,
    so it says when the tool is the right move.
    """

    async def web_search(query: str) -> str:
        """Search the web for current information about a topic.

        Use this whenever a question needs facts, figures, dates or events that
        may be outside your training data. Pass a focused natural-language
        query, not a keyword list.
        """
        return await searcher.search(query)

    web_search.__name__ = "web_search"
    return [web_search]
