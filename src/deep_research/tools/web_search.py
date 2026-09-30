"""Cached web search, wrapped as an agent tool.

Two providers are supported, because they fail differently and cost differently:
Tavily (an API built for agents, which also returns a synthesised `answer`) and
Google Serper (a thin front end over Google Search, cheaper per query and
without the synthesised answer). The provider is a setting, not a code path, so
switching is one environment variable.

The course dumped `str(await client.search(query))` into the tool output, which
hands the model a wall of raw JSON. Here each provider's payload is parsed into
the same normalised shape and rendered as a short answer plus a titled source
list, which cuts tokens and makes the agent's citations legible in the UI.

The two payload shapes are not interchangeable, which is the whole reason the
parsing is per-provider rather than duck-typed:

    Tavily   {"answer": str, "results": [{"title", "url", "content"}]}
    Serper   {"credits": int, "organic":  [{"title", "link", "snippet", "date"}]}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..cache import SearchCache
from ..config import Settings

# Longest source snippet handed to the model. Both providers return excerpts,
# not full pages, so this only trims pathologically long ones.
#
# This is a budget knob, not a formatting preference. A researcher's context
# grows with every tool call and the whole history is re-sent on the next turn,
# so four 600-char snippets plus the model's own prose reached ~8.3k tokens and
# tripped Groq's per-minute *input* limit on a request that no amount of
# waiting could fix. 300 keeps a useful excerpt and roughly halves the cost.
SNIPPET_CHARS = 300

# How many searches one researcher may make. The agent decides when to stop, and
# an unbounded loop is the most reliable way to outgrow the context window. Past
# this the tool says so and the model is asked to answer with what it has.
MAX_SEARCH_CALLS_PER_RUN = 3

SERPER_BASE_URL = "https://google.serper.dev"
SERPER_TIMEOUT = 20.0

# Bumped when the rendered text changes shape, so old cache entries are ignored.
_CACHE_FORMAT = "v3"


class SearchError(RuntimeError):
    """A search provider returned an error, or could not be reached."""


class TavilyClient(Protocol):
    """The slice of the Tavily client this module uses."""

    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(slots=True)
class SearchHit:
    """One source, normalised across providers."""

    title: str
    url: str
    snippet: str
    date: str = ""


class SerperClient:
    """Minimal async client for the Serper `/search` endpoint.

    Hand-rolled on httpx rather than pulled from a package: the endpoint is one
    POST, and this keeps the async behaviour the concurrent research workers
    need. `http` can be injected so tests exercise the request without a socket.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = SERPER_BASE_URL,
        http: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._http = http
        self._owns_http = http is None

    def _client(self) -> Any:
        if self._http is None:
            import httpx

            self._http = httpx.AsyncClient(timeout=SERPER_TIMEOUT)
        return self._http

    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        import httpx

        max_results = int(kwargs.get("max_results", 4))
        try:
            response = await self._client().post(
                f"{self.base_url}/search",
                json={"q": query, "num": max_results},
                headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise SearchError(f"serper request failed: {exc}") from exc

        if response.status_code != 200:
            raise SearchError(self._describe_failure(response))
        try:
            return response.json()
        except ValueError as exc:
            raise SearchError("serper returned a body that was not JSON") from exc

    @staticmethod
    def _describe_failure(response: Any) -> str:
        """Serper answers 403 with {"message": "Unauthorized."}; say what failed."""
        detail = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                detail = str(body.get("message") or body.get("error") or "")
        except ValueError:
            detail = ""
        if response.status_code in (401, 403):
            return f"serper rejected the API key (HTTP {response.status_code}) {detail}".strip()
        if response.status_code == 429:
            return "serper rate limit reached (HTTP 429); lower CONCURRENCY or retry later"
        return f"serper returned HTTP {response.status_code} {detail}".strip()

    async def aclose(self) -> None:
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None


@dataclass(slots=True)
class WebSearcher:
    """Builds the `web_search` tool, backed by a provider and an on-disk cache."""

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
        if self.client is None:
            self.client = self._build_client()

    def _build_client(self) -> Any | None:
        provider = self.settings.search_provider
        if provider == "tavily":
            from tavily import AsyncTavilyClient

            return AsyncTavilyClient(api_key=self.settings.tavily_api_key)
        if provider == "serper":
            return SerperClient(
                api_key=self.settings.serper_api_key,
                base_url=self.settings.serper_base_url,
            )
        return None

    async def search(self, query: str) -> str:
        """Return a condensed web search result for `query`."""
        assert self.cache is not None
        # The provider is part of the key: a Tavily answer and a Serper answer
        # for the same query are different results, and must not share a cache
        # entry just because the query text matches.
        key = self.cache.make_key(
            self.settings.search_provider,
            query,
            str(self.settings.search_max_results),
            _CACHE_FORMAT,
        )
        self.queries += 1
        return await self.cache.wrap(key, lambda: self._search_uncached(query))

    async def _search_uncached(self, query: str) -> str:
        if self.client is None:
            return "Web search is disabled for this run. Answer from your own knowledge."

        # One searcher is shared by every researcher in a run, so this counts the
        # whole run, not one agent. The cap exists because each result lands in
        # the calling agent's message history and is re-sent on every later turn;
        # an agent that keeps searching grows its own context without limit.
        if self.queries > MAX_SEARCH_CALLS_PER_RUN * max(1, self.settings.max_questions):
            return (
                f"Search budget spent ({self.queries} queries so far). "
                "Do not call this tool again. Write your final answer now using "
                "the sources already gathered, and say if something is unverified."
            )

        self.calls += 1
        payload = await self.client.search(query, max_results=self.settings.search_max_results)
        if self.settings.search_provider == "serper":
            return self.format_serper_results(payload)
        return self.format_results(payload)

    async def aclose(self) -> None:
        """Release the provider's HTTP connection pool, if it owns one."""
        closer = getattr(self.client, "aclose", None)
        if callable(closer):
            await closer()

    # -- rendering ---------------------------------------------------------
    @staticmethod
    def format_results(payload: dict[str, Any]) -> str:
        """Render a Tavily payload as compact, citable text."""
        if not isinstance(payload, dict):
            return str(payload)

        lines: list[str] = []
        answer = payload.get("answer")
        if answer:
            lines.append(f"ANSWER: {answer}")
        hits = [
            SearchHit(
                title=str(item.get("title") or "Untitled"),
                url=str(item.get("url") or ""),
                snippet=" ".join(str(item.get("content") or "").split()),
            )
            for item in (payload.get("results") or [])
            if isinstance(item, dict)
        ]
        return WebSearcher._render(lines, hits)

    @staticmethod
    def format_serper_results(payload: dict[str, Any]) -> str:
        """Render a Serper payload as compact, citable text.

        Serper has no synthesised answer, so the source list is the whole result.
        The `date` is kept when present: for a research report, how recent a
        source is often matters more than its wording.
        """
        if not isinstance(payload, dict):
            return str(payload)

        hits: list[SearchHit] = []
        for item in payload.get("organic") or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "Untitled").strip()
            url = str(item.get("link") or "").strip()
            if not url:
                # A hit with no link cannot be cited, so it is not worth the tokens.
                continue
            snippet = " ".join(str(item.get("snippet") or "").split())
            hits.append(
                SearchHit(
                    title=title,
                    url=url,
                    snippet=snippet,
                    date=str(item.get("date") or "").strip(),
                )
            )
        return WebSearcher._render([], hits)

    @staticmethod
    def _render(preamble: list[str], hits: list[SearchHit]) -> str:
        lines = list(preamble)
        if hits:
            lines.append("SOURCES:")
            for hit in hits:
                suffix = f" ({hit.date})" if hit.date else ""
                snippet = hit.snippet
                if len(snippet) > SNIPPET_CHARS:
                    # Say so, so the model does not read a hard cut as the whole source.
                    snippet = snippet[:SNIPPET_CHARS].rstrip() + " [...]"
                lines.append(f"- [{hit.title}]({hit.url}){suffix}: {snippet}")
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
