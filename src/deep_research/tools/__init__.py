"""Tools the agents can call.

Today that is exactly one: a cached Tavily web search. Tools are plain async
functions with type hints and a docstring — LlamaIndex reads the signature to
build the schema it shows the model, so both matter.
"""

from __future__ import annotations

from .web_search import WebSearcher, build_search_tools

__all__ = ["WebSearcher", "build_search_tools"]
