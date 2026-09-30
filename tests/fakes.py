"""Offline stand-ins for the LLM, the researcher agent and Tavily.

The point of these is that the entire pipeline — fan-out, `collect_events`, the
reflection loop, persistence — runs end to end in a test with no network, no
API keys and no flakiness. They implement only the surface the workflow
actually calls, which doubles as a check that the workflow stays decoupled from
any one provider.

They also copy the real signatures, including the parts that are inconvenient.
`FakeLLM.astructured_predict` insists on a `PromptTemplate` exactly as
LlamaIndex does, because a laxer fake hides real breakage: a prompt-type
mismatch once made the entire offline suite green while every live call failed
and silently degraded.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from deep_research.schemas import ResearchPlan, ReviewVerdict
from llama_index.core.prompts.base import BasePromptTemplate


@dataclass
class FakeLLM:
    """A scripted LLM.

    `plans` and `reviews` are consumed one item per call and the last item
    repeats once the script runs out, so `reviews=[ReviewVerdict(acceptable=True)]`
    means "always accept" and `reviews=[reject, accept]` means "reject once".
    """

    plans: Sequence[ResearchPlan | None] = ()
    reviews: Sequence[ReviewVerdict] = ()
    report: str = "# Report\n\nA short but complete briefing."
    chunk_size: int = 24

    plans_seen: list[str] = field(default_factory=list)
    writer_prompts: list[str] = field(default_factory=list)
    critic_prompts: list[str] = field(default_factory=list)
    complete_prompts: list[str] = field(default_factory=list)
    _plan_index: int = 0
    _review_index: int = 0

    def _next_in(self, script: Sequence[Any], index: int) -> Any:
        if not script:
            return None
        return script[min(index, len(script) - 1)]

    async def astructured_predict(self, output_cls: type, prompt: Any, **_: Any) -> Any:
        # Enforce the real LlamaIndex contract. `astructured_predict` declares
        # `prompt: PromptTemplate`, and handing it a bare `str` fails pydantic
        # validation before any request goes out. A fake that quietly accepted a
        # `str` is why that bug reached production: the suite passed while every
        # live call fell back to lenient parsing.
        if not isinstance(prompt, BasePromptTemplate):
            raise TypeError(
                "astructured_predict needs a PromptTemplate, not "
                f"{type(prompt).__name__}; wrap it with RichPromptTemplate"
            )
        text = prompt.template_str
        if output_cls is ResearchPlan:
            self.plans_seen.append(text)
            scripted = self._next_in(self.plans, self._plan_index)
            self._plan_index += 1
            return scripted if scripted is not None else ResearchPlan(questions=[])
        if output_cls is ReviewVerdict:
            self.critic_prompts.append(text)
            scripted = self._next_in(self.reviews, self._review_index)
            self._review_index += 1
            return scripted or ReviewVerdict(acceptable=True)
        raise AssertionError(f"unexpected structured output request: {output_cls}")

    async def astream_complete(self, prompt: str) -> AsyncIterator[Any]:
        self.writer_prompts.append(prompt)

        async def chunks() -> AsyncIterator[Any]:
            for start in range(0, len(self.report), self.chunk_size):
                await asyncio.sleep(0)  # let other tasks interleave
                yield SimpleNamespace(delta=self.report[start : start + self.chunk_size], text="")

        return chunks()

    async def acomplete(self, prompt: str) -> Any:
        self.complete_prompts.append(prompt)
        return SimpleNamespace(text=self.report)


@dataclass
class ScriptedResearchAgent:
    """A researcher agent that answers without touching a real model.

    Tracks how many runs overlapped, which is how the fan-out test proves the
    research step really is parallel.
    """

    delay: float = 0.0
    tools: dict[str, Any] = field(default_factory=dict)

    questions: list[str] = field(default_factory=list)
    concurrent_now: int = 0
    max_concurrent: int = 0

    async def run(self, user_msg: str, **_: Any) -> str:
        self.questions.append(user_msg)
        self.concurrent_now += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent_now)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            return f"Finding for question {len(self.questions)}: the answer body."
        finally:
            self.concurrent_now -= 1


@dataclass
class StubTavilyClient:
    """Returns canned Tavily payloads and counts the calls it received."""

    answer: str = "The stub answer."
    results: list[dict[str, str]] = field(
        default_factory=lambda: [
            {"title": "Example", "url": "https://example.com", "content": "Body text."}
        ]
    )
    calls: list[str] = field(default_factory=list)
    fail_with: Exception | None = None

    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        if self.fail_with is not None:
            raise self.fail_with
        self.calls.append(query)
        return {"answer": self.answer, "results": list(self.results)}


@dataclass
class StubSerperClient:
    """Returns a canned Serper payload, shaped like the real `/search` response.

    The field names are the ones Serper actually returns (`organic`, `link`,
    `snippet`) rather than Tavily's, because that difference is the whole
    reason the parsing is per-provider.
    """

    organic: list[dict[str, str]] = field(
        default_factory=lambda: [
            {
                "title": "Example",
                "link": "https://example.com",
                "snippet": "Body text.",
                "date": "Mar 15, 2026",
                "position": 1,
            }
        ]
    )
    calls: list[str] = field(default_factory=list)
    max_results: list[int] = field(default_factory=list)
    fail_with: Exception | None = None

    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        if self.fail_with is not None:
            raise self.fail_with
        self.calls.append(query)
        self.max_results.append(int(kwargs.get("max_results", 0)))
        return {"searchParameters": {"q": query}, "organic": list(self.organic), "credits": 1}


def extract_question(user_msg: str) -> str:
    """Pull the question out of the prompt the research step builds."""
    start = user_msg.find("<question>")
    end = user_msg.find("</question>")
    if start == -1 or end == -1:
        return user_msg.strip().splitlines()[0]
    return user_msg[start + len("<question>") : end].strip()
