"""Shared fixtures. Every test here runs offline: no keys, no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from deep_research.config import Settings
from deep_research.schemas import ResearchPlan, ReviewVerdict
from deep_research.storage import RunStore
from deep_research.tools.web_search import WebSearcher
from fakes import FakeLLM, ScriptedResearchAgent, StubTavilyClient

QUESTIONS = [
    "What is a helical gearbox?",
    "How do planetary gearboxes handle load?",
    "Which gearbox suits a mobile crane?",
]


def make_settings(tmp_path: Path, **overrides) -> Settings:
    """Settings wired to `tmp_path`, with web search disabled by default."""
    values = {
        "groq_api_key": "test-key",
        "tavily_api_key": "test-key",
        "serper_api_key": "test-key",
        "search_provider": "none",
        "runs_dir": tmp_path / "runs",
        "cache_file": tmp_path / "cache.json",
        "max_questions": 3,
        "max_review_cycles": 1,
        "cache_ttl_hours": 24,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def llm() -> FakeLLM:
    """Accepts the first draft."""
    return FakeLLM(
        plans=[ResearchPlan(questions=list(QUESTIONS))],
        reviews=[ReviewVerdict(acceptable=True)],
    )


@pytest.fixture
def agent() -> ScriptedResearchAgent:
    return ScriptedResearchAgent()


@pytest.fixture
def searcher(settings: Settings) -> WebSearcher:
    return WebSearcher(settings, client=StubTavilyClient())


@pytest.fixture
def store(settings: Settings) -> RunStore:
    return RunStore(settings.runs_dir)
