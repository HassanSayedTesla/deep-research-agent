"""Unit tests for settings, prompt/response salvage, search formatting and the graph spec."""

from __future__ import annotations

from io import BytesIO, TextIOWrapper

import pytest
from pydantic import ValidationError
from rich.console import Console

from deep_research import cli
from deep_research.config import Settings
from deep_research.events import (
    DONE,
    EDGE_FLOW,
    NODE_ACTIVATED,
    REPORT_DONE,
    RUN_FAILED,
    RUN_STARTED,
    RunEvent,
)
from deep_research.graph import (
    EDGES,
    LANE_CRITIC,
    LANE_PLANNER,
    LANE_RESEARCH,
    LANE_WRITER,
    NODES,
    graph_payload,
    parse_node_id,
    worker_node_id,
)
from deep_research.llm import _salvage, build_llm
from deep_research.schemas import ResearchPlan, ReviewVerdict
from deep_research.tools.web_search import WebSearcher, build_search_tools
from fakes import StubTavilyClient


# -- settings --------------------------------------------------------------
def test_settings_read_the_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "env-key")
    monkeypatch.setenv("MAX_QUESTIONS", "9")

    settings = Settings(_env_file=None)

    assert settings.groq_api_key == "env-key"
    assert settings.max_questions == 9


def test_settings_reject_nonsense(monkeypatch):
    monkeypatch.setenv("MAX_REVIEW_CYCLES", "not-a-number")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_reject_an_unknown_search_provider(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "bing")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_missing_llm_key_is_explained():
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        Settings(groq_api_key="", _env_file=None).require_llm_key()


def test_missing_search_key_is_explained():
    settings = Settings(groq_api_key="k", tavily_api_key="", _env_file=None)

    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        settings.require_search_key()


def test_search_disabled_needs_no_search_key():
    Settings(groq_api_key="k", search_provider="none", _env_file=None).require_search_key()


def test_paths_are_expanded():
    settings = Settings(runs_dir="~/research", _env_file=None)

    assert settings.runs_dir.is_absolute()
    assert "~" not in str(settings.runs_dir)


def test_build_llm_uses_the_configured_model(monkeypatch):
    settings = Settings(groq_api_key="k", model="llama-3.1-8b-instant", _env_file=None)

    llm = build_llm(settings)

    assert llm.model == "llama-3.1-8b-instant"
    assert llm.temperature == settings.temperature


def test_build_llm_refuses_without_a_key():
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        build_llm(Settings(groq_api_key="", _env_file=None))


# -- response salvage ------------------------------------------------------
def test_salvage_parses_plain_json():
    assert _salvage('{"questions": ["a?"]}', ResearchPlan).questions == ["a?"]


def test_salvage_strips_a_fenced_block():
    text = '```json\n{"questions": ["a?"]}\n```'

    assert _salvage(text, ResearchPlan).questions == ["a?"]


def test_salvage_ignores_trailing_commentary():
    text = '{"acceptable": true, "feedback": ""} -- hope that helps!'

    assert _salvage(text, ReviewVerdict).acceptable is True


def test_salvage_returns_none_for_garbage():
    assert _salvage("no json at all", ResearchPlan) is None


def test_salvage_returns_none_on_a_type_mismatch():
    assert _salvage('{"questions": "not a list"}', ResearchPlan) is None


# -- search formatting -----------------------------------------------------
def test_format_results_uses_answer_and_sources():
    payload = {
        "answer": "Planetary gearboxes share a sun gear.",
        "results": [
            {"title": "Wiki", "url": "https://w.example", "content": "x " * 400},
            {"title": "", "url": "https://b.example", "content": ""},
        ],
    }

    text = WebSearcher.format_results(payload)

    assert text.startswith("ANSWER: Planetary gearboxes")
    assert "- [Wiki](https://w.example):" in text
    assert "[...]" in text, "long snippets should be truncated"
    assert "- [Untitled](https://b.example)" in text


def test_format_results_without_an_answer():
    payload = {"results": [{"title": "T", "url": "u", "content": "c"}]}

    assert "SOURCES:" in WebSearcher.format_results(payload)


def test_format_results_of_an_empty_payload():
    assert WebSearcher.format_results({}) == "No results found."


def test_format_results_collapses_newlines():
    text = WebSearcher.format_results(
        {"results": [{"title": "T", "url": "u", "content": "a\n\nb   c"}]}
    )

    assert "a b c" in text


async def test_search_with_no_client_answers_from_knowledge():
    settings = Settings(groq_api_key="k", search_provider="none", _env_file=None)
    searcher = WebSearcher(settings)

    assert "disabled" in await searcher.search("anything")
    assert searcher.calls == 0


def test_the_search_tool_is_documented_for_the_model():
    settings = Settings(groq_api_key="k", _env_file=None)
    tools = build_search_tools(WebSearcher(settings, client=StubTavilyClient()))

    assert len(tools) == 1
    assert tools[0].__name__ == "web_search"
    # The docstring is the only thing the model sees when choosing a tool.
    assert "Search the web" in tools[0].__doc__
    assert "query" in tools[0].__annotations__


# -- graph spec ------------------------------------------------------------
def test_every_edge_points_at_a_known_node():
    node_ids = {node.id for node in NODES}

    for edge in EDGES:
        assert edge.source in node_ids
        assert edge.target in node_ids


def test_the_four_stages_have_their_own_lane():
    lanes = {node.id: node.lane for node in NODES}

    assert lanes["planner"] == LANE_PLANNER
    assert lanes["research"] == LANE_RESEARCH
    assert lanes["writer"] == LANE_WRITER
    assert lanes["critic"] == LANE_CRITIC


def test_only_the_research_lane_spawns_extra_nodes():
    assert [node.id for node in NODES if node.spawns] == ["research"]


def test_the_reflection_loop_is_declared():
    loop = [edge for edge in EDGES if edge.kind == "loop"]

    assert [(e.source, e.target) for e in loop] == [("critic", "planner")]


def test_graph_payload_is_json_serialisable():
    import json

    payload = graph_payload()

    assert json.loads(json.dumps(payload))["nodes"][0]["id"] == "planner"
    assert {lane["label"] for lane in payload["lanes"]} == {
        "Plan",
        "Research",
        "Write",
        "Critique",
    }


def test_worker_node_ids_round_trip():
    assert worker_node_id(0) == "research#0"
    assert parse_node_id("research#3") == ("research", 3)
    assert parse_node_id("writer") == ("writer", None)


def test_glyphs_fall_back_when_the_console_cannot_encode_them(monkeypatch):
    """A Windows console on cp1252 must not get a UnicodeEncodeError."""
    monkeypatch.setattr(cli, "console", _console(encoding="cp1252"))

    assert cli._glyph("✓", "OK") == "OK", "cp1252 has no check mark"
    assert cli._glyph("✗", "x") == "x"
    assert cli._glyph("↳", "->") == "->"
    # cp1252 does have a middle dot, so that one is worth keeping.
    assert cli._glyph("·", "|") == "·"


def test_glyphs_are_kept_on_a_unicode_console(monkeypatch):
    monkeypatch.setattr(cli, "console", _console(encoding="utf-8"))

    assert cli._glyph("✓", "OK") == "✓"


@pytest.mark.parametrize("encoding", ["cp1252", "utf-8"])
def test_rendering_a_run_never_raises_an_encoding_error(monkeypatch, encoding: str):
    """Every event kind has to survive a narrow, ASCII-only terminal."""
    console = _console(encoding=encoding)
    monkeypatch.setattr(cli, "console", console)
    samples = [
        (RUN_STARTED, {"run_id": "r1", "config": {"model": "m", "search_provider": "tavily"}}),
        (NODE_ACTIVATED, {"node": "research#0", "label": "Researcher", "detail": "Why?"}),
        (NODE_ACTIVATED, {"node": "writer", "label": "Writer", "detail": ""}),
        (EDGE_FLOW, {"source": "planner", "target": "research#0", "label": "question"}),
        (REPORT_DONE, {"report_path": "runs/r1/report.md"}),
        (RUN_FAILED, {"error": "RuntimeError: provider is down"}),
        (DONE, {}),
    ]

    for kind, data in samples:
        cli._render_event(RunEvent(kind=kind, data=data))

    written = console.file.detach().getvalue()
    assert written
    written.decode(encoding)  # would raise if a glyph had been dropped


def _console(encoding: str) -> Console:
    """A console writing to an in-memory stream with a strict codec."""
    stream = TextIOWrapper(BytesIO(), encoding=encoding, errors="strict", newline="")
    return Console(file=stream, width=80, force_terminal=False)
