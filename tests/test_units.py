"""Unit tests for settings, prompt/response salvage, search formatting and the graph spec."""

from __future__ import annotations

import asyncio
import json
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import RateLimitError
from pydantic import ValidationError
from rich.console import Console

from conftest import make_settings
from deep_research import cli
from deep_research.cache import SearchCache
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
from deep_research.llm import (
    LLMGate,
    ModelUnavailableError,
    _retry_delay,
    _salvage,
    ask_structured,
    available_models,
    build_llm,
    check_model_available,
    is_quota_error,
    is_retryable,
    with_rate_limit_retry,
)
from deep_research.prompts import PLANNER_PROMPT
from deep_research.schemas import ResearchPlan, ReviewVerdict
from deep_research.tools.web_search import (
    SearchError,
    SerperClient,
    WebSearcher,
    build_search_tools,
)
from fakes import FakeLLM, ScriptedResearchAgent, StubSerperClient, StubTavilyClient


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


def test_env_example_matches_the_real_defaults():
    """Docs drift silently otherwise, and a stale default is a silent bug.

    Every key in .env.example must exist on Settings, and every tunable must
    agree with the field default.
    """
    example = (Path(__file__).resolve().parent.parent / ".env.example").read_text(encoding="utf-8")
    declared: dict[str, str] = {}
    for raw in example.splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            declared[key.strip()] = value.strip()

    for name, field in Settings.model_fields.items():
        if name in {"groq_api_key", "tavily_api_key"}:
            continue
        assert name.upper() in declared, f"{name} is missing from .env.example"
        expected = declared[name.upper()]
        default = field.default
        if isinstance(default, Path):
            assert expected == default.as_posix(), name
        else:
            assert expected == str(default).lower(), (
                f"{name}: .env.example says {expected!r}, the default is {default!r}"
            )


def test_the_default_model_is_the_same_everywhere():
    """A model name lives in five places. Groq retired ours mid-project.

    Nothing in an offline suite can tell you a model still exists, but it can
    at least stop the five copies disagreeing, which is how the retired name
    survived a rename elsewhere.
    """
    root = Path(__file__).resolve().parent.parent
    from deep_research.llm import DEFAULT_MODEL

    assert Settings.model_fields["model"].default == DEFAULT_MODEL, (
        "config.Settings and llm.DEFAULT_MODEL disagree"
    )

    example = (root / ".env.example").read_text(encoding="utf-8")
    assert f"MODEL={DEFAULT_MODEL}" in example, ".env.example names a different model"

    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
    assert f"MODEL: ${{MODEL:-{DEFAULT_MODEL}}}" in compose, "docker-compose.yml disagrees"

    readme = (root / "README.md").read_text(encoding="utf-8")
    assert f"`MODEL` | `{DEFAULT_MODEL}`" in readme, "README config table disagrees"


@pytest.fixture
def groq_catalogue(monkeypatch):
    """Serve a fake `GET /models` without opening a socket."""

    def install(handler) -> None:
        transport = httpx.MockTransport(handler)
        original = httpx.AsyncClient

        def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
            return original(*args, **{**kwargs, "transport": transport})

        monkeypatch.setattr(httpx, "AsyncClient", factory)

    return install


def _catalogue(*model_ids: str):
    return lambda request: httpx.Response(200, json={"data": [{"id": m} for m in model_ids]})


async def test_available_models_lists_what_the_key_can_reach(groq_catalogue):
    groq_catalogue(_catalogue("qwen/qwen3.8-27b", "openai/gpt-oss-120b"))

    assert await available_models("gsk_fake") == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]


async def test_a_refused_catalogue_yields_no_models(groq_catalogue):
    groq_catalogue(lambda request: httpx.Response(401, json={"error": "bad key"}))

    assert await available_models("gsk_fake") == []


async def test_a_retired_model_is_reported_before_the_run_starts(monkeypatch):
    """A 404 on the first call is a bad way to learn a model was decommissioned."""
    monkeypatch.setattr(
        "deep_research.llm.available_models",
        AsyncMock(return_value=["qwen/qwen3.8-27b", "openai/gpt-oss-120b"]),
    )
    settings = Settings(_env_file=None, groq_api_key="gsk_fake", model="llama-3.3-70b-versatile")

    with pytest.raises(ModelUnavailableError) as caught:
        await check_model_available(settings)

    message = str(caught.value)
    assert "llama-3.3-70b-versatile" in message
    assert "qwen/qwen3.8-27b" in message, "the error should name the models that do work"


async def test_an_available_model_passes_the_preflight(monkeypatch):
    monkeypatch.setattr(
        "deep_research.llm.available_models", AsyncMock(return_value=["qwen/qwen3.8-27b"])
    )
    settings = Settings(_env_file=None, groq_api_key="gsk_fake", model="qwen/qwen3.8-27b")

    assert await check_model_available(settings) == "qwen/qwen3.8-27b"


async def test_an_unknown_catalogue_never_blocks_a_run(monkeypatch):
    """A preflight that can fail a good run is worse than no preflight at all."""
    monkeypatch.setattr(
        "deep_research.llm.available_models", AsyncMock(side_effect=httpx.ConnectError("no route"))
    )
    settings = Settings(_env_file=None, groq_api_key="gsk_fake", model="qwen/qwen3.8-27b")

    assert await check_model_available(settings) == "qwen/qwen3.8-27b"


async def test_a_refused_catalogue_never_blocks_a_run(monkeypatch):
    monkeypatch.setattr("deep_research.llm.available_models", AsyncMock(return_value=[]))
    settings = Settings(_env_file=None, groq_api_key="gsk_fake", model="qwen/qwen3.8-27b")

    assert await check_model_available(settings) == "qwen/qwen3.8-27b"


async def test_the_runner_refuses_to_start_on_a_retired_model(monkeypatch, settings: Settings):
    """The preflight is the whole point, so prove it stops a run, not just warns.

    Building the LLM and the searcher both cost money or a socket, so the check
    has to happen before either. Assert the run fails and nothing was started.
    """
    from deep_research.runner import ResearchRunner

    async def refuse(*_: Any, **__: Any) -> str:
        raise ModelUnavailableError(
            "MODEL=llama-3.3-70b-versatile is not available to your Groq key."
        )

    monkeypatch.setattr("deep_research.runner.check_model_available", refuse)

    settings.model = "llama-3.3-70b-versatile"
    runner = ResearchRunner("gearbox selection", settings)

    events = [event async for event in runner.run()]

    kinds = [event.kind for event in events]
    assert "run_failed" in kinds
    assert "run_started" not in kinds, "the run must not start if the model is gone"
    failure = next(event for event in events if event.kind == "run_failed")
    assert "not available" in failure.data.get("error", "")


async def test_a_runner_given_an_injected_llm_never_calls_the_network(
    monkeypatch, settings: Settings
):
    """Offline tests inject a fake LLM, so the preflight must stand down for it."""
    from deep_research.runner import ResearchRunner

    def explode(*_: Any, **__: Any) -> str:
        raise AssertionError("preflight ran even though a fake LLM was injected")

    monkeypatch.setattr("deep_research.runner.check_model_available", explode)

    runner = ResearchRunner(
        "gearbox selection",
        settings,
        llm=FakeLLM(),
        searcher=WebSearcher(settings, client=StubTavilyClient()),
        research_agent=ScriptedResearchAgent(),
    )
    events = [event async for event in runner.run()]

    assert "run_failed" not in [event.kind for event in events]


async def test_an_injected_searcher_is_left_open_for_its_owner(monkeypatch, settings: Settings):
    """Closing a caller's searcher is a bug the tests never noticed.

    The condition used to be `searcher is self._searcher`, which closed an
    injected searcher and leaked one the runner had built - exactly backwards.
    Both directions are asserted here because either one alone passes.
    """
    from deep_research.runner import ResearchRunner

    closed: list[bool] = []
    shared = WebSearcher(settings, client=StubTavilyClient())

    # WebSearcher is a slots dataclass, so the method cannot be swapped on the
    # instance. Patch the class and match on identity instead.
    original_close = WebSearcher.aclose

    async def spy(self: WebSearcher) -> None:
        if self is shared:
            closed.append(True)
        await original_close(self)

    monkeypatch.setattr(WebSearcher, "aclose", spy)

    runner = ResearchRunner(
        "gearbox selection",
        settings,
        llm=FakeLLM(),
        searcher=shared,
        research_agent=ScriptedResearchAgent(),
    )
    async for _ in runner.run():
        pass

    assert closed == [], "the runner closed a searcher it did not create"


async def test_a_runner_built_searcher_is_closed(monkeypatch, settings: Settings):
    """The other half of the pair: what the runner creates, the runner closes."""
    from deep_research.runner import ResearchRunner

    created: list[WebSearcher] = []
    closed: list[WebSearcher] = []
    real_init = WebSearcher.__init__
    real_close = WebSearcher.aclose

    def spy_init(self: WebSearcher, *args: Any, **kwargs: Any) -> None:
        real_init(self, *args, **kwargs)
        created.append(self)

    async def spy_close(self: WebSearcher) -> None:
        closed.append(self)
        await real_close(self)

    monkeypatch.setattr(WebSearcher, "__init__", spy_init)
    monkeypatch.setattr(WebSearcher, "aclose", spy_close)

    runner = ResearchRunner(
        "gearbox selection",
        settings,
        llm=FakeLLM(),
        research_agent=ScriptedResearchAgent(),
    )
    async for _ in runner.run():
        pass

    assert created, "expected the runner to build its own searcher"
    assert closed == created, "the runner must close exactly the searcher it created"


async def test_the_gate_caps_in_flight_llm_turns():
    """`num_workers` bounds researchers; this bounds requests to the provider.

    Four researchers each re-send the system prompt, tool schema and memory, so
    a free-tier 7000 input-token/minute budget only fits about two. The gate is
    what stops the run earning a 429 on the fan-out.
    """
    gate = LLMGate(2)
    order: list[int] = []
    overlap = 0
    active = 0

    async def worker(index: int) -> None:
        nonlocal overlap, active
        async with gate:
            active += 1
            overlap = max(overlap, active)
            await asyncio.sleep(0.01)
            order.append(index)
            active -= 1

    await asyncio.gather(*(worker(i) for i in range(6)))

    assert sorted(order) == list(range(6)), "every worker must run"
    assert overlap <= 2, f"at most 2 turns may be in flight, saw {overlap}"
    assert gate.peak_concurrency == 2


async def test_the_gate_is_reusable_across_sequential_batches():
    gate = LLMGate(1)
    for _ in range(3):
        async with gate:
            await asyncio.sleep(0)
    assert gate.peak_concurrency == 1


async def test_a_rate_limited_call_is_retried_not_fatal():
    """One unlucky minute must not discard a run that already paid for planning."""
    from openai import RateLimitError

    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RateLimitError(
                "Rate limit reached ... Please try again in 0.1s.",
                response=httpx.Response(429, request=httpx.Request("POST", "https://x")),
                body=None,
            )
        return "recovered"

    result = await with_rate_limit_retry(flaky, attempts=3, base_delay=0.01)

    assert result == "recovered"
    assert calls == 3


async def test_a_rate_limit_gives_up_after_the_attempt_budget():
    from openai import RateLimitError

    calls = 0

    async def always_limited() -> str:
        nonlocal calls
        calls += 1
        raise RateLimitError(
            "still limited",
            response=httpx.Response(429, request=httpx.Request("POST", "https://x")),
            body=None,
        )

    with pytest.raises(RateLimitError):
        await with_rate_limit_retry(always_limited, attempts=2, base_delay=0.01)
    assert calls == 2


async def test_a_non_rate_limit_error_is_not_retried():
    """Retrying a 400 wastes the budget and hides the real cause."""
    calls = 0

    async def broken() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("the prompt is malformed")

    with pytest.raises(ValueError, match="malformed"):
        await with_rate_limit_retry(broken, attempts=3, base_delay=0.01)
    assert calls == 1


def test_the_retry_delay_honours_the_providers_own_suggestion():

    exc = RuntimeError("Rate limit reached ... Please try again in 15.9s.")
    assert _retry_delay(exc, attempt=0) == pytest.approx(16.9)

    # No hint from the provider: fall back to exponential backoff.
    assert _retry_delay(RuntimeError("nope"), attempt=2) == 4.0
    # Never wait absurdly long, however the provider phrases it.
    assert _retry_delay(RuntimeError("try again in 900s"), attempt=0) == 60.0


async def test_researchers_are_paced_by_the_gate(settings: Settings, monkeypatch):
    """End-to-end proof the fan-out respects the gate, not just the unit test."""
    from deep_research.workflow import ResearchDeps

    gate = LLMGate(2)
    deps = ResearchDeps(
        llm=FakeLLM(),
        research_agent=ScriptedResearchAgent(),
        searcher=WebSearcher(settings, client=StubTavilyClient()),
        settings=settings,
        bridge=None,  # type: ignore[arg-type]
        llm_gate=gate,
    )

    async def one(index: int) -> None:
        async with deps.llm_slot():
            await asyncio.sleep(0.01)

    await asyncio.gather(*(one(i) for i in range(8)))
    assert gate.peak_concurrency <= 2


async def test_deps_without_a_gate_still_work(settings: Settings):
    """An injected fake must not need a limiter to run the pipeline."""
    from deep_research.workflow import ResearchDeps

    deps = ResearchDeps(
        llm=FakeLLM(),
        research_agent=ScriptedResearchAgent(),
        searcher=WebSearcher(settings, client=StubTavilyClient()),
        settings=settings,
        bridge=None,  # type: ignore[arg-type]
    )
    async with deps.llm_slot():
        pass


def test_search_budget_covers_planned_review_rounds(settings: Settings):
    """A critic revision must not inherit an already-spent search allowance.

    Each revised researcher gets a fresh per-question allowance from `for_agent`,
    so the run-wide ceiling has to cover the first research pass plus every
    revision the settings allow. A live run with one review cycle billed its
    whole allowance in the first pass; the revision pass was refused every
    search and could not repair the draft the critic had just rejected.
    """
    from deep_research.tools.web_search import MAX_SEARCH_CALLS_PER_RUN, WebSearcher

    fresh = WebSearcher(
        make_settings(settings.runs_dir, max_review_cycles=0),
        client=StubTavilyClient(),
    )
    revised = WebSearcher(
        make_settings(settings.runs_dir, max_review_cycles=1),
        client=StubTavilyClient(),
    )
    one_round = MAX_SEARCH_CALLS_PER_RUN * settings.max_questions

    assert fresh.search_budget() == one_round
    assert revised.search_budget() == 2 * one_round


async def test_the_spent_search_budget_tells_the_model_to_stop(settings: Settings):
    """The exhausted-budget reply must end the loop, not invite another call.

    Regression: the message used to read "Answer now using the sources already
    gathered". A researcher would call the tool again, get the same sentence,
    and spin until LlamaIndex aborted the whole run. Wording matters here, so
    the instruction is asserted rather than trusted.
    """
    from deep_research.tools.web_search import WebSearcher

    searcher = WebSearcher(settings, client=StubTavilyClient())
    allowed = searcher.search_budget()
    # Distinct queries, because a repeat of the same query is a cache hit and
    # costs the provider nothing - it no longer counts against the budget.
    for i in range(allowed + 2):
        await searcher.search(f"gearbox shock load {i}")

    # Through the public entry point: the budget check deliberately lives in
    # `search()` rather than in the cached producer, so calling the producer
    # directly would bypass it entirely.
    reply = await searcher.search("one more")

    assert "budget spent" in reply
    assert "Do not call this tool again" in reply
    # The provider was billed at most the allowance; the rest were refusals.
    assert searcher.calls <= allowed


def _groq_429(message: str) -> RateLimitError:
    return RateLimitError(
        message,
        response=httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com")),
        body=None,
    )


def test_a_daily_budget_is_not_retried():
    """The day's tokens are gone, so three retries in four seconds are wasted.

    A live run burned its whole 900-second budget on this: the retry helper saw
    a 429, waited 1s then 2s, failed, and the run limped on into a second
    research round that could not succeed either.
    """

    exc = _groq_429(
        "Rate limit reached for model `qwen/qwen3.8-27b` service tier `on_demand` "
        "on tokens per day (TPD): Limit 200000, Used 199383, Requested 1566. "
        "Please try again in 6m49.968s."
    )
    assert is_quota_error(exc), "it is still a quota error, so callers may degrade"
    assert not is_retryable(exc), "but waiting four seconds cannot help"


def test_a_minute_budget_is_still_retried():

    exc = _groq_429(
        "Rate limit reached for model `qwen/qwen3.8-27b` service tier `on_demand` "
        "on tokens per minute (TPM): Limit 7000, Requested 8308. "
        "Please try again in 6.95s."
    )
    assert is_retryable(exc), "a per-minute window reopens in seconds"


def test_an_oversized_request_is_neither_retried_nor_treated_as_transient():

    exc = _groq_429(
        "Request too large for model `qwen/qwen3.8-27b` on tokens per request "
        "(TPR): Limit 8192, Requested 8308"
    )
    assert is_quota_error(exc)
    assert not is_retryable(exc), "the next request will be just as large"


async def test_a_daily_budget_fails_fast_instead_of_retrying():
    """The retry helper must not sit in a sleep loop on a dead day's quota."""
    attempts = {"n": 0}

    async def always_denied():
        attempts["n"] += 1
        raise _groq_429(
            "Rate limit reached ... on tokens per day (TPD): Limit 200000, "
            "Used 199383, Requested 1566. Please try again in 6m49.968s."
        )

    with pytest.raises(RateLimitError):
        await with_rate_limit_retry(always_denied, base_delay=0.0)

    assert attempts["n"] == 1, "a dead daily budget should not be retried"


def test_truncating_a_note_keeps_its_links():
    """Regression: a hard cut stripped the sources and the report lost its citations.

    A live run produced a 5.7k-character briefing with zero links in it, because
    `MAX_NOTE_CHARS` cut each note mid-source-list and the writer had nothing
    left to cite.
    """
    from deep_research.workflow import MAX_NOTE_CHARS, _truncate_notes

    note = (
        "A long analysis. " * 200
        + " See [the manufacturer](https://example.com/a) and "
        + "[a standard](https://example.org/b) and [a paper](https://example.net/c)."
    )
    assert len(note) > MAX_NOTE_CHARS, "the fixture must actually be long enough to truncate"

    trimmed = _truncate_notes(note)

    assert len(trimmed) > MAX_NOTE_CHARS, "the source list adds to the length"
    for url in ("https://example.com/a", "https://example.org/b", "https://example.net/c"):
        assert url in trimmed, f"{url} was lost when the note was truncated"


def test_a_short_note_is_untouched():
    from deep_research.workflow import _truncate_notes

    note = "Short note with [a source](https://example.com/x)."
    assert _truncate_notes(note) == note


async def test_an_unreachable_critic_is_not_recorded_as_approval(monkeypatch, settings: Settings):
    """The silent-approval trap, closed.

    `ask_structured` returns None when the model cannot be reached. Treating that
    as "acceptable" shipped a report whose metadata claimed a review that never
    happened; a live run did exactly that when the verdict call died on an
    output-token limit.
    """
    from deep_research.runner import ResearchRunner

    llm = FakeLLM(plans=[ResearchPlan(questions=["What is the service factor?"])])

    # The critic alone is unreachable; planning and writing still work.
    real_structured = FakeLLM.astructured_predict

    async def unreachable_critic(self: Any, output_cls: type, prompt: Any, **kwargs: Any) -> Any:
        if output_cls is ReviewVerdict:
            raise RuntimeError("provider is down")
        return await real_structured(self, output_cls, prompt, **kwargs)

    monkeypatch.setattr(FakeLLM, "astructured_predict", unreachable_critic)

    runner = ResearchRunner(
        "shock loading",
        settings,
        llm=llm,
        searcher=WebSearcher(settings, client=StubTavilyClient()),
        research_agent=ScriptedResearchAgent(),
    )

    events = [event async for event in runner.run()]
    kinds = [event.kind for event in events]
    assert "report_done" in kinds, "the report should still be delivered"

    result = runner.result
    assert result is not None
    assert result.meta.verdict == "critic_unavailable", (
        f"an unreadable verdict must not be recorded as approval, got {result.meta.verdict!r}"
    )
    assert result.meta.review_cycles == 0

    logs = [str(event.data.get("msg", "")) for event in events if event.kind == "log"]
    assert any("unverified" in message.lower() for message in logs), (
        "the run should say out loud that the report was not reviewed"
    )


async def test_the_offline_fake_enforces_the_real_prompt_contract():
    """Guards the guard.

    `FakeLLM` rejecting a bare `str` is what makes the rest of the suite capable
    of catching the `astructured_predict` bug. If someone relaxes the fake, the
    suite goes quietly blind to that class of failure, and the only remaining
    signal is a live run. So the fake's own strictness is asserted directly.
    """
    llm = FakeLLM()

    with pytest.raises(TypeError, match="PromptTemplate"):
        await llm.astructured_predict(ResearchPlan, "a bare string, not a template")


async def test_structured_output_uses_a_prompt_template():
    """Regression: a bare `str` made every live plan and verdict fail silently.

    `astructured_predict` takes a `PromptTemplate`. Passing a string throws
    inside the call, which `ask_structured` catches, so the run continued on the
    lenient fallback and a critic that could not be parsed was read as approval.
    The fake now rejects strings, so this fails if the wrapping is ever removed.
    """
    prompt = PLANNER_PROMPT.format(topic="gear wear", min_questions=1, max_questions=1)
    llm = FakeLLM(plans=[ResearchPlan(questions=["Why do gears wear?"])])

    plan = await ask_structured(llm, ResearchPlan, prompt)

    assert plan is not None
    assert plan.questions == ["Why do gears wear?"]
    assert llm.plans_seen == [prompt], "the prompt text must survive the template wrapper"


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


def test_build_llm_disables_the_sdk_retry_loop():
    """The SDK must not retry underneath our own rate-limit policy.

    LlamaIndex wraps every chat call in `llm_retry_decorator`, which retries
    *any* 429 up to `max_retries` times with up to 20s of backoff. It cannot
    tell a transient per-minute limit from a dead daily budget, so on an
    exhausted day it slept through the whole retry budget - "Retrying
    llama_index.llms.openai.base.OpenAI._achat in 120 seconds ... on tokens per
    day (TPD): Limit 200000" - before handing the error up. That defeats
    `with_rate_limit_retry`, which exists to fail fast on a long window so the
    caller can degrade to something useful. `max_retries <= 0` makes the
    decorator call through untouched, leaving one place that decides to wait.
    """
    settings = Settings(groq_api_key="k", _env_file=None)

    assert build_llm(settings).max_retries == 0

    # And the decorator really does become a pass-through at zero.
    import llama_index.llms.openai.base as sdk

    calls: list[int] = []

    class _Llm:
        max_retries = 0

        @sdk.llm_retry_decorator
        def _achat(self) -> str:
            calls.append(1)
            return "ok"

    assert _Llm()._achat() == "ok"
    assert calls == [1], "a single attempt, with no retry wrapper in between"


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


# -- serper -----------------------------------------------------------------
def test_serper_results_are_rendered_from_organic():
    """Serper calls them `organic` and links them under `link`, not `results`/`url`."""
    payload = {
        "searchParameters": {"q": "gearboxes"},
        "credits": 1,
        "organic": [
            {
                "title": "Load Distribution",
                "link": "https://sgrgear.example/a",
                "snippet": "The load is shared.",
                "date": "Mar 15, 2026",
                "position": 1,
            }
        ],
    }

    text = WebSearcher.format_serper_results(payload)

    assert "SOURCES:" in text
    assert "- [Load Distribution](https://sgrgear.example/a) (Mar 15, 2026):" in text
    assert "The load is shared." in text
    # Serper has no synthesised answer, so there must not be an empty ANSWER line.
    assert not text.startswith("ANSWER")


def test_serper_results_truncate_long_snippets():
    payload = {"organic": [{"title": "T", "link": "u", "snippet": "x " * 400}]}

    assert "[...]" in WebSearcher.format_serper_results(payload)


def test_serper_results_drop_hits_with_no_link():
    """A hit that cannot be cited is not worth the tokens."""
    payload = {
        "organic": [
            {"title": "No link", "snippet": "s"},
            {"title": "Linked", "link": "https://ok.example", "snippet": "s"},
        ]
    }

    text = WebSearcher.format_serper_results(payload)

    assert "No link" not in text
    assert "Linked" in text


def test_serper_results_of_an_empty_payload():
    assert WebSearcher.format_serper_results({}) == "No results found."


def test_serper_results_collapses_newlines():
    text = WebSearcher.format_serper_results(
        {"organic": [{"title": "T", "link": "u", "snippet": "a\n\nb   c"}]}
    )

    assert "a b c" in text


def test_serper_is_built_when_selected():
    settings = Settings(
        groq_api_key="k", serper_api_key="s", search_provider="serper", _env_file=None
    )
    searcher = WebSearcher(settings)

    assert isinstance(searcher.client, SerperClient)
    assert searcher.client.api_key == "s"
    assert searcher.client.base_url == "https://google.serper.dev"


def test_the_serper_base_url_is_overridable():
    """A proxy or self-hosted instance should not need a code change."""
    settings = Settings(
        groq_api_key="k",
        serper_api_key="s",
        serper_base_url="https://proxy.internal/serper/",
        search_provider="serper",
        _env_file=None,
    )

    assert WebSearcher(settings).client.base_url == "https://proxy.internal/serper"


def test_tavily_is_built_when_selected():
    settings = Settings(groq_api_key="k", tavily_api_key="t", _env_file=None)

    assert WebSearcher(settings).client is not None
    assert not isinstance(WebSearcher(settings).client, SerperClient)


def test_no_client_is_built_when_search_is_disabled():
    settings = Settings(groq_api_key="k", search_provider="none", _env_file=None)

    assert WebSearcher(settings).client is None


async def test_a_serper_search_passes_the_result_limit(tmp_path: Path):
    settings = make_settings(tmp_path, search_provider="serper", search_max_results=7)
    client = StubSerperClient()
    searcher = WebSearcher(settings, client=client)

    await searcher.search("gearboxes")

    assert client.max_results == [7]


async def test_the_cache_does_not_leak_results_across_providers(tmp_path: Path):
    """Same query, different provider: a Tavily answer must not satisfy Serper."""
    tavily_settings = make_settings(tmp_path, search_provider="tavily")
    serper_settings = make_settings(tmp_path, search_provider="serper")
    # One shared cache file, which is what makes this a real risk.
    shared = tavily_settings.cache_file

    tavily_client = StubTavilyClient(answer="Tavily says this.")
    serper_client = StubSerperClient(
        organic=[{"title": "S", "link": "https://s.example", "snippet": "Serper says this."}]
    )

    first = await WebSearcher(
        tavily_settings, client=tavily_client, cache=SearchCache(shared)
    ).search("same query")
    second = await WebSearcher(
        serper_settings, client=serper_client, cache=SearchCache(shared)
    ).search("same query")

    assert "Tavily says this." in first
    assert "Serper says this." in second
    assert len(tavily_client.calls) == 1, "the Tavily call should not have been reused"
    assert len(serper_client.calls) == 1


async def test_the_cache_does_reuse_a_repeated_query_within_a_provider(tmp_path: Path):
    """The flip side: the cache must still work for the provider that filled it."""
    settings = make_settings(tmp_path, search_provider="serper")
    client = StubSerperClient()
    searcher = WebSearcher(settings, client=client)

    await searcher.search("same query")
    await searcher.search("same query")

    assert len(client.calls) == 1, "the second identical query should hit the cache"
    # A served-from-cache query reaches no provider, so it must not consume the
    # per-run search budget either. Counting it meant a run could exhaust its
    # allowance without ever doing the research it was budgeted for.
    assert searcher.queries == 1
    assert searcher.calls == 1


async def test_the_spent_budget_message_is_never_cached(tmp_path: Path):
    """The "budget spent" reply is control flow, not a search result.

    It used to be produced inside `cache.wrap`, which persists whatever its
    producer returns. It was then served from disk - carrying the *previous*
    run's query count - to a different run that had never searched anything,
    for the full cache TTL and with zero billed calls, telling that run's
    researcher to stop and answer from sources it had never been given.
    """
    from deep_research.tools.web_search import WebSearcher

    shared = tmp_path / "cache.json"
    budget_spent = WebSearcher(make_settings(tmp_path), client=StubTavilyClient())
    allowed = budget_spent.search_budget()
    for i in range(allowed + 2):
        await budget_spent.search(f"gearbox shock load {i}")
    assert "budget spent" in await budget_spent.search("one more")

    # A brand new searcher, sharing the same on-disk cache, must still be able
    # to search normally.
    fresh = WebSearcher(
        make_settings(tmp_path), client=StubTavilyClient(), cache=SearchCache(shared)
    )
    result = await fresh.search("a question no run has asked before")

    assert "budget spent" not in result
    assert fresh.calls == 1, "the fresh run should have been billed a real search"


# -- the Serper HTTP client ------------------------------------------------
async def test_serper_client_posts_the_documented_request():
    """Guards the wire format: endpoint, header name, and body keys."""
    import httpx

    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("X-API-KEY")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"organic": [], "credits": 1})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = SerperClient(api_key="secret", http=http)

    await client.search("gearboxes", max_results=3)

    assert seen["url"] == "https://google.serper.dev/search"
    assert seen["key"] == "secret", "Serper authenticates with X-API-KEY, not a bearer token"
    assert seen["body"] == {"q": "gearboxes", "num": 3}
    await http.aclose()


async def test_serper_client_reports_a_rejected_key():
    import httpx

    http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(403, json={"message": "Unauthorized."})
        )
    )
    client = SerperClient(api_key="bad", http=http)

    with pytest.raises(SearchError, match="rejected the API key"):
        await client.search("q")
    await http.aclose()


async def test_serper_client_reports_a_rate_limit_distinctly():
    import httpx

    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(429, json={})))
    client = SerperClient(api_key="k", http=http)

    with pytest.raises(SearchError, match="rate limit"):
        await client.search("q")
    await http.aclose()


async def test_serper_client_reports_a_network_failure():
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = SerperClient(api_key="k", http=http)

    with pytest.raises(SearchError, match="serper request failed"):
        await client.search("q")
    await http.aclose()


async def test_serper_client_does_not_close_an_injected_pool():
    import httpx

    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    client = SerperClient(api_key="k", http=http)

    await client.aclose()

    assert not http.is_closed, "the caller owns a pool it injected"


async def test_a_failing_provider_surfaces_as_a_search_error(tmp_path: Path):
    settings = make_settings(tmp_path, search_provider="serper")
    searcher = WebSearcher(
        settings, client=StubSerperClient(fail_with=SearchError("serper is down"))
    )

    with pytest.raises(SearchError, match="serper is down"):
        await searcher.search("q")


def test_serper_requires_its_own_key(tmp_path):
    settings = make_settings(tmp_path, tavily_api_key="t", serper_api_key="")
    settings.search_provider = "serper"

    with pytest.raises(ValueError, match="SERPER_API_KEY"):
        settings.require_search_key()


def test_a_tavily_key_does_not_satisfy_serper(tmp_path):
    settings = make_settings(tmp_path, tavily_api_key="t", serper_api_key="")
    settings.search_provider = "serper"

    with pytest.raises(ValueError, match="SERPER_API_KEY"):
        settings.require_search_key()


def test_api_key_for_returns_the_right_key(tmp_path):
    settings = make_settings(tmp_path, tavily_api_key="t", serper_api_key="s")

    assert settings.api_key_for("tavily") == "t"
    assert settings.api_key_for("serper") == "s"
    assert settings.api_key_for("none") == ""


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


def test_a_finished_report_is_not_mistaken_for_a_truncated_one():
    """Ending on a bullet, bold marker or non-Latin glyph is *normal*.

    `_looks_truncated` used to end in an unconditional `return True`, so its
    verdict depended only on the three checks above it. Every report ending in
    a bullet, a closing `**`, a URL or a full-width character was then declared
    cut off, which sent the writer into a second provider call on a quota that
    might already be spent - and, before the `retry_stream` fix, crashed the
    run outright on a name error.
    """
    from deep_research.workflow import _looks_truncated

    finished = [
        "The reducer tolerates shock by *not* absorbing it.",
        "Findings:\n- planetary carriers share the load\n- helical backup takes the peak",
        "Overall the design is **adequate**",
        "See https://example.com/gearbox-spec for the full table",
        "总结：行星齿轮减速器可以吸收冲击载荷",  # noqa: RUF001 - the fullwidth colon is the point
        "1. First\n2. Second",
        "```\nplain block\n```",
    ]
    for text in finished:
        assert _looks_truncated(text) is False, f"wrongly called truncated: {text!r}"


def test_a_draft_cut_off_by_the_token_cap_is_detected():
    """Unclosed markdown is the shape-based signal; the bare word is not.

    A report stopping at "...18CrNiMo" is indistinguishable from one ending in a
    bullet as a matter of text shape, which is why the provider's
    `finish_reason="length"` - not this function - is the primary signal (see
    `test_the_provider_length_cap_is_the_authoritative_signal`). What is left
    here are the constructs that cannot be finished.
    """
    from deep_research.workflow import _looks_truncated

    cut = [
        "",
        "The casing is cast in **ductile iron and the pinion is",
        "```python\ndef load(self):\n    return self._",
        "## Recommendations\n- upgrade the bearings\n## Risks",
    ]
    for text in cut:
        assert _looks_truncated(text) is True, f"missed a cut: {text!r}"


def test_the_provider_length_cap_is_the_authoritative_signal():
    """`finish_reason == "length"` beats guessing from the text."""
    from deep_research.workflow import _hit_length_cap

    class _Choice:
        def __init__(self, reason: str) -> None:
            self.finish_reason = reason

    class _Raw:
        def __init__(self, *choices: object) -> None:
            self.choices = choices

    class _Chunk:
        def __init__(self, raw: object) -> None:
            self.raw = raw

    assert _hit_length_cap(_Chunk(_Raw(_Choice("length")))) is True
    assert _hit_length_cap(_Chunk(_Raw(_Choice("stop")))) is False
    # A chunk with no provider detail at all must not raise or claim a cut.
    assert _hit_length_cap(_Chunk(None)) is False
    assert _hit_length_cap(_Chunk(_Raw())) is False


def test_a_successful_research_exits_zero(monkeypatch, tmp_path: Path):
    """A finished report must not be reported to the shell as a failure.

    `typer.Exit` subclasses `RuntimeError`, and the success path used to
    `raise typer.Exit(code=...)` *inside* the `except (ValueError, RuntimeError)`
    try block. The CLI therefore caught its own success, printed an empty red
    line, and exited 1 - after writing a perfectly good report. Every `&&`
    chain, CI step or wrapper checking `$?` saw a crash.
    """
    from typer.testing import CliRunner

    async def fake_run(topic: str, settings: Settings) -> int:
        return 0

    monkeypatch.setattr(cli, "_run", fake_run)
    monkeypatch.setattr(cli, "_settings", lambda **_kwargs: make_settings(tmp_path))

    result = CliRunner().invoke(cli.app, ["research", "gearbox selection"])

    assert result.exit_code == 0, f"expected success, got {result.exit_code}\n{result.output}"


def test_a_failed_research_exits_one(monkeypatch, tmp_path: Path):
    """The counterpart: a genuine failure must still be non-zero."""

    from typer.testing import CliRunner

    async def fake_run(topic: str, settings: Settings) -> int:
        raise RuntimeError("the provider is unreachable")

    monkeypatch.setattr(cli, "_run", fake_run)
    monkeypatch.setattr(cli, "_settings", lambda **_kwargs: make_settings(tmp_path))

    result = CliRunner().invoke(cli.app, ["research", "gearbox selection"])

    assert result.exit_code == 1
    assert "the provider is unreachable" in result.output


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
