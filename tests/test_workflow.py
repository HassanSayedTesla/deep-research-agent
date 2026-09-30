"""End-to-end tests of the pipeline, driven entirely by fakes.

These are the tests that matter: they exercise the fan-out, the `collect_events`
barrier, the reflection loop and persistence, with no network and no API keys.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from openai import RateLimitError

from conftest import QUESTIONS, make_settings
from deep_research.config import Settings
from deep_research.events import (
    DONE,
    EDGE_FLOW,
    LOG,
    NODE_ACTIVATED,
    NODE_FINISHED,
    REPORT_DELTA,
    REPORT_DONE,
    RUN_FAILED,
    RUN_STARTED,
)
from deep_research.runner import ResearchRunner
from deep_research.schemas import ResearchPlan, ReviewVerdict
from deep_research.storage import RunStore
from deep_research.tools.web_search import WebSearcher
from fakes import (
    FakeLLM,
    ScriptedResearchAgent,
    StubSerperClient,
    StubTavilyClient,
    extract_question,
)
from helpers import drain, events_of, kinds


def make_runner(
    settings: Settings,
    llm: FakeLLM,
    agent: ScriptedResearchAgent,
    searcher: WebSearcher | None = None,
    topic: str = "gearbox selection for a mobile crane",
) -> ResearchRunner:
    return ResearchRunner(
        topic=topic,
        settings=settings,
        store=RunStore(settings.runs_dir),
        llm=llm,
        research_agent=agent,
        searcher=searcher,
    )


async def test_the_writer_survives_a_provider_outage(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent, monkeypatch
):
    """A dead writer must not discard the research that already succeeded.

    Regression from a live run: the researchers finished in 64s, then the writer
    hit "TPD: Limit 200000, Used 199383" and the run produced nothing at all.
    On a free tier that is a routine failure, so the report falls back to the
    notes themselves - assembled without any model call, which is the only thing
    that works when the quota is what failed.
    """
    calls = {"n": 0}

    async def quota_exhausted(prompt: str, **_: object):
        calls["n"] += 1
        raise RateLimitError(
            "Rate limit reached ... on tokens per day (TPD): Limit 200000, "
            "Used 199383, Requested 1566.",
            response=httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com")),
            body=None,
        )

    monkeypatch.setattr(llm, "astream_complete", quota_exhausted)

    events, result = await drain(make_runner(settings, llm, agent))

    assert result is not None, "a dead writer should not void the run"
    assert result.meta.status == "ok"
    # Assembled locally, so exactly one attempted call and no second chance.
    assert calls["n"] == 1
    # The notes survive, in order, under their own questions.
    for question in QUESTIONS:
        assert question in result.report
    assert "Unprocessed research notes" in result.report
    # And the reader is told what they are looking at.
    logs = [e.data.get("msg", "") for e in events_of(events, LOG)]
    assert any("ran out of provider quota" in msg for msg in logs)
    assert RUN_FAILED not in kinds(events)


async def test_a_writer_bug_still_fails_loudly(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent, monkeypatch
):
    """Only quota trouble is absorbed; a real bug must not be hidden.

    Swallowing every writer error would turn a typo into a mysteriously plain
    report, which is far harder to diagnose than a failed run.
    """

    async def broken(prompt: str, **_: object):
        raise TypeError("writer called with the wrong arguments")

    monkeypatch.setattr(llm, "astream_complete", broken)

    events, result = await drain(make_runner(settings, llm, agent))

    assert result is None
    assert RUN_FAILED in kinds(events)
    assert "wrong arguments" in events_of(events, RUN_FAILED)[0].data["error"]


async def test_a_degraded_draft_is_not_sent_back_for_revision(
    settings: Settings, agent: ScriptedResearchAgent
):
    """A revision cannot fix an exhausted quota, so the loop must stop.

    A live run fell back to the raw notes, let the critic reject them, and then
    spent the rest of its 900-second budget re-running research and the writer
    into the same wall. The critic's feedback is kept; the second round is not.
    """

    class NoWriterLLM(FakeLLM):
        async def astream_complete(self, prompt, **_):
            raise RateLimitError(
                "Rate limit reached ... on tokens per day (TPD): Limit 200000, "
                "Used 199986, Requested 953.",
                response=httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com")),
                body=None,
            )

    llm = NoWriterLLM(
        plans=[ResearchPlan(questions=list(QUESTIONS))],
        reviews=[ReviewVerdict(acceptable=False, feedback="Too thin; add citations.")],
    )
    events, result = await drain(make_runner(settings, llm, agent))

    assert result is not None
    assert result.meta.review_cycles == 0, "no revision round should have started"
    assert result.meta.verdict == "unverified"
    # The feedback is not thrown away; it is why this draft is marked unverified.
    assert result.meta.reviewer_feedback == "Too thin; add citations."
    # The research ran exactly once.
    assert len(agent.questions) == len(QUESTIONS)
    assert RUN_FAILED not in kinds(events)


async def test_one_failing_researcher_does_not_lose_the_run(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent
):
    """A dead researcher costs one finding, not the whole run.

    A live run lost 7 minutes of work because a single researcher raised and
    took the pipeline down with it, even though its sibling had already
    answered. The report is still worth producing from whatever survived.
    """
    agent.fail_on = {2}
    agent.fail_with = RuntimeError("provider exploded")

    events, result = await drain(make_runner(settings, llm, agent))

    assert result is not None, "the run should finish on partial findings"
    assert result.meta.status == "ok"
    assert len(result.meta.questions) == len(QUESTIONS)
    # The surviving answer is in the report's notes, the dead one is flagged.
    assert "the answer body" in llm.writer_prompts[0]
    assert "could not be researched" in llm.writer_prompts[0]
    # And the operator is told, rather than left to wonder why a question is thin.
    logs = [e.data.get("msg", "") for e in events_of(events, LOG)]
    assert any("continuing without it" in msg for msg in logs)
    assert RUN_FAILED not in kinds(events)


async def test_a_cancelled_researcher_is_not_reported_as_a_failed_one(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent
):
    """A cancellation must not be laundered into a fabricated finding.

    `_is_shutdown` tested for a `WorkflowCancelledError` class that does not
    exist - the real one is `WorkflowCancelledByUser` - so pressing "stop" was
    caught by the researcher-failure handler. The run carried on, logged
    "continuing without it", and handed the writer a made-up "could not be
    researched" finding standing in for a question the user chose to stop.
    """
    from workflows.errors import WorkflowCancelledByUser

    agent.fail_on = {2}
    agent.fail_with = WorkflowCancelledByUser("run was cancelled by the user")

    _events, result = await drain(make_runner(settings, llm, agent))

    assert result is None or not llm.writer_prompts, (
        "a cancelled run must not reach the writer with invented findings"
    )


async def test_a_truncated_draft_is_rewritten_at_a_larger_cap(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent, monkeypatch
):
    """A draft cut off by the output cap must be retried, and the retry must work.

    The retry built a fresh LLM and then iterated a `retry_stream` it never
    assigned, so the whole branch raised `NameError` - which the surrounding
    `except` re-raised, because a `NameError` is not a quota error. The run died
    after the planner, every researcher and every billed search, on the one path
    that was supposed to rescue it. `FakeLLM.report` ended in a full stop, so
    the branch was never reached and the suite stayed green.
    """
    import deep_research.workflow as workflow_mod
    from fakes import FakeLLM as _Fake

    # A report the provider cut off mid-word: the live run ended "...18CrNiMo".
    llm.report = "The casing is cast in **ductile iron and the pinion is 18CrNiMo"
    longer = "# Report\n\nA complete briefing that also carries its citations."
    llm.reviews = [ReviewVerdict(acceptable=True)]

    retry_llm = _Fake(plans=[], reviews=[], report=longer)
    built: list[object] = []

    def _fake_build(cfg):
        built.append(cfg)
        return retry_llm

    monkeypatch.setattr(workflow_mod, "build_llm", _fake_build)

    events, result = await drain(make_runner(settings, llm, agent))

    assert result is not None, "the retry path must not lose the run"
    assert RUN_FAILED not in kinds(events)
    assert built, "a truncated draft should have triggered one retry"
    # The retry asked for more room than the first attempt.
    assert built[0].max_output_tokens > settings.max_output_tokens
    # And the longer, complete draft is the one that ships.
    assert result.report.startswith("# Report")
    assert "18CrNiMo" not in result.report


async def test_a_planner_that_cannot_be_reached_is_reported(
    settings: Settings, agent: ScriptedResearchAgent
):
    """A dead planner must not look like a merely thinner report.

    `ask_structured` returns None on a quota refusal as well as on bad JSON, and
    the two are nothing alike to the user. The fallback to a single generic
    question therefore made an exhausted daily quota look like a successful run
    that happened to research one thing. The critic path already surfaced an
    unusable verdict this way; the planner must too.
    """

    class DeadPlanner(FakeLLM):
        async def astructured_predict(self, output_cls, prompt, **_):
            raise RateLimitError(
                "Rate limit reached ... on tokens per day (TPD): Limit 200000, Used 199383.",
                response=httpx.Response(429, request=httpx.Request("POST", "https://x")),
                body=None,
            )

        async def acomplete(self, prompt, **_):
            raise RateLimitError(
                "Rate limit reached ... on tokens per day (TPD): Limit 200000, Used 199383.",
                response=httpx.Response(429, request=httpx.Request("POST", "https://x")),
                body=None,
            )

    llm = DeadPlanner(reviews=[ReviewVerdict(acceptable=True)])
    events, result = await drain(make_runner(settings, llm, agent))

    assert result is not None
    logs = [e.data.get("msg", "") for e in events_of(events, LOG)]
    assert any("planner could not be reached" in msg for msg in logs), (
        f"the operator must be told the planner failed; logs were {logs!r}"
    )


async def test_a_consumer_stopping_at_done_does_not_get_a_cancellation(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent, monkeypatch
):
    """`run()` must end cleanly when the event stream ends before the task does.

    `bridge.close()` is the *last* thing `_execute` does, so a consumer woken by
    it can reach `run()`'s `finally` before the task has actually returned. The
    task is then not done, gets cancelled, and the bare `await task` that
    followed re-raised `CancelledError` - a `BaseException`, so it escaped
    `run()` entirely and replaced a fully populated, already-persisted result
    with a traceback. The CLI caught only `ValueError`/`RuntimeError`.

    Gating `aclose` parks the task inside its own cleanup, so the ordering that
    triggers this is deterministic instead of a rare flake.
    """
    from deep_research.tools.web_search import WebSearcher

    gate = asyncio.Event()

    async def _blocked_aclose(self) -> None:
        await gate.wait()

    monkeypatch.setattr(WebSearcher, "aclose", _blocked_aclose)
    runner = make_runner(settings, llm, agent)

    seen: list[str] = []
    stream = runner.run()
    while True:
        try:
            event = await stream.__anext__()
        except StopAsyncIteration:
            break
        seen.append(event.kind)
        if event.kind == DONE:
            # The task is parked in `aclose` and cannot return on its own. Close
            # the queue, which is the last thing it would have done anyway.
            await runner.bridge.close()

    gate.set()

    assert seen[-1] == DONE
    assert runner.result is not None, "the result should still be there"
    assert runner.result.meta.status == "ok"


async def test_the_writer_waits_out_a_short_rate_limit(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent
):
    """A six-second rate limit must be waited out, not degraded away.

    `_stream_draft` was the one provider call in the pipeline not routed through
    `with_rate_limit_retry`. A live run was refused with
    `OTPM: Limit 1000 ... Please try again in 5.94s` and the writer treated it
    as terminal, so the run degraded to raw research notes - throwing away a
    minute of research over a pause shorter than the cost of reading this
    comment.
    """
    calls = {"n": 0}
    real_stream = llm.astream_complete

    async def flaky_stream(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitError(
                "Rate limit reached ... on output tokens per minute (OTPM): "
                "Limit 1000, Used 997, Requested 102. Please try again in 5.94s.",
                response=httpx.Response(429, request=httpx.Request("POST", "https://x")),
                body=None,
            )
        return await real_stream(prompt)

    llm.astream_complete = flaky_stream  # type: ignore[method-assign]
    events, result = await drain(make_runner(settings, llm, agent))

    assert result is not None
    assert calls["n"] == 2, "the writer should have waited and tried again"
    # A real written briefing, not the degraded notes banner.
    assert "Unprocessed research notes" not in result.report
    assert RUN_FAILED not in kinds(events)


async def test_a_shared_search_budget_cannot_starve_a_researcher(settings: Settings):
    """Each researcher gets its own search allowance, so none can starve another.

    The budget used to be a single pool shared by every researcher, sized from
    the shared `RESEARCHER_MAX_ITERATIONS`. In a live run a researcher that
    looped to its cap spent the whole pool, and its sibling was told the budget
    was gone having retrieved nothing - so it wrote a finding from memory and
    the writer cited it as research.
    """
    from deep_research.tools.web_search import MAX_SEARCH_CALLS_PER_RUN, WebSearcher

    root = WebSearcher(settings, client=StubTavilyClient())
    greedy = root.for_agent()
    sibling = root.for_agent()

    # The greedy researcher burns its entire allowance.
    for i in range(MAX_SEARCH_CALLS_PER_RUN):
        await greedy.search(f"greedy {i}")
    assert "budget spent" in await greedy.search("one more")

    # The sibling is untouched: it still has its full allowance.
    assert sibling.queries == 0
    for i in range(MAX_SEARCH_CALLS_PER_RUN):
        result = await sibling.search(f"sibling {i}")
        assert "budget spent" not in result, "a sibling's searching must not be cut short"

    # But the run-wide ceiling still holds, so spend is bounded however the
    # planner behaves.
    assert root.calls <= root.search_budget()
    # And the allowance is per researcher, not per query.
    assert greedy.agent_budget() == MAX_SEARCH_CALLS_PER_RUN
    assert sibling.agent_budget() == MAX_SEARCH_CALLS_PER_RUN


async def test_the_search_allowance_is_shared_state_not_a_copy(settings: Settings):
    """`for_agent` must not duplicate the provider client, cache or billing.

    A per-researcher allowance is only affordable if it is a *view*: copying the
    searcher per question would open a second HTTP pool per researcher and reset
    the run's billed-call count, so the run-wide ceiling would never be reached.
    """
    from deep_research.tools.web_search import WebSearcher

    root = WebSearcher(settings, client=StubTavilyClient())
    view = root.for_agent()

    assert view is not root
    assert view.client is root.client, "a second HTTP client per researcher"
    assert view.cache is root.cache, "a second cache per researcher"

    await view.search("shared")
    assert root.calls == 1, "the run must see what its researchers spent"
    assert view.calls == 1


async def _cut_off_stream(text: str):
    """A stream that emits `text`, then fails the way a mid-stream 429 does."""

    class _Cuts:
        def __init__(self) -> None:
            self.step = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.step += 1
            if self.step == 1:
                return _Delta(text)
            raise RateLimitError(
                "Rate limit reached ... on output tokens per minute (OTPM): "
                "Limit 1000, Used 998, Requested 90. Please try again in 4s.",
                response=httpx.Response(429, request=httpx.Request("POST", "https://x")),
                body=None,
            )

    return _Cuts()


class _Delta:
    """The slice of a streamed chunk `_stream_draft` actually reads."""

    def __init__(self, text: str) -> None:
        self.delta = text
        self.raw = None


async def test_a_writer_stream_cut_off_midway_keeps_what_reached_the_browser(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent
):
    """A partial draft is a result, not a failure to throw away.

    Those tokens were already streamed to the browser. Discarding them and
    raising means the user watches a report being written and then gets raw
    notes or an error instead.
    """
    attempts = {"n": 0}

    async def cutting_stream(prompt):
        attempts["n"] += 1
        return await _cut_off_stream("Half a report that stops")

    llm.astream_complete = cutting_stream  # type: ignore[method-assign]
    events, result = await drain(make_runner(settings, llm, agent))

    assert result is not None
    assert attempts["n"] == 1, "a refusal after tokens were sent must not be retried"
    assert "Half a report that stops" in result.report, (
        "text already streamed to the user was discarded"
    )
    assert RUN_FAILED not in kinds(events)


async def test_the_writer_does_not_duplicate_a_report_it_already_streamed(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent
):
    """A retry mid-stream would append a second draft to the first."""
    attempts = {"n": 0}
    streams = iter(
        [
            "First attempt, already on screen",
            "Second attempt, which must never happen",
        ]
    )

    async def cutting_stream(prompt):
        attempts["n"] += 1
        return await _cut_off_stream(next(streams))

    llm.astream_complete = cutting_stream  # type: ignore[method-assign]
    _events, result = await drain(make_runner(settings, llm, agent))

    assert result is not None
    assert attempts["n"] == 1, f"the stream was attempted {attempts['n']} times"
    assert "Second attempt" not in result.report, "a retried stream duplicated the report"


async def test_a_failed_researcher_does_not_put_the_provider_error_in_the_report(
    settings: Settings, llm: FakeLLM
):
    """A failed researcher's note reaches the reader; the error body must not.

    The note becomes a finding, so it lands in the finished document. A live run
    put the provider's entire error there:

        This question could not be researched: Error code: 429 - {'error':
        {'message': 'Rate limit reached for model `qwen/qwen3.8-27b` in
        organization `org_01ksmg...' on input tokens per minute (ITPM): ...

    The reader needs to know the question went unanswered and that nothing in it
    is verified. The quota detail belongs in the log.
    """
    agent = ScriptedResearchAgent(
        # A long-window limit, which `with_rate_limit_retry` deliberately does not
        # retry. A short-window one would simply be waited out and never reach the
        # failure path this test is about.
        fail_on={2},
        fail_with=RateLimitError(
            "Rate limit reached for model `qwen/qwen3.8-27b` in organization "
            "`org_01ksmgvpe2fxna84nt5b315tkm` service tier `on_demand` on tokens per "
            "day (TPD): Limit 200000, Used 199601. Please try again in 12h.",
            response=httpx.Response(429, request=httpx.Request("POST", "https://x")),
            body=None,
        ),
    )

    # Degrade the writer, so the report is assembled from the findings verbatim.
    # That is the path where a raw provider error reaches the reader: a live run
    # put the whole body in the finished document.
    async def no_writer(prompt):
        raise RateLimitError(
            "Rate limit reached ... on tokens per day (TPD): Limit 200000, Used 199601.",
            response=httpx.Response(429, request=httpx.Request("POST", "https://x")),
            body=None,
        )

    llm.astream_complete = no_writer  # type: ignore[method-assign]
    events, result = await drain(make_runner(settings, llm, agent))

    assert result is not None
    # Findings are in the report, so this is where a leak would be visible.
    assert "Finding for question" in result.report, result.report
    # The gap is disclosed, and the point is marked unverified.
    assert "could not be researched" in result.report, result.report
    assert "unverified" in result.report.lower()

    # None of the provider's internals reach the reader. A live run's report
    # contained the whole thing, org id and all.
    for leak in (
        "Error code: 429",
        "{'error'",
        "'message'",
        "org_01",
        "service tier",
    ):
        assert leak not in result.report, f"the report leaked {leak!r}"

    # The run is still a success, and the operator is still told.
    assert RUN_FAILED not in kinds(events)
    logs = [e.data.get("msg", "") for e in events_of(events, LOG)]
    assert any("Researcher 2 failed" in msg for msg in logs), f"logs were {logs!r}"


async def test_the_writer_never_asks_for_more_than_the_provider_will_grant(
    settings: Settings,
):
    """A truncation retry must request a reachable allowance.

    Tripling the cap looked generous and was impossible on a free tier: a live run
    set 512, the retry asked for 1536, and Groq refused with
    `429 Request too large` - the request itself is over the per-minute ceiling,
    so waiting can never help. That turned a recoverable truncation into an
    unconditional fallback to raw notes.
    """
    from deep_research.workflow import _larger_draft_budget

    free = make_settings(settings.runs_dir, max_output_tokens=512, max_output_ceiling=1000)
    bigger = _larger_draft_budget(free)
    assert bigger > free.max_output_tokens, "a retry must actually ask for more room"
    assert bigger <= free.max_output_ceiling, "it must not ask for the impossible"

    paid = make_settings(settings.runs_dir, max_output_tokens=512, max_output_ceiling=8192)
    assert _larger_draft_budget(paid) == 1536, "a paid tier should get the full increase"

    tight = make_settings(settings.runs_dir, max_output_tokens=512, max_output_ceiling=512)
    assert _larger_draft_budget(tight) == 512, "never ask for more than is already used"


async def test_the_spent_budget_reply_does_not_claim_you_have_sources(settings: Settings):
    """The degradation message must be true, or the finding is fiction.

    A live run spent its whole pool and the reply still said "using the sources
    already gathered". A researcher that had retrieved nothing then answered at
    length from memory, and the writer received that as a research finding.
    """
    from deep_research.tools.web_search import WebSearcher
    from fakes import StubTavilyClient

    searcher = WebSearcher(settings, client=StubTavilyClient())
    for i in range(searcher.search_budget()):
        await searcher.search(f"query {i}")

    spent = await searcher.search("one more")

    assert "budget spent" in spent
    assert "Do not call this tool again" in spent

    # The same message, whatever the state. The counter is run-wide and the tool
    # cannot see the calling agent's history, so it must not claim that sources
    # exist - a wrong claim here is how an unverified answer becomes a cited
    # finding. A live run did exactly that.
    blind = WebSearcher(settings, client=StubTavilyClient())
    for i in range(blind.search_budget()):
        await blind.search(f"fill the pool {i}")
    spent_before = blind.calls
    nothing = await blind.search("anything")

    assert "budget spent" in nothing
    assert "already gathered" not in nothing, (
        "claiming sources the researcher never had is how an unverified answer "
        "becomes a cited finding"
    )
    assert "contains none" in nothing, "the no-sources case must be spelled out"
    assert blind.calls == spent_before, "a refused search must not be billed"


async def test_researchers_are_given_a_bounded_agent_loop(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent
):
    """Each researcher must be capped, and capped with a generated stop.

    Unbounded, a researcher whose search budget is spent keeps calling the tool
    and LlamaIndex aborts the run with "Max iterations of 20 reached".
    `early_stopping_method="generate"` is what turns that abort into a written
    answer from the sources gathered so far.
    """
    await drain(make_runner(settings, llm, agent))

    assert agent.run_kwargs, "the agent was never called"
    for kwargs in agent.run_kwargs:
        assert kwargs["max_iterations"] == settings.researcher_max_iterations
        assert kwargs["early_stopping_method"] == "generate"


async def test_happy_path_produces_and_persists_a_report(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent
):
    runner = make_runner(settings, llm, agent)
    events, result = await drain(runner)

    assert result is not None
    assert result.report == llm.report
    assert result.meta.verdict == "acceptable"
    assert result.meta.status == "ok"
    assert result.meta.review_cycles == 0

    store = RunStore(settings.runs_dir)
    assert store.exists(runner.run_id)
    # save_report normalises the trailing newline; the in-memory result is raw.
    assert store.load_report(runner.run_id) == llm.report + "\n"
    # The archived log is the whole stream, terminal events included, so a run
    # can be replayed or debugged without re-spending tokens.
    assert [event["kind"] for event in store.load_events(runner.run_id)] == kinds(events)
    assert kinds(events)[-1] == DONE
    assert REPORT_DONE in kinds(events)


async def test_planner_questions_become_one_researcher_each(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent
):
    _, result = await drain(make_runner(settings, llm, agent))

    assert len(agent.questions) == len(QUESTIONS)
    assert result is not None
    assert result.meta.questions == QUESTIONS


async def test_findings_reach_the_writer_in_question_order(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent
):
    await drain(make_runner(settings, llm, agent))

    writer_prompt = llm.writer_prompts[0]
    positions = [writer_prompt.index(f"Question {i + 1}: {q}") for i, q in enumerate(QUESTIONS)]
    assert positions == sorted(positions), "writer prompt is not in question order"
    # Every finding body made it into the prompt, none were dropped by the barrier.
    for i in range(1, len(QUESTIONS) + 1):
        assert f"Question {i}:" in writer_prompt


async def test_research_fans_out_concurrently(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent
):
    agent.delay = 0.02
    await drain(make_runner(settings, llm, agent))

    assert agent.max_concurrent > 1, "researchers ran serially"


async def test_critic_feedback_loops_back_through_the_planner(
    settings: Settings, agent: ScriptedResearchAgent
):
    llm = FakeLLM(
        plans=[
            ResearchPlan(questions=list(QUESTIONS)),
            ResearchPlan(questions=["What is the service life of the gearbox?"]),
        ],
        reviews=[
            ReviewVerdict(acceptable=False, feedback="Nothing on maintenance intervals."),
            ReviewVerdict(acceptable=True),
        ],
    )
    events, result = await drain(make_runner(settings, llm, agent))

    assert result is not None
    assert result.meta.review_cycles == 1
    assert result.meta.verdict == "acceptable"
    assert result.meta.reviewer_feedback == "Nothing on maintenance intervals."

    # Re-planned: the second planning prompt carries the feedback.
    assert len(llm.plans_seen) == 2
    assert "Nothing on maintenance intervals." in llm.plans_seen[1]
    # Both rounds researched, the second round the extra question only.
    assert len(agent.questions) == len(QUESTIONS) + 1
    # The loop-back edge is on the wire.
    loop_edges = [
        event
        for event in events_of(events, EDGE_FLOW)
        if event.data["source"] == "critic" and event.data["target"] == "planner"
    ]
    assert loop_edges


async def test_review_cycles_are_capped(settings: Settings, agent: ScriptedResearchAgent):
    capped = settings.model_copy(update={"max_review_cycles": 1})
    llm = FakeLLM(
        plans=[ResearchPlan(questions=list(QUESTIONS))],
        reviews=[ReviewVerdict(acceptable=False, feedback="Still thin.")],
    )
    _, result = await drain(make_runner(capped, llm, agent))

    assert result is not None
    # The critic never accepted; the cap stopped the loop instead.
    assert result.meta.verdict == "cycle_limit"
    assert result.meta.review_cycles == 1
    assert result.report == llm.report


async def test_no_review_cycles_accepts_immediately(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent
):
    no_review = settings.model_copy(update={"max_review_cycles": 0})
    always_reject = FakeLLM(
        plans=[ResearchPlan(questions=list(QUESTIONS))],
        reviews=[ReviewVerdict(acceptable=False, feedback="Nope.")],
    )
    _, result = await drain(make_runner(no_review, always_reject, agent))

    assert result is not None
    assert result.meta.review_cycles == 0
    assert result.meta.verdict == "cycle_limit"


async def test_event_stream_is_well_formed(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent
):
    events, _ = await drain(make_runner(settings, llm, agent))
    sequence = kinds(events)

    assert sequence[0] == RUN_STARTED
    assert sequence[-1] == DONE
    assert RUN_FAILED not in sequence
    assert sequence.count(RUN_STARTED) == 1

    # The graph announcement carries the topology the UI draws.
    started = events[0].data
    assert {n["id"] for n in started["graph"]["nodes"]} == {
        "planner",
        "research",
        "writer",
        "critic",
    }
    assert started["config"]["model"] == settings.model

    # Every worker node announced both its arrival and its departure.
    activated = [e.data["node"] for e in events_of(events, NODE_ACTIVATED)]
    finished = [e.data["node"] for e in events_of(events, NODE_FINISHED)]
    for index in range(len(QUESTIONS)):
        assert f"research#{index}" in activated
        assert f"research#{index}" in finished

    # The report was streamed token by token and then delivered whole.
    deltas = events_of(events, REPORT_DELTA)
    assert "".join(e.data["text"] for e in deltas) == llm.report
    done = events_of(events, REPORT_DONE)[0].data
    assert done["markdown"] == llm.report
    assert done["report_path"].endswith("report.md")


async def test_a_second_run_does_not_reuse_the_first_context(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent
):
    first, _ = await drain(make_runner(settings, llm, agent))
    second, result2 = await drain(make_runner(settings, llm, agent))

    assert kinds(first)[0] == RUN_STARTED
    assert kinds(second)[0] == RUN_STARTED
    assert result2 is not None
    assert result2.meta.questions == QUESTIONS


async def test_consumer_leaving_early_cancels_the_run(
    settings: Settings, llm: FakeLLM, agent: ScriptedResearchAgent
):
    runner = make_runner(settings, llm, agent)
    seen = 0
    async for _ in runner.run():
        seen += 1
        if seen == 2:
            break

    assert seen == 2
    await asyncio.sleep(0.05)
    assert runner.result is None, "the run kept going after the consumer walked away"


async def test_search_results_reach_the_researcher(settings: Settings, llm: FakeLLM):
    searcher = WebSearcher(settings, client=StubTavilyClient(answer="Gearboxes are mechanical."))
    # Exercise the tool directly: the researcher agent is faked in these tests.
    result = await searcher.search("how gearboxes work")

    assert "Gearboxes are mechanical." in result
    assert "https://example.com" in result
    assert searcher.calls == 1

    # Second identical query is served from the cache, not from Tavily.
    again = await searcher.search("how gearboxes work")
    assert again == result
    assert searcher.calls == 1
    assert searcher.cache.stats()["hits"] == 1


async def test_a_serper_backed_run_produces_a_report(tmp_path):
    """The whole pipeline on the second provider, not just the parsing."""
    from conftest import make_settings

    settings = make_settings(tmp_path, search_provider="serper")
    serper = StubSerperClient(
        organic=[
            {
                "title": "Gearbox basics",
                "link": "https://example.com/gearbox",
                "snippet": "Gears trade speed for torque.",
                "date": "Mar 15, 2026",
            }
        ]
    )
    searcher = WebSearcher(settings, client=serper)
    llm = FakeLLM(
        plans=[ResearchPlan(questions=list(QUESTIONS))],
        reviews=[ReviewVerdict(acceptable=True)],
    )
    runner = ResearchRunner(
        topic="gearbox selection",
        settings=settings,
        store=RunStore(settings.runs_dir),
        llm=llm,
        research_agent=ScriptedResearchAgent(),
        searcher=searcher,
    )

    events, result = await drain(runner)

    assert result is not None
    assert result.meta.status == "ok"
    assert result.meta.search_calls == 0, "the faked agent never calls the tool"
    assert result.meta.cache == searcher.cache.stats()
    assert DONE in kinds(events)


async def test_the_reported_provider_reaches_the_ui(settings: Settings, llm: FakeLLM):
    """The UI shows which provider ran, so it has to be in the run config."""
    serper_settings = Settings(
        groq_api_key="k",
        serper_api_key="s",
        search_provider="serper",
        runs_dir=settings.runs_dir,
        cache_file=settings.cache_file,
        _env_file=None,
    )
    runner = ResearchRunner(
        topic="gearboxes",
        settings=serper_settings,
        store=RunStore(serper_settings.runs_dir),
        llm=llm,
        research_agent=ScriptedResearchAgent(),
        searcher=WebSearcher(serper_settings, client=StubSerperClient()),
    )

    events, _ = await drain(runner)
    started = events_of(events, RUN_STARTED)[0]

    assert started.data["config"]["search_provider"] == "serper"


async def test_empty_plan_does_not_deadlock(settings: Settings, agent: ScriptedResearchAgent):
    """A planner that returns nothing must still produce a report."""
    llm = FakeLLM(plans=[ResearchPlan(questions=[])], reviews=[ReviewVerdict(acceptable=True)])
    _, result = await drain(make_runner(settings, llm, agent))

    assert result is not None
    assert len(agent.questions) == 1
    assert "gearbox selection" in agent.questions[0]


async def test_failing_writer_is_reported_not_raised(
    settings: Settings, agent: ScriptedResearchAgent
):
    class ExplodingLLM(FakeLLM):
        async def astream_complete(self, prompt):
            raise RuntimeError("provider is down")

    llm = ExplodingLLM(plans=[ResearchPlan(questions=list(QUESTIONS))])
    events, result = await drain(make_runner(settings, llm, agent))

    assert result is None
    assert RUN_FAILED in kinds(events)
    failure = events_of(events, RUN_FAILED)[0].data["error"]
    assert "provider is down" in failure
    assert kinds(events)[-1] == DONE, "the stream must end cleanly even on failure"


async def test_a_failed_run_is_archived_for_later_inspection(
    settings: Settings, agent: ScriptedResearchAgent
):
    class ExplodingLLM(FakeLLM):
        async def astream_complete(self, prompt):
            raise RuntimeError("provider is down")

    llm = ExplodingLLM(plans=[ResearchPlan(questions=list(QUESTIONS))])
    runner = make_runner(settings, llm, agent)
    events, result = await drain(runner)

    store = RunStore(settings.runs_dir)
    meta = store.load_meta(runner.run_id)
    assert result is None
    assert meta.status == "failed"
    assert "provider is down" in meta.error
    assert meta.questions == list(QUESTIONS), "the plan survives the failure"
    # The archive holds every event, so the failure is reproducible offline.
    assert [event["kind"] for event in store.load_events(runner.run_id)] == kinds(events)


async def test_missing_api_key_fails_cleanly(tmp_path):
    settings = make_settings(tmp_path, groq_api_key="")
    runner = ResearchRunner(topic="anything", settings=settings)
    events, result = await drain(runner)

    assert result is None
    assert "GROQ_API_KEY" in events_of(events, RUN_FAILED)[0].data["error"]


async def test_blank_topic_is_rejected(settings: Settings):
    with pytest.raises(ValueError, match="topic is required"):
        ResearchRunner(topic="   ", settings=settings)


async def test_questions_are_deduped_and_numbering_stripped(
    settings: Settings, agent: ScriptedResearchAgent
):
    llm = FakeLLM(
        plans=[
            ResearchPlan(
                questions=[
                    "1. What is a helical gearbox used for?",
                    "  What is a helical gearbox used for?  ",
                    "- Which gearbox suits a crane?",
                ]
            )
        ],
        reviews=[ReviewVerdict(acceptable=True)],
    )
    _, result = await drain(make_runner(settings, llm, agent))

    assert result is not None
    assert result.meta.questions == [
        "What is a helical gearbox used for?",
        "Which gearbox suits a crane?",
    ]
    assert extract_question(agent.questions[0]) == result.meta.questions[0]
