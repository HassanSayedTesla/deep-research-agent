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
